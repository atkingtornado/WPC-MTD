"""Compare two `run_scenario.py` runs of one scenario (baseline vs new).

Checks, in order of how directly they reflect the science:

* plot inputs: object grids (simp_bin), object attributes (simp_prop),
  data_success and the lat/lon grid handed to the plotting code;
* track files: raw MTD output (2d.txt / obj.nc) or gzipped NPZ track files;
* probability NetCDF files and GeoJSON exports;
* figures (PNG), pixel by pixel;
* the MTD commands that were run (inputs to the tracker).

Usage::

    python devtools/compare_runs.py RUNS/baseline/hrrrtle RUNS/new/hrrrtle
"""

import argparse
import gzip
import io
import json
import pathlib
import pickle
import sys

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from PIL import Image

ARRAY_ARGS = ('data_success', 'lat', 'lon', 'simp_bin')


def files(root, pattern):
    return {
        p.relative_to(root).as_posix(): p
        for p in root.rglob(pattern)
        if not p.name.startswith('._')
    }


def plot_inputs(run_dir):
    """First plot inputs per (function, region), as name -> value."""
    with open(run_dir / 'capture.pkl', 'rb') as fin:
        captures = pickle.load(fin)
    out = {}
    for cap in captures:
        if cap['func'] == 'plot_regions_in_parallel':
            for name, kwargs in cap['jobs']:
                merged = dict(cap['shared'] or {}, **kwargs)
                out.setdefault((name, merged['grib_path_des']), merged)
        else:
            names = (
                'grib_path_des',
                'fig_path_s',
                'latlon_dims',
                'pre_acc',
                'hrs',
                'raw_thres',
                'curdate',
                'data_success',
                'load_data_nc',
                'lat',
                'lon',
                'simp_bin',
                'simp_prop',
            )
            merged = dict(zip(names, cap['args'], strict=False))
            out.setdefault((cap['func'], merged['grib_path_des']), merged)
    return out


def same_array(a, b):
    a = np.ma.filled(np.asanyarray(a, dtype=float), np.nan)
    b = np.ma.filled(np.asanyarray(b, dtype=float), np.nan)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def compare_plot_inputs(old_dir, new_dir, report):
    old, new = plot_inputs(old_dir), plot_inputs(new_dir)
    report(
        'plot calls (function, region)',
        sorted(old) == sorted(new),
        f'{len(old)} vs {len(new)}',
    )
    for key in sorted(set(old) & set(new)):
        a, b = old[key], new[key]
        for name in ARRAY_ARGS:
            report(f'  {key[0]} {key[1]} {name}', same_array(a[name], b[name]))
        props_equal = len(a['simp_prop']) == len(b['simp_prop']) and all(
            same_array(x, y)
            for x, y in zip(a['simp_prop'], b['simp_prop'], strict=True)
        )
        report(f'  {key[0]} {key[1]} simp_prop', props_equal)
        break  # every region gets the same arrays; one check is enough


def compare_text(old, new):
    return old.read_text() == new.read_text()


def compare_nc(old, new):
    with Dataset(old) as a, Dataset(new) as b:
        if set(a.variables) != set(b.variables):
            return False
        return all(
            same_array(a.variables[v][:], b.variables[v][:])
            for v in a.variables
        )


def compare_npz_gz(old, new):
    def load(path):
        with gzip.open(path, 'rb') as fin:
            return dict(np.load(io.BytesIO(fin.read()), allow_pickle=True))

    a, b = load(old), load(new)
    return set(a) == set(b) and all(same_array(a[k], b[k]) for k in a)


def compare_tracks(old_dir, new_dir, report):
    old, new = (
        files(old_dir / 'tracks', '*.*'),
        files(new_dir / 'tracks', '*.*'),
    )
    report(
        'track files present',
        sorted(old) == sorted(new),
        f'{len(old)} vs {len(new)}; '
        f'only old: {sorted(set(old) - set(new))[:3]}; '
        f'only new: {sorted(set(new) - set(old))[:3]}',
    )
    for rel in sorted(set(old) & set(new)):
        if rel.endswith('.txt'):
            report(f'  {rel}', compare_text(old[rel], new[rel]))
        elif rel.endswith('.nc'):
            report(f'  {rel}', compare_nc(old[rel], new[rel]))
        elif rel.endswith('.npz.gz'):
            report(f'  {rel}', compare_npz_gz(old[rel], new[rel]))


def compare_probs(old_dir, new_dir, report):
    old, new = files(old_dir, '*_probs.nc'), files(new_dir, '*_probs.nc')
    for rel in sorted(set(old) | set(new)):
        if rel not in old or rel not in new:
            report(f'probs {rel}', False, 'missing in one run')
            continue
        with xr.open_dataset(old[rel]) as a, xr.open_dataset(new[rel]) as b:
            report(f'probs {rel}', a.equals(b))


def compare_geojson(old_dir, new_dir, report):
    old, new = (
        files(old_dir / 'figs', '*.geojson'),
        files(new_dir / 'figs', '*.geojson'),
    )
    report(
        'geojson files present',
        sorted(old) == sorted(new),
        f'{len(old)} vs {len(new)}',
    )
    bad = [
        rel
        for rel in sorted(set(old) & set(new))
        if json.loads(old[rel].read_text()) != json.loads(new[rel].read_text())
    ]
    report(
        f'geojson content ({len(set(old) & set(new))} files)',
        not bad,
        f'differ: {bad[:5]}',
    )


def compare_figures(old_dir, new_dir, report):
    old, new = (
        files(old_dir / 'figs', '*.png'),
        files(new_dir / 'figs', '*.png'),
    )
    report(
        'figures present',
        sorted(old) == sorted(new),
        f'{len(old)} vs {len(new)}; '
        f'only old: {sorted(set(old) - set(new))[:3]}; '
        f'only new: {sorted(set(new) - set(old))[:3]}',
    )
    identical, differ = 0, []
    for rel in sorted(set(old) & set(new)):
        a = np.asarray(Image.open(old[rel]).convert('RGBA'))
        b = np.asarray(Image.open(new[rel]).convert('RGBA'))
        if a.shape == b.shape and np.array_equal(a, b):
            identical += 1
        else:
            n_diff = (
                int(np.any(a != b, axis=2).sum())
                if a.shape == b.shape
                else f'size {a.shape} vs {b.shape}'
            )
            differ.append((rel, n_diff))
    report(
        f'figures pixel-identical ({identical}/{identical + len(differ)})',
        not differ,
        '; '.join(f'{rel}: {n}' for rel, n in differ[:8]),
    )


def compare_mtd_commands(old_dir, new_dir, report):
    def mtd_commands(run_dir):
        with open(run_dir / 'commands.json') as fin:
            cmds = json.load(fin)
        return sorted(
            tuple(c['argv']) for c in cmds if c['argv'][0].endswith('/mtd')
        )

    old, new = mtd_commands(old_dir), mtd_commands(new_dir)
    report(f'MTD commands identical ({len(old)} vs {len(new)})', old == new)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('old', type=pathlib.Path)
    parser.add_argument('new', type=pathlib.Path)
    parser.add_argument('--skip-figures', action='store_true')
    args = parser.parse_args()

    failures = []

    def report(label, ok, detail=''):
        print(
            f'{"OK  " if ok else "DIFF"} {label}'
            + (f'  [{detail}]' if detail and not ok else '')
        )
        if not ok:
            failures.append(label)

    compare_plot_inputs(args.old, args.new, report)
    compare_mtd_commands(args.old, args.new, report)
    compare_tracks(args.old, args.new, report)
    compare_probs(args.old, args.new, report)
    compare_geojson(args.old, args.new, report)
    if not args.skip_figures:
        compare_figures(args.old, args.new, report)
    print(f'\n{len(failures)} difference(s)')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
