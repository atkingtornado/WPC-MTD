"""Run WPC-MTD end to end against a local test archive and record everything.

Each scenario mirrors one of the production driver scripts (WPC-MTD-SCRIPTS)
but points at local paths.  Besides the normal outputs (figures, track files,
probability NetCDF) the harness records:

* ``commands.json`` - every external command the package ran, with local
  paths replaced by placeholders so runs from different trees can be diffed;
* ``capture.pkl`` - deep copies of the arrays handed to each plotting
  function (the numerical end product of the MET/MTD pipeline);
* ``timing.json`` - wall-clock time of the run.

It can drive any checkout of the package (``--src``), which is how the
refactor was regression-tested against the original code::

    python devtools/run_scenario.py --src . --label new --scenario hrrrtle \\
        --archive /Volumes/LaCie/wpcmtd-test/archive \\
        --out /Volumes/LaCie/wpcmtd-test/runs --met-path ~/opt/met/12.2.2/bin
"""

import argparse
import concurrent.futures
import copy
import dataclasses
import datetime
import importlib
import json
import os
import pathlib
import pickle
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

REGIONS = ['_ALL', '_NW', '_SW', '_NC', '_SC', '_NE', '_SE']
REGION_BOUNDS = [
    [-130, 20, -64, 52],
    [-130, 36, -100, 52],
    [-128, 26, -98, 42],
    [-112, 34, -82, 50],
    [-115, 24, -85, 40],
    [-93, 34, -63, 50],
    [-100, 22, -70, 40],
]
HRRR_TLE = ['HRRRv4_TLE_lag12', 'HRRRv4_TLE_lag06', 'HRRRv4_TLE_lag00']

COMMON = dict(
    latlon_dims=[-126, 23, -65, 50],
    init_inc=1,
    season_sub=list(range(1, 13)),
    plot_allhours=False,
    ti_thresh=0.7,
    transfer_to_prod=False,
)

SCENARIOS = {
    # thepro_hrrrtle.py
    'hrrrtle': dict(
        date=datetime.datetime(2026, 9, 27, 12),
        mtd_mode='Operational',
        load_model=HRRR_TLE,
        load_qpe=['None'],
        snow_mask=False,
        pre_acc=1.0,
        grib_path_des='MTD_QPF_HRRRTLE_EXT',
        thresh=0.1,
        end_fcst_hrs=36.0,
        conv_radius=6,
        min_volume=700,
        sigma=2,
        domain_sub=REGIONS,
        latlon_sub=REGION_BOUNDS,
        save_probs_to_file=True,
    ),
    # spt_hrrrtle_t0.1.py
    'spt_hrrrtle': dict(
        date=datetime.datetime(2026, 9, 27, 12),
        mtd_mode='Operational',
        load_model=HRRR_TLE,
        load_qpe=['None'],
        snow_mask=True,
        pre_acc=1.0,
        grib_path_des='MTD_HRRRTLE_EXT',
        thresh=0.1,
        end_fcst_hrs=36.0,
        conv_radius=6,
        min_volume=700,
        sigma=2,
        domain_sub=REGIONS,
        latlon_sub=REGION_BOUNDS,
        save_probs_to_file=False,
    ),
    # thepro_hrrr_subhourly.py
    'subhourly': dict(
        date=datetime.datetime(2026, 9, 27, 12),
        mtd_mode='Operational',
        load_model=[
            'HRRRv415min_lag00',
            'HRRRv415min_lag01',
            'HRRRv415min_lag02',
            'HRRRv415min_lag03',
        ],
        load_qpe=['None'],
        snow_mask=False,
        pre_acc=0.25,
        grib_path_des='MTD_QPF_HRRRv415min',
        thresh=0.05,
        end_fcst_hrs=15.0,
        conv_radius=2,
        min_volume=700,
        sigma=0.5,
        domain_sub=['_ALL'],
        latlon_sub=[[-130, 20, -64, 52]],
        save_probs_to_file=False,
    ),
    # thepro_hrrr.py
    'hrrr': dict(
        date=datetime.datetime(2026, 9, 26, 12),
        mtd_mode='Operational',
        load_model=['HRRRv4_TLE_lag00'],
        load_qpe=['None'],
        snow_mask=False,
        pre_acc=1.0,
        grib_path_des='MTD_QPF_HRRR_EXT',
        thresh=0.1,
        end_fcst_hrs=48.0,
        conv_radius=6,
        min_volume=1000,
        sigma=2,
        domain_sub=[],
        latlon_sub=[[]],
        save_probs_to_file=False,
    ),
    # mtd_retro_save_props_HRRR.py (single init)
    'retro': dict(
        date=datetime.datetime(2026, 9, 26, 12),
        mtd_mode='Retro',
        load_model=['HRRRv4_TLE_lag00'],
        load_qpe=['ST4_lag00'],
        snow_mask=False,
        pre_acc=1.0,
        grib_path_des='MTD_RETRO_HRRRv4_t0.1',
        thresh=0.1,
        end_fcst_hrs=48.0,
        conv_radius=6,
        min_volume=1000,
        sigma=2,
        domain_sub=[],
        latlon_sub=[[]],
        save_probs_to_file=False,
    ),
}

PLOT_FUNCS = [
    'mtd_plot_all_fcst',
    'mtd_plot_tle_fcst',
    'mtd_plot_all_snow_fcst',
    'mtd_plot_retro',
]


class Recorder:
    """Collects external commands and plot-function inputs."""

    def __init__(self, replacements):
        self.replacements = replacements
        self.commands = []
        self.captures = []

    def normalize(self, text):
        for old, new in self.replacements:
            text = text.replace(old, new)
        return text

    def add_command(self, cmd, shell):
        argv = (
            shlex.split(cmd) if isinstance(cmd, str) else [str(c) for c in cmd]
        )
        self.commands.append(
            {'argv': [self.normalize(a) for a in argv], 'shell': bool(shell)}
        )


def install_hooks(recorder, mtdmod, plotmod, serial_plots=False):
    real_run = subprocess.run

    def run(cmd, *args, **kwargs):
        recorder.add_command(cmd, kwargs.get('shell', False))
        return real_run(cmd, *args, **kwargs)

    subprocess.run = run

    real_system = os.system

    def system(cmd):
        recorder.add_command(cmd, True)
        return real_system(cmd)

    os.system = system

    parallel = getattr(mtdmod, 'plot_regions_in_parallel', None)
    if parallel is not None:
        # Refactored code: plotting runs in worker processes, so record the
        # jobs here (plot functions themselves must stay picklable).
        def record_jobs(jobs, max_workers, shared=None):
            recorder.captures.append(
                {
                    'func': 'plot_regions_in_parallel',
                    'jobs': [(f.__name__, copy.deepcopy(k)) for f, k in jobs],
                    'shared': copy.deepcopy(shared),
                }
            )
            return parallel(jobs, max_workers, shared)

        mtdmod.plot_regions_in_parallel = record_jobs

    for name in PLOT_FUNCS:
        func = getattr(mtdmod, name, None)
        if func is None or (parallel is not None and name != 'mtd_plot_retro'):
            continue

        def wrapper(*args, _func=func, _name=name, **kwargs):
            recorder.captures.append(
                {
                    'func': _name,
                    'args': copy.deepcopy(args),
                    'kwargs': copy.deepcopy(kwargs),
                }
            )
            return _func(*args, **kwargs)

        setattr(mtdmod, name, wrapper)

    # Never block on the operational "wait for late data" loop.
    mtdmod.time.sleep = lambda seconds: None

    if serial_plots:
        # The original code plots regions on a thread pool, but pyplot is not
        # thread-safe (figures get mixed up or the run crashes).  Force one
        # worker so the reference output is well defined.
        real_pool = concurrent.futures.ThreadPoolExecutor

        def one_worker_pool(*args, **kwargs):
            kwargs['max_workers'] = 1
            return real_pool(**kwargs)

        mtdmod.concurrent.futures.ThreadPoolExecutor = one_worker_pool


def run_scenario(name, args):
    os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
    cfg = dict(COMMON, **copy.deepcopy(SCENARIOS[name]))
    date = cfg.pop('date')
    out = (args.out / args.label / name).resolve()
    temp = (args.temp / name).resolve()
    for path in (out, temp):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    sys.path.insert(0, str(args.src.resolve()))
    mtdmod = importlib.import_module('wpcmtd.mtd')
    plotmod = importlib.import_module('wpcmtd.plot')

    met_path = args.met_path.expanduser().resolve()
    recorder = Recorder(
        [
            (str(temp), '{TEMP}'),
            (str(out), '{OUT}'),
            (str(args.archive.resolve()), '{ARCHIVE}'),
            (str(met_path), '{MET}'),
        ]
    )
    install_hooks(recorder, mtdmod, plotmod, serial_plots=args.serial_plots)

    fields = {f.name for f in dataclasses.fields(mtdmod.WPCMTD)}
    if 'met_version' in fields:
        cfg['met_version'] = args.met_version
    obj = mtdmod.WPCMTD(
        met_path=met_path,
        temp_dir=temp,
        fig_path=pathlib.Path(str(out / 'figs') + '/'),
        grib_path=args.archive.resolve(),
        config_path=out / 'config',
        track_path=out / 'tracks',
        beg_date=date,
        end_date=date,
        **cfg,
    )
    obj.ops_check = 0
    if args.workers and 'max_workers' in fields:
        obj.max_workers = args.workers

    prev_cwd = os.getcwd()
    os.chdir(out)  # save_probs writes into the working directory
    start = time.perf_counter()
    try:
        obj.run_mtd()
    finally:
        elapsed = time.perf_counter() - start
        os.chdir(prev_cwd)
        with open(out / 'commands.json', 'w') as fout:
            json.dump(recorder.commands, fout, indent=1)
        with open(out / 'capture.pkl', 'wb') as fout:
            pickle.dump(recorder.captures, fout)
        with open(out / 'timing.json', 'w') as fout:
            json.dump(
                {
                    'scenario': name,
                    'label': args.label,
                    'seconds': elapsed,
                    'numpy': np.__version__,
                },
                fout,
                indent=1,
            )
    print(
        f'{name}: {elapsed:.1f}s, {len(recorder.commands)} commands, '
        f'{len(recorder.captures)} plot calls'
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--src',
        type=pathlib.Path,
        default=pathlib.Path('.'),
        help='source tree containing the wpcmtd package',
    )
    parser.add_argument(
        '--label', required=True, help='run label, e.g. baseline or new'
    )
    parser.add_argument(
        '--scenario', nargs='+', choices=sorted(SCENARIOS), required=True
    )
    parser.add_argument('--archive', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument(
        '--temp',
        type=pathlib.Path,
        default=pathlib.Path.home() / '.cache' / 'wpcmtd-test' / 'tmp',
    )
    parser.add_argument('--met-path', type=pathlib.Path, required=True)
    parser.add_argument('--met-version', default='V12.2')
    parser.add_argument(
        '--workers', type=int, help='max_workers for the refactored code'
    )
    parser.add_argument(
        '--serial-plots',
        action='store_true',
        help='force single-threaded plotting in the original code',
    )
    args = parser.parse_args()
    for name in args.scenario:
        run_scenario(name, args)


if __name__ == '__main__':
    main()
