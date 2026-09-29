"""Build a small local GRIB archive for exercising WPC-MTD end to end.

Downloads only the GRIB2 messages the package needs (via byte-range
requests against the ``.idx`` inventories) and lays them out using the
same directory/file naming as the WPC archive, gzipped, so that
``wpcmtd.utils.load_data_str`` resolves them unchanged:

* HRRR hourly (APCP + CSNOW) from ``noaa-hrrr-bdp-pds`` on AWS ->
  ``HRRRv4/<YYYYMMDD>/hrrr_2d_<YYYYMMDDHH><FF>.grib2.gz``
* HRRR 15-minute APCP (sub-hourly files, split per 15-minute step) ->
  ``HRRRv4_SubHourly/<YYYYMMDD>/hrrr_2d_<YYYYMMDDHH><FF><MM>.grib2.gz``
* Stage IV hourly QPE from NCEP NOMADS ->
  ``ST4/<YYYYMM>/ST4.<YYYYMMDDHH>.01h.gz``

Example::

    python devtools/fetch_test_data.py --archive ARCHIVE_DIR \\
        --hrrr-init 2026092700 2026092706 2026092712 --hrrr-fhrs 1-36 \\
        --hrrr-subh-init 2026092709 2026092710 2026092711 2026092712 \\
        --hrrr-subh-fhrs 1-15 \\
        --st4-start 2026092613 --st4-end 2026092812
"""

import argparse
import concurrent.futures
import datetime
import gzip
import pathlib
import re
import urllib.request

HRRR_BASE = 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com'
ST4_BASE = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/pcpanl/prod'

SUBH_APCP = re.compile(r'APCP:surface:(\d+)-(\d+) min acc fcst')


def _get(url, byte_range=None, timeout=120):
    req = urllib.request.Request(url)
    if byte_range is not None:
        req.add_header('Range', f'bytes={byte_range[0]}-{byte_range[1]}')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _read_idx(url):
    """``[(start, end_or_None, line), ...]`` for a GRIB2 ``.idx`` file."""
    lines = _get(url + '.idx').decode().strip().splitlines()
    starts = [int(line.split(':')[1]) for line in lines]
    ends = [s - 1 for s in starts[1:]] + [None]
    return list(zip(starts, ends, lines, strict=True))


def _fetch_messages(url, entries):
    parts = []
    for start, end, _ in entries:
        parts.append(_get(url, (start, end if end is not None else '')))
    return b''.join(parts)


def _write_gz(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb', compresslevel=6) as fout:
        fout.write(data)


def _parse_range(text):
    lo, _, hi = text.partition('-')
    return range(int(lo), int(hi or lo) + 1)


def fetch_hrrr_hourly(archive, init, fhr, fields=('APCP', 'CSNOW')):
    ymd, hh = init[:8], init[8:]
    url = f'{HRRR_BASE}/hrrr.{ymd}/conus/hrrr.t{hh}z.wrfsfcf{fhr:02d}.grib2'
    out = archive / 'HRRRv4' / ymd / f'hrrr_2d_{init}{fhr:02d}.grib2.gz'
    if out.exists():
        return out, 'exists'
    keep = [e for e in _read_idx(url) if e[2].split(':')[3] in fields]
    _write_gz(out, _fetch_messages(url, keep))
    return out, f'{len(keep)} msgs'


def fetch_hrrr_subhourly(archive, init, fhr):
    ymd, hh = init[:8], init[8:]
    url = f'{HRRR_BASE}/hrrr.{ymd}/conus/hrrr.t{hh}z.wrfsubhf{fhr:02d}.grib2'
    written = []
    for entry in _read_idx(url):
        match = SUBH_APCP.search(entry[2])
        if not match:
            continue
        end_min = int(match.group(2))
        out = (
            archive
            / 'HRRRv4_SubHourly'
            / ymd
            / f'hrrr_2d_{init}{end_min // 60:02d}{end_min % 60:02d}.grib2.gz'
        )
        if not out.exists():
            _write_gz(out, _fetch_messages(url, [entry]))
        written.append(out)
    return written, f'{len(written)} steps'


def fetch_st4(archive, valid):
    ymd = valid.strftime('%Y%m%d')
    stamp = valid.strftime('%Y%m%d%H')
    url = f'{ST4_BASE}/pcpanl.{ymd}/st4_conus.{stamp}.01h.grb2'
    out = archive / 'ST4' / valid.strftime('%Y%m') / f'ST4.{stamp}.01h.gz'
    if out.exists():
        return out, 'exists'
    _write_gz(out, _get(url))
    return out, 'ok'


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--archive', type=pathlib.Path, required=True)
    parser.add_argument('--hrrr-init', nargs='*', default=[])
    parser.add_argument('--hrrr-fhrs', default='1-36')
    parser.add_argument('--hrrr-subh-init', nargs='*', default=[])
    parser.add_argument('--hrrr-subh-fhrs', default='1-15')
    parser.add_argument('--st4-start')
    parser.add_argument('--st4-end')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    jobs = []
    for init in args.hrrr_init:
        for fhr in _parse_range(args.hrrr_fhrs):
            jobs.append((fetch_hrrr_hourly, (args.archive, init, fhr)))
    for init in args.hrrr_subh_init:
        for fhr in _parse_range(args.hrrr_subh_fhrs):
            jobs.append((fetch_hrrr_subhourly, (args.archive, init, fhr)))
    if args.st4_start:
        valid = datetime.datetime.strptime(args.st4_start, '%Y%m%d%H')
        end = datetime.datetime.strptime(
            args.st4_end or args.st4_start, '%Y%m%d%H'
        )
        while valid <= end:
            jobs.append((fetch_st4, (args.archive, valid)))
            valid += datetime.timedelta(hours=1)

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = {
            pool.submit(func, *fargs): (func.__name__, fargs[1:])
            for func, fargs in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            name, fargs = futures[future]
            try:
                _, status = future.result()
                print(f'{name} {fargs}: {status}')
            except Exception as exc:  # report and keep going
                failures += 1
                print(f'FAILED {name} {fargs}: {exc}')
    print(f'{len(jobs) - failures}/{len(jobs)} jobs succeeded')


if __name__ == '__main__':
    main()
