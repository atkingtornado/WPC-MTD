"""MET Mode Time Domain (MTD) driver for WPC object-based QPF products.

`WPCMTD` converts model (and optionally analysis) precipitation to MET
NetCDF with ``pcp_combine``, regrids it with ``regrid_data_plane``, tracks
precipitation objects through time with ``mtd``, and turns the MTD output
into track files and figures.
"""

import concurrent.futures
import contextlib
import dataclasses
import datetime
import glob
import gzip
import io
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import warnings
from typing import ClassVar

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter

from .plot import (
    animate_all_fcst,
    animate_all_snow_fcst,
    mtd_plot_all_fcst,
    mtd_plot_all_snow_fcst,
    mtd_plot_retro,
    mtd_plot_tle_fcst,
    plot_regions_in_parallel,
    split_hours,
)
from .utils import (
    MET_VERSION,
    adjust_date_range,
    gen_mtdconfig,
    gen_mtdconfig_15m,
    load_data_str,
)

logger = logging.getLogger(__name__)

# MET picks up the extra GRIB2 table (MRMS fields) from this variable.
os.environ['MET_GRIB_TABLES'] = str(
    pathlib.Path(__file__).parent.resolve()
    / 'met_grib_tables'
    / 'grib2_mrms.txt'
)

# NAM nest data only has hourly accumulations from this date on.
NAM_HOURLY_START = datetime.datetime(2017, 3, 21, 6)

# Default cap on concurrent MET processes / plotting workers.
DEFAULT_MAX_WORKERS = 8

# Columns read from the MTD 2D attribute file, in output-array row order.
MTD_2D_COLUMNS = (
    'OBJECT_ID',
    'OBJECT_CAT',
    'TIME_INDEX',
    'AREA',
    'CENTROID_LAT',
    'CENTROID_LON',
    'AXIS_ANG',
)


class PerfTimer:
    """Accumulate wall-clock time per named section (thread-safe)."""

    def __init__(self):
        self.timers = {}
        self.counts = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger('wpcmtd.perf')

    def reset(self):
        with self._lock:
            self.timers.clear()
            self.counts.clear()

    def add(self, name, seconds):
        with self._lock:
            self.timers[name] = self.timers.get(name, 0.0) + seconds
            self.counts[name] = self.counts.get(name, 0) + 1

    @contextlib.contextmanager
    def section(self, name):
        """Time the body of a ``with`` block under ``name``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    def summary(self):
        """Log a table of all accumulated timers.

        Sections that run inside worker pools (e.g. ``regrid_data_plane``)
        accumulate the time of every worker, so they can exceed wall time.
        """
        if not self.timers:
            return
        log = self.logger.info
        log('')
        log('=' * 70)
        log('PERFORMANCE TIMING SUMMARY')
        log('=' * 70)
        log(f'{"Category":<35} {"Total (s)":>10} {"Count":>7} {"Avg (s)":>10}')
        log('-' * 70)
        for name in sorted(self.timers):
            total = self.timers[name]
            count = self.counts[name]
            log(
                f'{name:<35} {total:>10.2f} {count:>7d} {total / count:>10.2f}'
            )
        log('=' * 70)


# Module-level performance timer instance
perf = PerfTimer()


def _run_cmd(args, cwd=None):
    """Run a program (argument list, no shell); return its exit code."""
    return subprocess.run(
        [str(arg) for arg in args], cwd=cwd, check=False
    ).returncode


def _run_shell(cmd):
    """Run a shell command line (remote transfers that rely on globbing)."""
    return subprocess.run(cmd, shell=True, check=False).returncode


def _configure_logging():
    """Log to stdout, which the cron drivers capture."""
    logging.basicConfig(
        level=logging.INFO, format='%(message)s', stream=sys.stdout
    )


def _accum_str(hours):
    """MET accumulation string, e.g. 1 -> 'A010000' and 0.25 -> 'A001500'."""
    seconds = datetime.timedelta(hours=hours).seconds
    return f'A{seconds // 3600:02d}{(seconds % 3600) // 60:02d}00'


def _run_cycle(name, pre_acc, curr_date):
    """
    Return ``(run_times, acc_int)`` for a model/analysis name.

    ``run_times`` are the UTC hours (0-24 inclusive) at which the data are
    initialized/available and ``acc_int`` is the accumulation interval (hours)
    of the stored precipitation.  The first matching rule wins, so e.g.
    HRRRe60min follows the HRRR rule.
    """
    if 'ST4' in name or 'MRMS_' in name:
        if pre_acc in (
            6,
            12,
            18,
            24,
        ):  # 24 h of hourly data takes too long operationally
            return np.linspace(0, 24, 5), 6
        return np.linspace(0, 24, 25), 1
    if 'MRMS15min' in name:
        return np.linspace(0, 24, 4 * 24 + 1), 15.0 / 60.0
    if 'MRMS2min' in name:
        return np.linspace(0, 24, 30 * 24 + 1), 2.0 / 60.0
    if 'NAM' in name:
        if curr_date < NAM_HOURLY_START:
            raise ValueError(
                'Specified Interval of 1 Hour Impossible with NAM before '
                '2017032106'
            )
        return np.linspace(0, 24, 5), 1
    if 'HIRESW' in name:
        return np.linspace(0, 24, 3), 1
    if 'HRRR' in name and '15min' not in name:
        return np.linspace(0, 24, 25), 1
    if 'NSSL-OP' in name:
        return np.linspace(0, 24, 3), 1
    if 'NSSL-ARW' in name or 'NSSL-NMB' in name:
        return np.linspace(0, 24, 2), 1
    if 'NEWSe5min' in name:
        return np.linspace(0, 24, 25), 1.0 / 12.0
    if 'NEWSe60min' in name:
        return np.linspace(0, 24, 25), 1
    if 'FV3LAM' in name:
        return np.linspace(0, 24, 25), 1
    if 'HRRRv415min' in name:
        return np.linspace(0, 24, 25), 1.0 / 4.0
    raise ValueError(f'Unrecognized model/analysis name: {name}')


def _domain_slices(lat, lon, latlon_dims, pad=3):
    """
    Row/column slices of the sub-grid covering ``latlon_dims`` +/- ``pad``
    degrees.

    Parameters
    ----------
    lat, lon : numpy.ndarray
        2-D grid coordinates.
    latlon_dims : list
        [WLON, SLAT, ELON, NLAT].
    """
    inside = (
        (lat >= latlon_dims[1] - pad)
        & (lat < latlon_dims[3] + pad)
        & (lon >= latlon_dims[0] - pad)
        & (lon < latlon_dims[2] + pad)
    ).astype(int)
    x_beg = np.argmax(np.diff(inside, axis=1) == 1, axis=1)
    x_end = np.argmax(np.diff(inside, axis=1) == -1, axis=1)
    y_beg = np.argmax(np.diff(inside, axis=0) == 1, axis=0)
    y_end = np.argmax(np.diff(inside, axis=0) == -1, axis=0)
    x_beg = np.min(x_beg[x_beg > 0])
    x_end = np.max(x_end[x_end > 0])
    y_beg = np.min(y_beg[y_beg > 0])
    y_end = np.max(y_end[y_end > 0])
    return slice(y_beg, y_end), slice(x_beg, x_end)


def _all_missing(values):
    """True when ``values`` is empty or entirely NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        frac = np.mean(np.isnan(values))
    return frac == 1 or np.isnan(frac) == 1


def _gunzip(src, dst):
    """Decompress ``src`` to ``dst``; log and skip a missing source."""
    try:
        with gzip.open(src, 'rb') as fin, open(dst, 'wb') as fout:
            shutil.copyfileobj(fin, fout, length=1024 * 1024)
    except FileNotFoundError as err:
        logger.warning(f'WARNING: grib file not found, skipping: {err}')


def _move(src, dst, what):
    """Move a file, logging (not raising) when the source does not exist."""
    try:
        shutil.move(src, dst)
    except FileNotFoundError as err:
        logger.warning(f'WARNING: {what} missing: {err}')


def _remove_matching(pattern):
    """Delete files and directories matching a glob pattern."""
    for match in glob.glob(pattern):
        if os.path.isdir(match):
            shutil.rmtree(match)
        else:
            os.remove(match)


def _read_mtd_2d(path):
    """
    Parse an MTD 2D attribute file.

    The file lists simple forecast objects (``F001``...), then simple
    observed objects (``O001``...), then matched cluster pairs (``CF001``,
    ``CO001``...).  Object and cluster IDs are returned as numbers, negative
    for observations.

    Returns
    -------
    simp_prop, pair_prop : numpy.ndarray
        ``(10, n)`` arrays whose rows are OBJECT_ID, OBJECT_CAT, TIME_INDEX,
        AREA, CENTROID_LAT, CENTROID_LON, AXIS_ANG and three intensity rows
        (10th/50th/90th percentile) left as NaN for the caller to fill.
    """
    with open(path) as fin:
        header = fin.readline().split()
        rows = [line.split() for line in fin]
    cols = [header.index(name) for name in MTD_2D_COLUMNS]

    simp, pair = [], []
    in_pairs = False
    prev_id = ''
    for fields in rows:
        if not fields:
            prev_id = ''
            continue
        obj_id = fields[cols[0]]
        # Cluster pairs start when the listing goes from observed back to
        # forecast objects.
        if 'F' in obj_id and 'O' in prev_id:
            in_pairs = True
        prev_id = obj_id
        if 'F' in obj_id:
            sign, prefix = 1.0, 'F'
        elif 'O' in obj_id:
            sign, prefix = -1.0, 'O'
        else:
            continue

        cat = fields[cols[1]]
        values = [
            sign * float(obj_id.lstrip('C').replace(prefix, '')),
            np.nan if 'NA' in cat else sign * float(cat[2:]),
        ]
        values += [float(fields[col]) for col in cols[2:]]
        (pair if in_pairs else simp).append(values)

    def as_array(table):
        out = np.full((10, len(table)), np.nan)
        if table:
            out[:7] = np.array(table, dtype=float).T
        return out

    return as_array(simp), as_array(pair)


def _fill_intensity(
    prop, obs_ids, obs_grid, mod_ids, mod_grid, obs_90th_ignores_nan
):
    """
    Fill rows 7-9 of ``prop`` with the 10/50/90th percentile of each object's
    raw precipitation (inches).  Observed objects have negative IDs.
    """
    for col in range(prop.shape[1]):
        obj_id = int(prop[0, col])
        try:
            t_idx = int(prop[2, col])
        except ValueError:
            prop[7:10, col] = np.nan
            continue
        ids, grid = (obs_ids, obs_grid) if obj_id < 0 else (mod_ids, mod_grid)
        try:
            values = grid[t_idx, ids[t_idx, :, :] == abs(obj_id)]
            prop[7, col] = np.round(np.nanpercentile(values, 10), 3)
            prop[8, col] = np.round(np.nanpercentile(values, 50), 3)
            if obj_id < 0 and not obs_90th_ignores_nan:
                prop[9, col] = np.round(np.percentile(values, 90), 3)
            else:
                prop[9, col] = np.round(np.nanpercentile(values, 90), 3)
        except (ValueError, IndexError, TypeError):
            prop[7:10, col] = np.nan


@dataclasses.dataclass
class _Step:
    """One verification time of one model/analysis."""

    index: int  # position in the model's list of hours
    fcst_hr: float  # verification hour (hours after the cycle time)
    beyond: bool  # past the model's last forecast hour; nothing to load
    nc: str  # regridded MET NetCDF file for this hour
    nc_prev: str  # Retro: matching model file ('null' otherwise)
    raw: str = ''  # pcp_combine output before regridding
    sources: list = dataclasses.field(
        default_factory=list
    )  # archive files (no .gz)
    ahead: datetime.datetime | None = None  # end of the accumulation


@dataclasses.dataclass
class _ModelRun:
    """Everything needed to preprocess and track one model/analysis."""

    index: int
    name: str  # e.g. 'HRRRv4_TLE_lag06'
    temp_dir: pathlib.Path  # working directory for this model
    hrs_all: np.ndarray  # verification hours
    steps: list
    grib_names: list  # decompressed GRIB files, in load order
    last_fcst_hr_str: str  # forecast hour of the last step (for file names)
    success: np.ndarray = (
        None  # per step: 0 = ok, nonzero = failed, NaN = beyond range
    )

    @property
    def nc_name(self):
        return self.name + '.nc'

    @property
    def grib_dir(self):
        # pcp_combine scans this directory and chokes on anything that is not
        # a data file (including sub-directories), so it holds only GRIB input.
        return self.temp_dir / 'grib'

    @property
    def data_name_nc(self):
        return np.array([step.nc for step in self.steps])

    @property
    def data_name_nc_prev(self):
        return np.array([step.nc_prev for step in self.steps])


@dataclasses.dataclass
class WPCMTD:
    """
    Configure and run MET Mode Time Domain (MTD) verification/tracking.

    ``mtd_mode`` is 'Operational' (track the latest model cycle(s) and make
    figures) or 'Retro' (pair one model with QPE over a date range and save
    the object attributes as track files).
    """

    met_path: pathlib.Path  # Location of MET executables
    temp_dir: pathlib.Path  # Location where all temporary files are stored
    fig_path: pathlib.Path  # Location for figure generation
    grib_path: pathlib.Path  # Location of model GRIB files
    config_path: pathlib.Path  # Location of MET config files
    track_path: pathlib.Path  # Location to store NPZ track files
    latlon_dims: list  # Lat/lon bounds for plotting [WLON,SLAT,ELON,NLAT]
    beg_date: datetime.datetime  # Beginning time of retrospective runs
    end_date: datetime.datetime  # Ending time of retrospective runs
    init_inc: int  # Hours between model initializations
    mtd_mode: str  # 'Operational', 'Retro', or 'Both'
    load_model: (
        list  # Models to load, e.g. ['HRRRv4_TLE_lag12', 'HRRRv4_TLE_lag06']
    )
    load_qpe: list  # ['ST4_lag00'], ['MRMS_lag00'] or ['None']
    snow_mask: (
        bool  # Keep only regions that are snow (True) or no masking (False)
    )
    pre_acc: float  # Precipitation accumulation interval (hours)
    season_sub: list  # Months of retro data to keep
    grib_path_des: str  # Label for temporary directories and output files
    thresh: float  # Precip. threshold for defining objects (inches)
    end_fcst_hrs: float  # Last forecast hour to be considered
    conv_radius: int  # Radius of the smoothing (grid squares)
    min_volume: int  # Volume threshold (grid squares) to keep an object
    ti_thresh: float  # Total interest threshold for determining matches
    plot_allhours: bool  # Plot every track figure hour-by-hour (Retro)
    sigma: int  # Gaussian filter (grid points) for smoothing probabilities
    domain_sub: list  # Sub-domain names for plotting, e.g. ['_ALL', '_NW']
    latlon_sub: list  # Sub-domain bounds for plotting
    transfer_to_prod: bool  # Copy figures to the production web servers
    save_probs_to_file: bool  # Save hourly object probabilities to NetCDF

    # Optional keyword-only settings.
    met_version: str = dataclasses.field(default=MET_VERSION, kw_only=True)
    max_workers: int | None = dataclasses.field(default=None, kw_only=True)

    # Seconds to keep waiting for late operational data.
    ops_check: ClassVar[float] = 60 * 60 * 0.5
    start_hrs: ClassVar[float] = 0.0
    # Polygon file for gen_vx_mask regional masking (Retro); None disables it.
    reg_mask_file: ClassVar[str | None] = None

    def __post_init__(self):
        # MTD names its output files to the minute, so seconds in the cycle
        # time (e.g. a driver's ``now() - 2 h``) would break the file names.
        self.beg_date, self.end_date = adjust_date_range(
            self.beg_date.replace(second=0, microsecond=0),
            self.end_date.replace(second=0, microsecond=0),
            self.init_inc,
        )

        if self.mtd_mode == 'Operational':
            # No verification and only one cycle in operational mode.
            if self.load_qpe[0] != 'None':
                raise ValueError(
                    "If mtd_mode = 'Operational' then load_qpe = 'None'"
                )
        elif self.mtd_mode == 'Retro':
            self.fig_path = str(self.fig_path) + 'RETRO/'
            if len(self.load_model) > 1:
                raise ValueError(
                    "If mtd_mode = 'Retro' then only one model specification "
                    'at a time'
                )
        elif self.mtd_mode == 'Both':
            first = self.load_model[0][: self.load_model[0].find('TLE')]
            if any(
                model[: model.find('TLE')] != first
                for model in self.load_model
            ):
                raise ValueError(
                    "'load_model' must be a single model or a TLE."
                )

        if 'NEWSe' in ''.join(self.load_model) and self.snow_mask:
            raise ValueError(
                'NEWSe does not contain the appropriate snow mask information'
            )

        if (
            'NEWSe' in ''.join(self.load_model) or 'MRMS' in self.load_qpe
        ) and self.pre_acc > 1:
            raise ValueError(
                'NEWSe or MRMS data can only Have a precipitation '
                'accumulation '
                'interval of 1 hour or less'
            )

        # Keep SPT (snow) track files apart from the QPF ones.
        if self.snow_mask:
            self.track_path = self.track_path / 'SPT'

        self._init_state()

    def _init_state(self):
        """Per-cycle state, filled in by setup_data()."""
        self._fig_root = self.fig_path
        self.grib_path_temp = None
        self.lat = None
        self.lon = None
        self.offset_fcsthr = []
        self.acc_int = []
        self.init_yrmondayhr = []
        self._static_grid = None
        self._state_lock = threading.Lock()

    @property
    def load_data(self):
        """Models followed by the QPE (if any)."""
        if self.load_qpe[0].lower() != 'none':
            return list(self.load_model) + [self.load_qpe[0]]
        return list(self.load_model)

    @property
    def ismodel(self):
        """1.0 for each model, 0.0 for each analysis (ST4/MRMS)."""
        return np.array(
            [
                0.0 if ('ST4' in name or 'MRMS' in name) else 1.0
                for name in self.load_data
            ]
        )

    @property
    def list_date(self):
        """Model initializations (cycle times) to process."""
        if self.mtd_mode == 'Operational':
            return np.array([self.beg_date])
        delta = self.end_date - self.beg_date
        total_incs = int(
            ((delta.days * 24) + (delta.seconds / 3600) - 1) / self.init_inc
        )
        dates = [
            self.beg_date + datetime.timedelta(hours=i * self.init_inc)
            for i in range(total_incs + 1)
        ]
        return np.array(
            [date for date in dates if date.month in self.season_sub]
        )

    @property
    def hrs(self):
        """Verification hours (end of each accumulation period)."""
        hrs = np.arange(0.0, self.end_fcst_hrs + 0.01, self.pre_acc)
        return hrs[hrs > 0]

    @property
    def n_workers(self):
        """Number of concurrent MET processes / plotting workers."""
        if self.max_workers:
            return max(1, int(self.max_workers))
        return max(1, min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1))

    def setup_data(self, curr_date):
        """
        Determine the model cycles to load for ``curr_date`` and prepare the
        working directories and analysis grid.

        When adding another model/ensemble member, make sure its name matches
        a rule in `_run_cycle` (run times and accumulation interval) and in
        `wpcmtd.utils.load_data_str` (file names).

        Parameters
        ----------
        curr_date : datetime.datetime
            Cycle (verification start) time.
        """
        yrmondayhr = curr_date.strftime('%Y%m%d%H')

        # Most recent available run of each model given its lag.
        self.init_yrmondayhr = []
        self.offset_fcsthr = []
        self.acc_int = []
        for name in self.load_data:
            run_times, acc_int = _run_cycle(name, self.pre_acc, curr_date)
            self.acc_int.append(acc_int)

            lag = float(name[name.index('lag') + 3 :])
            if lag % (24.0 / (float(len(run_times)) - 1)) != 0:
                lag = 0  # lag does not match the run frequency; ignore it
            init_offset = (
                curr_date.hour
                - run_times[np.argmax(run_times > curr_date.hour) - 1]
            )
            self.offset_fcsthr.append(init_offset + lag)
            init_date = curr_date - datetime.timedelta(
                hours=int(self.offset_fcsthr[-1])
            )
            self.init_yrmondayhr.append(init_date.strftime('%Y%m%d%H'))

        # Temporary directory for this cycle, emptied of any leftovers.
        self.grib_path_temp = pathlib.Path(
            self.temp_dir,
            f'{yrmondayhr}_p_{self.pre_acc}_t{self.thresh}_'
            f'{self.grib_path_des}',
        )
        self.grib_path_temp.mkdir(parents=True, exist_ok=True)
        for path in self.grib_path_temp.glob('**/*'):
            if path.is_file():
                path.unlink()

        self.fig_path = pathlib.Path(self._fig_root, yrmondayhr)
        self.fig_path.mkdir(parents=True, exist_ok=True)
        self.track_path.mkdir(parents=True, exist_ok=True)

        # Analysis grid, cropped to the domain of interest.
        if self._static_grid is None:
            with Dataset(
                pathlib.Path(self.grib_path, 'static', 'STATIC.nc'), 'r'
            ) as nc:
                self._static_grid = (
                    nc.variables['lat'][:],
                    nc.variables['lon'][:],
                )
        lat, lon = self._static_grid
        rows, cols = _domain_slices(lat, lon, self.latlon_dims)
        self.lat = lat[rows, cols]
        self.lon = lon[rows, cols]

    # ------------------------------------------------------------------
    # MET preprocessing
    # ------------------------------------------------------------------

    def pcp_combine(self, run, step, apcp_str_beg, apcp_str_end):
        """
        Sum/convert one verification time of one model to MET NetCDF with
        ``pcp_combine`` (the first MET step).

        Parameters
        ----------
        run : _ModelRun
            Model being processed.
        step : _Step
            Verification time being processed.
        apcp_str_beg, apcp_str_end : str
            Accumulation of the source data and of the target interval,
            e.g. 'A010000'.
        """
        model = run.index
        name = run.name
        fcst_hr = step.fcst_hr
        count = step.index
        temp = str(run.grib_dir)
        grib = run.grib_names
        pcp = f'{self.met_path}/pcp_combine'
        init = self.init_yrmondayhr[model]

        beg = (
            '00000000_000000'
            if 'ST4' in name
            else f'{init[:-2]}_{init[-2:]}0000'
        )
        end = f'{step.ahead:%Y%m%d}_{step.ahead:%H%M}00'
        name_arg = ['-name', apcp_str_end]
        sum_args = [
            pcp,
            '-sum',
            beg,
            apcp_str_beg[1:],
            end,
            apcp_str_end[1:],
            step.raw,
        ]

        # The NAM nest stores 1, 2 and 3 h buckets, so hourly totals need
        # differencing.
        if 'HIRESNAM' in name and self.pre_acc == 1:
            if (fcst_hr - 1) % 3 == 0:  # 1 h bucket
                args = sum_args + ['-pcpdir', f'{temp}/'] + name_arg
            elif (fcst_hr - 2) % 3 == 0:  # 2 h bucket minus 1 h bucket
                args = [
                    pcp,
                    '-subtract',
                    f'{temp}/{grib[count]}',
                    '020000',
                    f'{temp}/{grib[count - 1]}',
                    '010000',
                    step.raw,
                ] + name_arg
            else:  # 3 h bucket minus 2 h bucket
                args = [
                    pcp,
                    '-subtract',
                    f'{temp}/{grib[count]}',
                    '030000',
                    f'{temp}/{grib[count - 1]}',
                    '020000',
                    step.raw,
                ] + name_arg
        elif 'HIRESNAM' in name and self.pre_acc % 3 == 0:
            args = [
                pcp,
                '-sum',
                beg,
                '030000',
                end,
                apcp_str_end[1:],
                step.raw,
                '-pcpdir',
                f'{temp}/',
            ] + name_arg
        elif 'HIRESNAM' in name and self.pre_acc % 3 == 1:
            raise ValueError(
                'NAM pre_acc variable can only be 1 or divisible by 3'
            )
        elif 'NEWSe60min' in name:
            # Only the first model hour can be used directly; later hours
            # would need differencing, which is not implemented.
            if fcst_hr + self.offset_fcsthr[model] != 1:
                return
            args = sum_args + name_arg
        elif 'NEWSe5min' in name:
            args = (
                sum_args
                + ['-field', f'name="{apcp_str_beg}"; level="Surface";']
                + name_arg
            )
        elif 'MRMS_' in name or 'MRMS15min_' in name:
            field = (
                'name="MultiSensor_QPE_01H_Pass2" ; level="L0" ; '
                if 'MRMS_' in name
                else 'name="RadarOnly_QPE_15M" ; level="L0" ; '
            )
            per_step = self.pre_acc / self.acc_int[model]
            args = [pcp, '-add']
            for i in range(int(count * per_step), int((count + 1) * per_step)):
                args += [grib[i], field]
            args += [step.raw] + name_arg
            if 'MRMS_' in name:
                args += ['-v', '3']
        elif 'HRRRe' in name:
            if self.pre_acc != 1:
                raise ValueError('HRRRe pre_acc variable can only be set to 1')
            # HRRRe accumulates from initialization: difference hours.
            if fcst_hr == 1:
                args = sum_args + ['-pcpdir', f'{temp}/'] + name_arg
            else:
                args = [
                    pcp,
                    '-subtract',
                    f'{temp}/{grib[count]}',
                    f'{count + 1:02d}0000',
                    f'{temp}/{grib[count - 1]}',
                    f'{count:02d}0000',
                    step.raw,
                ] + name_arg
        else:
            args = sum_args + name_arg

        _run_cmd(args, cwd=temp)

    def apply_snow_mask(self, run, step, prev_grib_names, apcp_str_end):
        """
        Zero out precipitation that is not snow, using the model's categorical
        snow (CSNOW), the NSSL snow-water equivalent change (WEASD) or the MRMS
        precipitation flag.

        Parameters
        ----------
        run : _ModelRun
            Model being processed.
        step : _Step
            Verification time being processed.
        prev_grib_names : list
            GRIB files of the previously processed model (ST4 has no snow
            information, so it is masked with the model's field).
        apcp_str_end : str
            Accumulation field name, e.g. 'A010000'.
        """
        if not self.snow_mask:
            return

        name = run.name
        count = step.index
        temp = str(run.grib_dir)
        out_dir = str(run.temp_dir)
        base_temp = str(self.grib_path_temp)
        grib = run.grib_names

        if 'NSSL' in name:
            masking_str, masking_val = 'WEASD', 'le0'
        elif 'MRMS' in name:
            masking_str, masking_val = 'PrecipFlag', 'ne3'
        else:
            masking_str, masking_val = 'CSNOW', 'le0'

        # ST4 has no snow information, so reuse the model's mask field.
        # NOTE: load_qpe is a list, so "'ST4' in self.load_qpe" only matches
        # an entry that is exactly 'ST4'.
        if 'ST4' not in name and 'ST4' not in self.load_qpe:
            mask_file_in = f'{temp}/{grib[count]}'
            mask_file_ou = f'{out_dir}/{grib[count]}.nc'
        elif 'ST4' not in name and 'ST4' in self.load_qpe:
            mask_file_in = f'{temp}/{grib[count]}'
            mask_file_ou = f'{out_dir}/{grib[count]}.nc'
            pathlib.Path(f'{base_temp}_2').mkdir(exist_ok=True)
            try:
                shutil.copy2(mask_file_in, f'{base_temp}_2/{grib[count]}')
            except FileNotFoundError as err:
                logger.warning(
                    f'WARNING: snow_mask copy failed (missing file): {err}'
                )
        else:
            mask_file_in = f'{base_temp}_2/{prev_grib_names[count]}'
            mask_file_ou = f'{out_dir}/{prev_grib_names[count]}.nc'

        met = str(self.met_path)
        _run_cmd(
            [
                f'{met}/regrid_data_plane',
                mask_file_in,
                f'{self.grib_path}/temp/ensmean_sample_NODELETE',
                mask_file_ou,
                '-field',
                f'name="{masking_str}"; level="L0";',
                '-name',
                apcp_str_end,
            ]
        )

        # WEASD is cumulative, so difference it with the previous hour.
        if masking_str == 'WEASD' and count > 0:
            mask_file_ou_prev = f'{out_dir}/{grib[count - 1]}.nc'
            _run_cmd(
                [
                    f'{met}/pcp_combine',
                    '-subtract',
                    mask_file_ou,
                    'name="WEASD_L0"; level="L0";',
                    mask_file_ou_prev,
                    'name="WEASD_L0"; level="L0";',
                    mask_file_ou + '2',
                ]
            )
        else:
            try:
                shutil.copy2(mask_file_ou, mask_file_ou + '2')
            except FileNotFoundError as err:
                logger.warning(
                    f'WARNING: snow_mask regrid output missing: {err}'
                )

        field = f'name="{apcp_str_end}"; level="(*,*)";'
        _run_cmd(
            [
                f'{met}/gen_vx_mask',
                step.nc,
                mask_file_ou + '2',
                step.nc + '_2',
                '-type',
                'data',
                '-input_field',
                field,
                '-mask_field',
                field,
                '-thresh',
                masking_val,
                '-value',
                '0',
                '-name',
                apcp_str_end,
            ]
        )
        _move(step.nc + '_2', step.nc, 'snow_mask result')

    def apply_reg_mask(self, run, step, apcp_str_end):
        """
        Mask both the model and the analysis to the polygon in
        ``reg_mask_file`` (Retro mode with ST4/MRMS only).
        """
        if not self.reg_mask_file:
            return
        if not (self.mtd_mode == 'Retro' and self.load_qpe[0] in run.name):
            return
        if 'MRMS' not in self.load_qpe[0] and 'ST4' not in self.load_qpe[0]:
            return
        for path in (step.nc_prev, step.nc):
            _run_cmd(
                [
                    f'{self.met_path}/gen_vx_mask',
                    '-type',
                    'poly',
                    path,
                    self.reg_mask_file,
                    path + '2',
                    '-complement',
                    '-input_field',
                    f'name="{apcp_str_end}" ; level="(*,*)" ;',
                    '-value',
                    '-9999',
                    '-name',
                    apcp_str_end,
                ]
            )
            _move(path + '2', path, 'reg_mask file')

    def _plan_model(self, curr_date, model, temp_dir, retro_model_dir):
        """Work out (without I/O) the files each verification time needs."""
        name = self.load_data[model]
        stem = name
        offset = self.offset_fcsthr[model]
        acc_int = self.acc_int[model]
        init = self.init_yrmondayhr[model]

        mod_lag = int(name[name.find('lag') + 3 : name.find('lag') + 5])
        # NEWSe 60 min files only store accumulated precip, so lagged runs
        # also need the previous hour.
        first_hr = (
            self.hrs[0] - 1
            if ('NEWSe60min' in name and mod_lag > 0)
            else self.hrs[0]
        )
        hrs_all = np.arange(
            first_hr, self.hrs[-1] + self.pre_acc, self.pre_acc
        )

        retro_pair = (
            self.mtd_mode == 'Retro' and self.load_model[0] not in name
        )
        steps, grib_names = [], []
        last_fcst_hr_str = ''
        n_beyond = 0
        for index, fcst_hr in enumerate(hrs_all):
            last_fcst_hr_str = f'{int(fcst_hr + offset):02d}'
            fcst_hr_load = float(fcst_hr) + offset
            hr_str = f'{int(fcst_hr_load):02d}'
            min_str = (
                f'{int(round((fcst_hr_load - int(fcst_hr_load)) * 60)):02d}'
            )

            past_end = fcst_hr_load > self.end_fcst_hrs
            sources = []
            ahead = None
            # Files making up the accumulation ending at this hour.
            for sum_hr in np.arange(
                self.pre_acc - acc_int, -acc_int, -acc_int
            ):
                fcst_hr_load = (
                    float(fcst_hr) + offset - float(round(sum_hr, 2))
                )
                hr_str = f'{int(fcst_hr_load):02d}'
                frac = fcst_hr_load - int(fcst_hr_load)
                min_str = f'{int(round(frac * 60)):02d}'
                ahead = curr_date + datetime.timedelta(
                    hours=fcst_hr_load - offset
                )
                sources += load_data_str(
                    self.grib_path, name, init, acc_int, hr_str, min_str
                )
            # A lagged member reaches past end_fcst_hrs.  Use those hours
            # when the run went that far (e.g. HRRR's 48 h synoptic runs);
            # otherwise the hour is beyond the model's range and is skipped
            # without waiting, since the files will never exist.
            beyond = past_end and not all(
                os.path.exists(src + '.gz') for src in sources
            )
            if beyond:
                sources, ahead = [], None
                n_beyond += 1
            grib_names += [os.path.basename(src) for src in sources]

            suffix = f'_f{hr_str}{min_str}.nc'
            steps.append(
                _Step(
                    index=index,
                    fcst_hr=fcst_hr,
                    beyond=beyond,
                    nc=f'{temp_dir}/{stem}{suffix}',
                    nc_prev=f'{retro_model_dir}/{self.load_data[0]}{suffix}'
                    if retro_pair
                    else 'null',
                    raw=f'{temp_dir}/{stem}{suffix[:-3]}_pcp.nc',
                    sources=sources,
                    ahead=ahead,
                )
            )

        if n_beyond:
            logger.info(
                f'INFO: Model {name} (offset {offset}h) will skip '
                f'{n_beyond} hrs beyond its last forecast hour'
            )
        return _ModelRun(
            index=model,
            name=name,
            temp_dir=temp_dir,
            hrs_all=hrs_all,
            steps=steps,
            grib_names=grib_names,
            last_fcst_hr_str=last_fcst_hr_str,
        )

    def _needs_serial_steps(self):
        """Snow masking with WEASD (NSSL) or ST4 chains steps together."""
        return self.snow_mask and any(
            'NSSL' in name or 'ST4' in name for name in self.load_data
        )

    def _process_step(
        self, run, step, prev_grib_names, apcp_str_beg, apcp_str_end
    ):
        """pcp_combine + regrid (+ masks) for one time; return exit code."""
        with perf.section('pcp_combine'):
            self.pcp_combine(run, step, apcp_str_beg, apcp_str_end)

        # The NEWSe (WoF) domain moves: take the plotting domain from the data.
        if (
            self.mtd_mode == 'Operational'
            and 'NEWSe' in self.load_model[run.index]
        ):
            try:
                with Dataset(step.raw, 'r') as nc:
                    lon = np.array(nc.variables['lon'][:])
                    lat = np.array(nc.variables['lat'][:])
                with self._state_lock:
                    self.latlon_sub = [
                        [
                            np.nanmin(lon) - 0.5,
                            np.nanmin(lat) - 0.1,
                            np.nanmax(lon),
                            np.nanmax(lat) + 0.5,
                        ]
                    ]
                    self.domain_sub = ['ALL']
            except (OSError, KeyError):
                pass

        with perf.section('regrid_data_plane'):
            code = _run_cmd(
                [
                    f'{self.met_path}/regrid_data_plane',
                    step.raw,
                    f'{self.grib_path}/temp/ensmean_sample_NODELETE',
                    step.nc,
                    '-field',
                    f'name="{apcp_str_end}"; level="{apcp_str_end}";',
                    '-name',
                    apcp_str_end,
                ]
            )
        if os.path.exists(step.raw):
            os.remove(step.raw)

        with perf.section('snow_mask'):
            self.apply_snow_mask(run, step, prev_grib_names, apcp_str_end)
        self.apply_reg_mask(run, step, apcp_str_end)
        return code

    def _preprocess_model(
        self, run, prev_run, pool, start_time, apcp_str_beg, apcp_str_end
    ):
        """
        Decompress, combine and regrid every verification time of one model.

        In operational mode, keep retrying (every 2 minutes, for up to
        ``ops_check`` seconds after the run started) while data are missing.
        """
        run.grib_dir.mkdir(parents=True, exist_ok=True)
        prev_grib_names = prev_run.grib_names if prev_run is not None else []
        todo = [step for step in run.steps if not step.beyond]

        while True:
            run.success = np.ones(len(run.steps))
            for step in run.steps:
                if step.beyond:
                    # Past the model's range: NaN (not 1) so the retry logic
                    # below does not wait for data that will never exist.
                    run.success[step.index] = np.nan

            with perf.section('file_copy'):
                sources = {
                    src: run.grib_dir / os.path.basename(src)
                    for step in todo
                    for src in step.sources
                }
                list(
                    pool.map(
                        lambda item: _gunzip(item[0] + '.gz', item[1]),
                        sources.items(),
                    )
                )

            futures = [
                pool.submit(
                    self._process_step,
                    run,
                    step,
                    prev_grib_names,
                    apcp_str_beg,
                    apcp_str_end,
                )
                for step in todo
            ]
            for step, future in zip(todo, futures, strict=True):
                run.success[step.index] = future.result()

            # Remove the decompressed GRIB files (and snow-mask by-products).
            for grib in run.grib_names:
                for match in glob.glob(f'{run.grib_dir}/{grib}*') + glob.glob(
                    f'{run.temp_dir}/{grib}*'
                ):
                    with contextlib.suppress(OSError):
                        os.remove(match)

            if self.mtd_mode != 'Operational' or np.nanmean(run.success) == 0:
                return
            if (datetime.datetime.now() - start_time).seconds > self.ops_check:
                logger.info('Missing Model Data; Plotting What We Have')
                return
            logger.info('There is missing Model Data; Pausing')
            time.sleep(120)

    # ------------------------------------------------------------------
    # MTD output
    # ------------------------------------------------------------------

    def load_data_mtdv90(self, data_path, mtdfile_new, model):
        """
        Load MTD object data (METv9+ output format).

        The gridded simple and cluster object IDs (simp_bin, clus_bin) are
        returned for the model *or* the observation, depending on ``model``,
        while the attribute tables (simp_prop, pair_prop) contain both the
        forecast and observed objects (observed IDs are negative).

        The MTD 2D text file first lists simple forecast objects, then simple
        observed objects, then forecast/observed cluster pairs.  The first
        group is returned as 'simp_prop' and the pairs as 'pair_prop'.

        Parameters
        ----------
        data_path : pathlib.Path or str
            Directory containing the MTD output.
        mtdfile_new : str
            Base name of the renamed MTD output files.
        model : int
            Index into `load_data`.

        Returns
        -------
        lat, lon : numpy.ndarray
            Grid coordinates (cropped to the domain).
        grid_mod, grid_obs : numpy.ndarray
            Raw forecast/observed precipitation (inches), (y, x, time).
        simp_bin, clus_bin : numpy.ndarray
            Simple object / cluster IDs, (y, x, time).
        simp_prop, pair_prop : numpy.ndarray
            Object attributes: OBJECT_ID, OBJECT_CAT, TIME_INDEX, AREA,
            CENTROID_LAT, CENTROID_LON, AXIS_ANG, INTENSITY_10/50/90.
        data_success : numpy.ndarray or int
            Zeros when the file was loaded, 1 when it does not exist.
        """
        obj_file = f'{data_path}/{mtdfile_new}_obj.nc'
        if not os.path.isfile(obj_file):
            logger.info(f'NO FILE:  {obj_file}')
            return (np.nan,) * 8 + (1,)

        dims = self.latlon_dims
        with Dataset(obj_file, 'r') as nc:
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
            grid_mod = nc.variables['fcst_raw'][:] / 25.4
            grid_mod[grid_mod < 0] = np.nan
            simp_bin_mod = nc.variables['fcst_object_id'][:]
            simp_bin_mod[simp_bin_mod < 0] = 0
            if 'fcst_cluster_id' in nc.variables:
                clus_bin_mod = nc.variables['fcst_cluster_id'][:]
                clus_bin_mod[clus_bin_mod < 0] = 0
            else:
                clus_bin_mod = np.zeros(simp_bin_mod.shape)

            if 'obs_object_id' in nc.variables and 'obs_raw' in nc.variables:
                simp_bin_obs = nc.variables['obs_object_id'][:]
                simp_bin_obs[simp_bin_obs < 0] = 0
                if 'obs_cluster_id' in nc.variables:
                    clus_bin_obs = nc.variables['obs_cluster_id'][:]
                    clus_bin_obs[clus_bin_obs < 0] = 0
                else:
                    clus_bin_obs = np.zeros(simp_bin_obs.shape)
                grid_obs = nc.variables['obs_raw'][:] / 25.4
                grid_obs[grid_obs < 0] = np.nan
            else:
                simp_bin_obs = clus_bin_obs = None
                grid_obs = []

        rows, cols = _domain_slices(lat, lon, dims)

        def crop(grid):
            return np.transpose(grid[:, rows, cols], (1, 2, 0))

        try:
            simp_prop, pair_prop = _read_mtd_2d(
                f'{data_path}/{mtdfile_new}_2d.txt'
            )
            with warnings.catch_warnings():
                # Percentiles of masked arrays warn that the mask is ignored.
                warnings.simplefilter('ignore', UserWarning)
                _fill_intensity(
                    simp_prop,
                    simp_bin_obs,
                    grid_obs,
                    simp_bin_mod,
                    grid_mod,
                    obs_90th_ignores_nan=False,
                )
                _fill_intensity(
                    pair_prop,
                    clus_bin_obs,
                    grid_obs,
                    clus_bin_mod,
                    grid_mod,
                    obs_90th_ignores_nan=True,
                )

            if self.ismodel[model] == 0:
                simp_bin, clus_bin = simp_bin_obs, clus_bin_obs
            else:
                simp_bin, clus_bin = simp_bin_mod, clus_bin_mod

            # Drop objects whose centroid lies outside the domain.
            def in_domain(prop):
                return (
                    (prop[4, :] >= dims[1])
                    & (prop[4, :] < dims[3])
                    & (prop[5, :] >= dims[0])
                    & (prop[5, :] < dims[2])
                )

            simp_prop = simp_prop[:, in_domain(simp_prop)]
            pair_prop = pair_prop[:, in_domain(pair_prop)]
            simp_bin = crop(simp_bin)
            clus_bin = crop(clus_bin)
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            # Output exists but could not be interpreted: report no objects.
            logger.exception(
                f'ERROR: could not read MTD objects for {mtdfile_new}'
            )
            simp_bin = np.zeros(crop(grid_mod).shape, dtype=bool)
            clus_bin = np.zeros(simp_bin.shape, dtype=bool)
            simp_prop = pair_prop = np.nan

        if not isinstance(grid_obs, list):
            grid_obs = crop(grid_obs)
        return (
            lat[rows, cols],
            lon[rows, cols],
            crop(grid_mod),
            grid_obs,
            simp_bin,
            clus_bin,
            simp_prop,
            pair_prop,
            np.zeros(clus_bin.shape[2]),
        )

    def _track_file_stem(self, track_save_path, mem, curr_date, prop_type):
        return (
            f'{track_save_path}/{self.grib_path_des}{mem}_'
            f'{curr_date:%Y%m%d%H}_{prop_type}'
            f'_s{int(self.start_hrs)}'
            f'_e{int(self.start_hrs + self.end_fcst_hrs)}'
            f'_h{self.pre_acc:.2f}_t{self.thresh}'
        )

    def port_data_files(
        self,
        curr_date,
        hour_success,
        mtdfile_new,
        simp_prop,
        pair_prop,
        data_exist,
        mem,
    ):
        """
        Save track files: move the raw MTD output (operational HRRR runs) or
        write gzipped NPZ files of the object attributes (Retro).

        Parameters
        ----------
        curr_date : datetime.datetime
            Cycle time.
        hour_success : numpy.ndarray
            True for each verification hour that was loaded.
        mtdfile_new : list of str
            Base names of the MTD output of each model.
        simp_prop, pair_prop : list
            Object attributes of each model.
        data_exist : numpy.ndarray
            True where both model and analysis data exist.
        mem : list of str
            Ensemble member suffix of each model ('' if none).
        """
        # The raw MTD output feeds the HRRR bias look-up (WPCMTDBiasLookup).
        all_hrrr = all('HRRR' in name for name in self.load_data)
        if (
            self.mtd_mode == 'Operational' and all_hrrr
        ) or self.mtd_mode == 'Both':
            for model in range(len(self.load_data)):
                init = self.init_yrmondayhr[model]
                track_save_path = pathlib.Path(
                    self.track_path, init[:4], init[:-2]
                )
                track_save_path.mkdir(parents=True, exist_ok=True)
                logger.info(f'MOVING TRACK FILES TO  {track_save_path}')
                model_temp = pathlib.Path(f'{self.grib_path_temp}_m{model}')
                search_path = (
                    model_temp if model_temp.exists() else self.grib_path_temp
                )
                for match in glob.glob(f'{search_path}/{mtdfile_new[model]}*'):
                    shutil.move(
                        match, track_save_path / os.path.basename(match)
                    )

        if self.mtd_mode == 'Retro':
            track_save_path = pathlib.Path(
                self.track_path, f'{curr_date.year:04d}', f'{curr_date:%Y%m%d}'
            )
            track_save_path.mkdir(parents=True, exist_ok=True)
            logger.info(f'TRACK SAVING: {track_save_path}')
            if (
                np.sum(hour_success) > 0
                and self.load_qpe[0] in self.load_data[1]
            ):
                for prop_type, key, prop in (
                    ('simp_prop', 'simp_prop_k', simp_prop[0]),
                    ('pair_prop', 'pair_prop_k', pair_prop[0]),
                ):
                    npz_path = (
                        self._track_file_stem(
                            track_save_path, mem[0], curr_date, prop_type
                        )
                        + '.npz'
                    )
                    buffer = io.BytesIO()
                    np.savez(buffer, **{key: prop, 'data_exist': data_exist})
                    with gzip.open(npz_path + '.gz', 'wb') as fout:
                        fout.write(buffer.getvalue())
                    logger.info(f'TRACK GZIP: {npz_path}.gz')

    def port_data_figs(self, curdate):
        """
        Copy figures to the WPC web server and S3, and delete old ones there.

        Parameters
        ----------
        curdate : datetime.datetime
            Cycle time.
        """
        if self.mtd_mode != 'Operational':
            return
        www = '/home/people/hpc/www/htdocs/'
        s3 = 's3://s3-east-www.wpc.woc.noaa.gov/public/'
        des = str(self.grib_path_des)
        for domain in self.domain_sub:
            region = domain[1:]
            if self.snow_mask:
                remote, keep_days = 'snowbands/images/', 5
            elif 'HRRRv415min' in ''.join(self.load_data):
                remote, keep_days = f'HRRRSubHr/images/{region}/', 2
            else:
                logger.info(region)
                remote, keep_days = 'verification/mtd/images/', 2
            old = f'{curdate - datetime.timedelta(days=keep_days):%Y%m%d}'

            _run_shell(
                f'scp {self.fig_path}/{des}*{region}*.* '
                f'hpc@wpcrzdm:{www}{remote}'
            )
            _run_shell(
                f'aws s3 cp {self.fig_path}/ {s3}{remote} --recursive '
                '--exclude "*" '
                f'--include "*{des}*{region}*.*"'
            )
            # Clear old files from the servers to save space.
            _run_shell(f"ssh hpc@wpcrzdm 'rm {www}{remote}*{des}*{old}*.*'")
            _run_shell(
                f'aws s3 rm {s3}{remote} --recursive --exclude "*" '
                f'--include "*{old}*.*"'
            )

    def sweep(self, curr_date, mtdfile_new):
        """
        Delete temporary data once a cycle is finished.

        Parameters
        ----------
        curr_date : datetime.datetime
            Cycle time.
        mtdfile_new : list of str
            Base names of the MTD output of each model.
        """
        _remove_matching(f'{self.grib_path_temp}*')
        logger.info(f'rm -rf {self.grib_path_temp}*')

        # HRRR TLE operational MTD output is only needed by the bias look-up.
        if str(self.grib_path_des) == 'MTD_QPF_HRRRTLE_EXT_OPER':
            for name in mtdfile_new:
                logger.info(f'rm -rf {self.track_path}/{name[0:21]}*')
                _remove_matching(f'{self.track_path}/{name[0:21]}*')

    def save_probs(self, curr_date, load_data_nc, simp_bin, data_success):
        """
        Save hourly smoothed ensemble object probabilities (%) to NetCDF in
        the current working directory.

        Parameters
        ----------
        curr_date : datetime.datetime
            Cycle time.
        load_data_nc : list of str
            Model names with '.nc' appended.
        simp_bin : numpy.ndarray
            Simple object IDs, (y, x, hour, model).
        data_success : numpy.ndarray
            0 where a model hour was loaded, (hour, model).
        """
        is_model = np.array(['ST4' not in name for name in load_data_nc])
        probs = []
        for hr_count in range(len(self.hrs)):
            members = is_model & (data_success[hr_count, :] == 0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                ens_mean = np.mean(
                    simp_bin[:, :, hr_count, members] > 0, axis=2
                )
            ens_mean[ens_mean > 0.95] = 0.95
            probs.append(gaussian_filter(ens_mean * 100, self.sigma))

        ny, nx = self.lat.shape
        ds = xr.Dataset(
            {'object_probability': (['fcsthr', 'y', 'x'], np.array(probs))},
            coords={
                'fcsthr': np.array(self.hrs),
                'y': np.arange(ny),
                'x': np.arange(nx),
                'lat': (['y', 'x'], self.lat),
                'lon': (['y', 'x'], self.lon),
            },
        )

        output_filename = (
            f'{os.getcwd()}/{self.grib_path_des}_ALL_TOTENS_objprob_byhour_'
            f'{curr_date:%Y%m%d%H}_p{self.pre_acc:.2f}_t{self.thresh}_probs.nc'
        )
        logger.info('Saving probs file to:')
        logger.info(output_filename)
        ds.to_netcdf(output_filename)

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run_mtd(self):
        """
        Run MTD for every cycle in `list_date`.

        Operational mode tracks one or more models/ensemble members and makes
        figures; Retro mode pairs one model with QPE and saves track files.
        Only one threshold is handled per run.  Accumulation intervals are
        resolved to the minute and model initializations to the hour.
        """
        _configure_logging()
        perf.reset()
        start_time = datetime.datetime.now()

        for curr_date in self.list_date:
            with perf.section('setup_data'):
                self.setup_data(curr_date)
            self._run_cycle_date(curr_date, start_time)

        perf.summary()

    def _model_temp_dirs(self):
        base = self.grib_path_temp
        if len(self.load_data) == 1:
            return [base]
        dirs = [
            pathlib.Path(f'{base}_m{model}')
            for model in range(len(self.load_data))
        ]
        for path in dirs:
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True)
        return dirs

    def _mtd_output_prefix(self, curr_date, run, hour_success):
        """MTD names its output after the valid time of the first input."""
        first = int(np.argmax(hour_success == 1))
        valid = curr_date + datetime.timedelta(hours=float(run.hrs_all[first]))
        return f'mtd_{valid:%Y%m%d_%H%M%S}V'

    def _run_cycle_date(self, curr_date, start_time):
        """Preprocess, track, post-process and plot one cycle."""
        load_data = self.load_data
        n_models = len(load_data)
        load_data_nc = [name + '.nc' for name in load_data]
        apcp_str_end = _accum_str(self.pre_acc)

        # MTD config file (the same for every model).
        conf_path = pathlib.Path(
            self.config_path,
            f'MTDConfig_USE_SAMP_f{int(self.end_fcst_hrs)}'
            f'_p{apcp_str_end}_t{self.thresh}',
        )
        make_config = (
            gen_mtdconfig_15m
            if (
                'NEWSe' in ''.join(self.load_model)
                or '15min' in ''.join(self.load_model)
            )
            else gen_mtdconfig
        )
        config_name = make_config(
            conf_path,
            self.thresh,
            self.conv_radius,
            self.min_volume,
            self.ti_thresh,
            apcp_str_end,
            met_version=self.met_version,
        )

        mem = [
            f'_m{name[name.find("mem") + 3 : name.find("mem") + 5]}'
            if 'mem' in name
            else ''
            for name in load_data_nc
        ]

        model_dirs = self._model_temp_dirs()
        runs = [
            self._plan_model(
                curr_date, model, model_dirs[model], model_dirs[0]
            )
            for model in range(n_models)
        ]

        # 1) pcp_combine + regrid_data_plane (+ masks), every model and hour.
        serial = self._needs_serial_steps()
        workers = 1 if serial else self.n_workers
        with (
            perf.section('preprocess (wall)'),
            concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as step_pool,
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1 if serial else n_models
            ) as model_pool,
        ):
            futures = []
            for model, run in enumerate(runs):
                prev_run = runs[model - 1] if model > 0 else None
                apcp_str_beg = _accum_str(self.acc_int[model])
                futures.append(
                    model_pool.submit(
                        self._preprocess_model,
                        run,
                        prev_run,
                        step_pool,
                        start_time,
                        apcp_str_beg,
                        apcp_str_end,
                    )
                )
            for future in futures:
                future.result()

        # Hours loaded for each model; NEWSe 60 min adds a leading hour.
        n_rows = len(self.hrs) + sum(
            len(run.hrs_all) > len(self.hrs) for run in runs
        )
        data_success = np.ones([n_rows, n_models])
        for run in runs:
            data_success[: len(run.success), run.index] = run.success

        qpe_idx = next(
            (
                i
                for i, name in enumerate(load_data)
                if name == self.load_qpe[0]
            ),
            None,
        )
        mod_idx = next(
            i for i, name in enumerate(load_data) if name == self.load_model[0]
        )

        def hours_loaded(model):
            if self.mtd_mode == 'Retro':
                return (
                    data_success[:, qpe_idx] + data_success[:, mod_idx]
                ) == 0
            return data_success[:, model] == 0

        # 2) MTD: forecast vs QPE (Retro) or each model alone (Operational).
        mtdfile_new = []
        mtd_jobs = []
        for run in runs:
            init = self.init_yrmondayhr[run.index]
            mtdfile_new.append(
                f'mtd_{init[:-2]}_h{init[-2:]}_f{run.last_fcst_hr_str}_'
                f'{run.name}_p{self.pre_acc:.2f}_t{self.thresh}'
            )
            if (
                self.mtd_mode == 'Retro'
                and run.index != qpe_idx
                and np.nanmean(data_success[:, run.index]) == 1
            ):
                logger.info('skipped')
                break
            hour_success = hours_loaded(run.index)
            if self.mtd_mode == 'Retro' and self.load_qpe[0] in run.name:
                args = (
                    ['-fcst']
                    + list(run.data_name_nc_prev[hour_success])
                    + ['-obs']
                    + list(run.data_name_nc[hour_success])
                )
            elif (
                self.mtd_mode == 'Operational'
                and self.load_qpe[0] not in run.name
            ):
                args = ['-single'] + list(run.data_name_nc[hour_success])
            else:
                continue
            mtd_jobs.append(
                (
                    run,
                    hour_success,
                    [f'{self.met_path}/mtd']
                    + args
                    + ['-config', config_name, '-outdir', run.temp_dir],
                )
            )
        mtdfile_new += ['dummyname'] * (n_models - len(mtdfile_new))

        def run_mtd_job(job):
            run, hour_success, args = job
            with perf.section('mtd_tool'):
                _run_cmd(args)
            # Rename the MTD output after the model/cycle.
            prefix = self._mtd_output_prefix(curr_date, run, hour_success)
            old = f'{run.temp_dir}/{prefix}'
            new = f'{run.temp_dir}/{mtdfile_new[run.index]}'
            _move(
                f'{old}_obj.nc',
                f'{new}_obj.nc',
                f'MTD obj.nc output for model {run.index}',
            )
            _move(
                f'{old}_2d.txt',
                f'{new}_2d.txt',
                f'MTD 2d.txt output for model {run.index}',
            )
            _remove_matching(f'{old}_3d_*.txt')

        with perf.section('mtd (wall)'):
            mtd_workers = (
                1
                if self.mtd_mode == 'Retro'
                else min(self.n_workers, len(mtd_jobs) or 1)
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=mtd_workers
            ) as pool:
                list(pool.map(run_mtd_job, mtd_jobs))

        # The per-hour NetCDF inputs are no longer needed.
        for run in runs:
            shutil.rmtree(run.grib_dir, ignore_errors=True)
            for step in run.steps:
                with contextlib.suppress(OSError):
                    os.remove(step.nc)

        # 3) Read the MTD output into grids and attribute tables.
        lat_shape = self.lat.shape
        simp_bin = np.zeros(
            (lat_shape[0], lat_shape[1], len(self.hrs), n_models),
            dtype=np.int8,
        )
        clus_bin = np.zeros_like(simp_bin)
        simp_prop = [[] for _ in range(n_models)]
        pair_prop = [[] for _ in range(n_models)]
        hour_success = None
        data_exist = None
        for model in range(n_models):
            with perf.section('load_data_mtdv90'):
                if self.mtd_mode == 'Retro':
                    # Model and analysis objects are both in the paired output.
                    loaded = self.load_data_mtdv90(
                        runs[qpe_idx].temp_dir, mtdfile_new[1], model
                    )
                else:
                    loaded = self.load_data_mtdv90(
                        runs[model].temp_dir, mtdfile_new[model], model
                    )
            (
                lat_t,
                lon_t,
                fcst_p,
                obs_p,
                simp_bin_p,
                clus_bin_p,
                simp_prop_k,
                pair_prop_k,
                _,
            ) = loaded

            hour_success = hours_loaded(model)
            hours_true = np.where(hour_success == 1)[0]

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                has_objects = not np.isnan(np.nanmean(simp_prop_k))
            if has_objects:
                simp_prop[model] = simp_prop_k
                pair_prop[model] = pair_prop_k
            else:
                simp_prop[model] = np.full((10, 1), np.nan)
                pair_prop[model] = np.full((10, 1), np.nan)

            if not _all_missing(simp_bin_p):
                simp_bin[:, :, hours_true, model] = simp_bin_p
            if not _all_missing(clus_bin_p):
                clus_bin[:, :, hours_true, model] = clus_bin_p

            # True where both model and analysis data exist.
            if _all_missing(obs_p) or _all_missing(fcst_p):
                data_exist = np.zeros(lat_shape)
            else:
                data_exist = (
                    np.isnan(obs_p[:, :, 0].data) * 1
                    + np.isnan(fcst_p[:, :, 0].data) * 1
                ) == 0

            # Optionally plot hourly paired model/obs objects to check by eye.
            if self.plot_allhours and model > 0:
                wet = fcst_p[:, :, 0].data > 0
                bounds = [
                    np.nanmin(lon_t[wet]),
                    np.nanmin(lat_t[wet]),
                    np.nanmax(lon_t[wet]),
                    np.nanmax(lat_t[wet]),
                ]
                mtd_plot_retro(
                    str(self.grib_path_des),
                    str(self.fig_path),
                    bounds,
                    self.pre_acc,
                    runs[-1].hrs_all,
                    self.thresh,
                    curr_date,
                    data_success,
                    load_data_nc,
                    self.lat,
                    self.lon,
                    fcst_p,
                    obs_p,
                    clus_bin,
                    pair_prop,
                )

        # 4) Track files, figures and web transfer.
        with perf.section('port_data_files'):
            logger.info('PORT DATA FILES')
            self.port_data_files(
                curr_date,
                hour_success,
                mtdfile_new,
                simp_prop,
                pair_prop,
                data_exist,
                mem,
            )

        if self.mtd_mode == 'Operational':
            if self.save_probs_to_file:
                self.save_probs(
                    curr_date, load_data_nc, simp_bin, data_success
                )

            with perf.section('plotting_total'):
                self._plot_regions(
                    curr_date, data_success, load_data_nc, simp_bin, simp_prop
                )

            if self.transfer_to_prod:
                with perf.section('port_data_figs'):
                    self.port_data_figs(curr_date)

        with perf.section('sweep'):
            self.sweep(curr_date, mtdfile_new)

    def _plot_regions(
        self, curr_date, data_success, load_data_nc, simp_bin, simp_prop
    ):
        """
        Make the operational figures for every plotting region.

        The hourly figures of each region are split into chunks so the work
        spreads evenly over the plotting processes; GIFs are assembled once
        all frames exist.
        """
        if not self.domain_sub:
            return
        workers = self.n_workers
        chunks = split_hours(
            len(self.hrs), max(1, round(4 * workers / len(self.domain_sub)))
        )
        jobs = []
        for region, bounds in zip(
            self.domain_sub, self.latlon_sub, strict=False
        ):
            common = dict(
                grib_path_des=str(self.grib_path_des) + region,
                fig_path_s=str(self.fig_path),
                latlon_dims=bounds,
                pre_acc=self.pre_acc,
                hrs=self.hrs,
                raw_thres=self.thresh,
                curdate=curr_date,
                data_success=data_success,
                load_data_nc=load_data_nc,
                simp_prop=simp_prop,
            )
            if self.snow_mask:
                jobs += [
                    (
                        mtd_plot_all_snow_fcst,
                        dict(common, hour_indices=chunk, animate=False),
                    )
                    for chunk in chunks
                ]
            else:
                fcst = dict(common, sigma=self.sigma, animate=False)
                jobs.append((mtd_plot_all_fcst, dict(fcst, hour_indices=[])))
                jobs += [
                    (
                        mtd_plot_all_fcst,
                        dict(fcst, hour_indices=chunk, summary=False),
                    )
                    for chunk in chunks
                ]
                jobs.append(
                    (mtd_plot_tle_fcst, dict(common, sigma=self.sigma))
                )

        shared = dict(lat=self.lat, lon=self.lon, simp_bin=simp_bin)
        plot_regions_in_parallel(jobs, workers, shared)

        animate = animate_all_snow_fcst if self.snow_mask else animate_all_fcst
        for region in self.domain_sub:
            animate(
                str(self.grib_path_des) + region,
                str(self.fig_path),
                self.pre_acc,
                self.hrs,
                self.thresh,
                curr_date,
            )
