"""Bias look-up tables linking real-time MTD objects to retrospective pairs.

`WPCMTDBiasLookup` aggregates retrospective model/analysis object pairs
(written by `WPCMTD` in 'Retro' mode) onto a coarse grid, then attaches the
historical model-minus-analysis differences near each real-time object to
that object and plots them.
"""

import dataclasses
import datetime
import gzip
import io
import logging
import os
import pathlib
import time
from collections import defaultdict

import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter

from .mtd import WPCMTD
from .plot import mtd_plot_retro_disp_vect, mtd_plot_retro_prob
from .utils import haversine2_distance, read_CONUS_mask

logger = logging.getLogger(__name__)


def _diurnal_hour(time_index, pre_acc, init_hour):
    """Hour of day (UTC) at the end of accumulation period ``time_index``."""
    hour = ((time_index + 1) * pre_acc) + init_hour
    return hour - ((hour // 24) * 24)


def _outside(values, low, high):
    """True if any of ``values`` is outside [low, high] (False if empty)."""
    return bool(np.any(values < low) or np.any(values > high))


def _signed_offsets(lat_mod, lon_mod, lat_obs, lon_obs):
    """N-S and E-W model displacement (km); south/west are negative."""
    ysign = 1 if lat_mod > lat_obs else -1
    xsign = 1 if lon_mod > lon_obs else -1
    ydiff = ysign * haversine2_distance((lat_mod, lon_mod), (lat_obs, lon_mod))
    xdiff = xsign * haversine2_distance((lat_mod, lon_mod), (lat_mod, lon_obs))
    return ydiff, xdiff


def _object_grid(cells, shape):
    """Convert {(y, x): [row, ...]} to an object array of 2-D row arrays."""
    grid = np.empty(shape, dtype=object)
    for yloc in range(shape[0]):
        for xloc in range(shape[1]):
            rows = cells.get((yloc, xloc))
            grid[yloc, xloc] = np.array(rows) if rows else []
    return grid


@dataclasses.dataclass
class WPCMTDBiasLookup(WPCMTD):
    """
    Provide displacement/intensity bias information for real-time objects
    by linking them to retrospective model objects and their observations.
    """

    grib_path_ret: (
        str  # Label (grib_path_des) of the retro track files to load
    )
    grid_res_nat: (
        float  # Native grid resolution (deg); [] aggregates everything
    )
    grid_res_agg: float  # Aggregation radius (deg); [] aggregates everything
    end_fcst_hrs_ret: float  # Last forecast hour of the retro track files
    update_freq: int  # Hours before the look-up tables are regenerated
    hourly_sub: int  # Use retro objects within +/- this many forecast hours

    def __post_init__(self):
        # The WPCMTD checks/date adjustment are deliberately not applied here.
        self._init_state()
        self.lon_nat = None
        self.lat_nat = None
        self.conus_mask = None
        self.grid_pair = (
            None  # retro model - obs differences of paired objects
        )
        self.grid_init = None  # ... at object initiation
        self.grid_disp = None  # ... at object dissipation
        self.pair_prop_count = None
        self.grid_poygpmy = None  # P(observed object | model object), %

        # Aggregate over the whole domain unless both resolutions are given.
        if self.grid_res_nat == [] or self.grid_res_agg == []:
            self.grid_res_nat = []
            self.grid_res_agg = []
        elif self.grid_res_agg < self.grid_res_nat:
            self.grid_res_agg = self.grid_res_nat

    def create_grid(self):
        """Create the grid (and CONUS mask) that retro biases are mapped to."""
        dims = self.latlon_dims
        if self.grid_res_nat != []:
            res = self.grid_res_nat
            self.lon_nat, self.lat_nat = np.meshgrid(
                np.linspace(
                    dims[0],
                    dims[2],
                    int(np.round((dims[2] - dims[0]) / res, 2) + 1),
                ),
                np.linspace(
                    dims[3],
                    dims[1],
                    int(np.round((dims[3] - dims[1]) / res, 2) + 1),
                ),
            )
            self.conus_mask, self.lat, self.lon = read_CONUS_mask(
                self.grib_path, dims, res
            )
            self.conus_mask[np.isnan(self.conus_mask)] = 0
            # The mask grid is offset by half a cell; shift it and smooth it to
            # keep slightly offshore regions.
            self.conus_mask = interpolate.griddata(
                np.stack((self.lon.flatten(), self.lat.flatten()), axis=1),
                self.conus_mask.flatten(),
                (self.lon - res / 2, self.lat + res / 2),
                method='nearest',
            )
            self.conus_mask = gaussian_filter(self.conus_mask, 1) >= 0.1
        else:
            # Everything is aggregated to one point.
            self.lon_nat = np.array([[dims[0], dims[2]], [dims[0], dims[2]]])
            self.lat_nat = np.array([[dims[3], dims[3]], [dims[1], dims[1]]])
            self.conus_mask = np.ones(self.lat_nat.shape)

    def _grid_index(self, lat, lon):
        """(row, column) of the native grid cell containing (lat, lon)."""
        yloc = int(np.argmax(np.diff([self.lat_nat[:, 1] > lat])) + 1)
        xloc = int(np.argmax(np.diff([self.lon_nat[1, :] < lon])) + 1)
        return yloc, xloc

    def _search_radius(self):
        """Aggregation radius in native grid points."""
        if self.grid_res_agg == []:
            return 0
        grid_res_nat = np.unique(np.round(np.diff(self.lon_nat), 1))[0]
        return float(np.floor(self.grid_res_agg / grid_res_nat) - 1)

    def _neighbors(self, yloc, xloc, radius):
        """Grid cells within ``radius`` points of (yloc, xloc), x-major."""
        xsearch = np.arange(xloc - radius, xloc + radius + 1, 1)
        xsearch = xsearch[
            (xsearch >= 0) & (xsearch < self.lon_nat.shape[1])
        ].astype(int)
        ysearch = np.arange(yloc - radius, yloc + radius + 1, 1)
        ysearch = ysearch[
            (ysearch >= 0) & (ysearch < self.lat_nat.shape[0])
        ].astype(int)
        return [(yloc_l, xloc_l) for xloc_l in xsearch for yloc_l in ysearch]

    def _track_file(self, date, model, prop_type, fcst_end):
        """Gzipped retro track file written by `WPCMTD.port_data_files`."""
        mem = (
            f'_m{model[model.find("mem") + 3 : model.find("mem") + 5]}'
            if 'mem' in model
            else ''
        )
        folder = pathlib.Path(
            self.track_path, f'{date.year:04d}', f'{date:%Y%m%d}'
        )
        return (
            f'{folder}/{self.grib_path_ret}{mem}_{date:%Y%m%d%H}_{prop_type}'
            f'_s{int(self.start_hrs)}_e{int(self.start_hrs + fcst_end)}'
            f'_h{self.pre_acc:.2f}_t{self.thresh}.npz'
        )

    @staticmethod
    def _load_track(filename, key):
        """Read one array from a gzipped NPZ track file."""
        with gzip.open(filename + '.gz', 'rb') as fin:
            with np.load(io.BytesIO(fin.read())) as data:
                return data[key]

    def load_pair_prop(self):
        """
        Load the retrospective paired objects for every date and aggregate
        model-minus-analysis differences onto the native grid: while paired
        (``grid_pair``), at initiation (``grid_init``) and at dissipation
        (``grid_disp``).  Differences are mapped to the analysis location.
        """
        fcst_hrs_all = np.arange(
            self.start_hrs, self.end_fcst_hrs_ret, self.pre_acc
        )
        radius = self._search_radius()
        pair_cells = defaultdict(list)
        init_cells = defaultdict(list)
        disp_cells = defaultdict(list)
        self.pair_prop_count = 0
        lat_lim = (np.nanmin(self.lat_nat), np.nanmax(self.lat_nat))
        lon_lim = (np.nanmin(self.lon_nat), np.nanmax(self.lon_nat))

        for date in self.list_date:
            filename = self._track_file(
                date, self.load_model[0], 'pair_prop', self.end_fcst_hrs_ret
            )
            try:
                pair_prop_k = self._load_track(filename, 'pair_prop_k')
            except OSError:
                logger.info(filename + ' Not Found')
                continue

            if np.mean(np.isnan(pair_prop_k)) >= 1:
                continue
            if np.mean(pair_prop_k) == 0:  # model exists but has no objects
                self.pair_prop_count += 1
                continue

            prop = pair_prop_k
            for obs_num in np.unique(
                prop[0][prop[0] < 0]
            ):  # each observed "storm"
                obs_use = np.where(prop[0] == obs_num)[0]
                mod_use = np.where(prop[0] == -obs_num)[0]
                time_use = np.union1d(prop[2][obs_use], prop[2][mod_use])
                # One extra step so the end of the track is always captured.
                time_use = np.hstack((time_use, time_use[-1] + 1))

                first, final = {}, {}
                obs_ind_prev = mod_ind_prev = None
                for time_val in time_use:
                    obs_ind = np.where(
                        (prop[1] == obs_num) & (prop[2] == time_val)
                    )[0]
                    mod_ind = np.where(
                        (prop[1] == -obs_num) & (prop[2] == time_val)
                    )[0]

                    if _outside(prop[4][obs_ind], *lat_lim) or _outside(
                        prop[5][obs_ind], *lon_lim
                    ):
                        continue  # analysis object outside the domain

                    # First/last appearance of the analysis and model objects.
                    for side, ind, ind_prev in (
                        ('obs', obs_ind, obs_ind_prev),
                        ('mod', mod_ind, mod_ind_prev),
                    ):
                        if len(ind) > 0 and side not in first:
                            first[side] = self._object_state(
                                prop, ind[0], int(time_val), date
                            )
                        elif (
                            len(ind) == 0
                            and side in first
                            and side not in final
                        ):
                            final[side] = self._object_state(
                                prop, ind_prev[0], int(time_val - 1), date
                            )

                    # Differences while both objects exist.
                    if len(mod_ind) > 0 and len(obs_ind) > 0:
                        o, m = obs_ind[0], mod_ind[0]
                        ydiff, xdiff = _signed_offsets(
                            prop[4][m], prop[5][m], prop[4][o], prop[5][o]
                        )
                        row = [
                            ydiff,
                            xdiff,
                            prop[7][m] - prop[7][o],
                            prop[8][m] - prop[8][o],
                            prop[9][m] - prop[9][o],
                            prop[3][m] - prop[3][o],
                            prop[6][m] - prop[6][o],
                            (prop[2][o] + 1) * self.pre_acc,
                            _diurnal_hour(prop[2][o], self.pre_acc, date.hour),
                        ]
                        yloc, xloc = self._grid_index(prop[4][o], prop[5][o])
                        for cell in self._neighbors(yloc, xloc, radius):
                            pair_cells[cell].append(row)

                    obs_ind_prev, mod_ind_prev = obs_ind, mod_ind

                # Initiation (ignoring objects that existed at time zero).
                if (
                    first.get('obs', {}).get('t', 0) != 0
                    and first.get('mod', {}).get('t', 0) != 0
                ):
                    self._add_lifecycle_row(
                        init_cells, first['mod'], first['obs'], radius
                    )
                # Dissipation (ignoring objects lasting to the final time).
                last = len(fcst_hrs_all) - 1
                if (
                    final.get('obs', {}).get('t', last) != last
                    and final.get('mod', {}).get('t', last) != last
                ):
                    self._add_lifecycle_row(
                        disp_cells, final['mod'], final['obs'], radius
                    )

            self.pair_prop_count += 1

        shape = self.lat_nat.shape
        self.grid_pair = _object_grid(pair_cells, shape)
        self.grid_init = _object_grid(init_cells, shape)
        self.grid_disp = _object_grid(disp_cells, shape)

    def _object_state(self, prop, ind, time_index, date):
        """Attributes of one object at one time, for initiation/dissipation."""
        return dict(
            t=time_index,
            loc=self._grid_index(prop[4][ind], prop[5][ind]),
            lat=prop[4][ind],
            lon=prop[5][ind],
            ang=prop[6][ind],
            int10=prop[7][ind],
            int50=prop[8][ind],
            int90=prop[9][ind],
            area=prop[3][ind],
            fhour=(prop[2][ind] + 1) * self.pre_acc,
            dhour=_diurnal_hour(prop[2][ind], self.pre_acc, date.hour),
        )

    def _add_lifecycle_row(self, cells, mod, obs, radius):
        """Append a model-minus-analysis row at the analysis location."""
        ydiff, xdiff = _signed_offsets(
            mod['lat'], mod['lon'], obs['lat'], obs['lon']
        )
        row = [
            ydiff,
            xdiff,
            mod['int10'] - obs['int10'],
            mod['int50'] - obs['int50'],
            mod['int90'] - obs['int90'],
            mod['area'] - obs['area'],
            mod['ang'] - obs['ang'],
            mod['t'] - obs['t'],
            obs['fhour'],
            obs['dhour'],
        ]
        for cell in self._neighbors(*obs['loc'], radius):
            cells[cell].append(row)

    def load_pair_POYgPMY(self):  # noqa: N802 (public name)
        """
        Probability (%) that an observed object exists given that a simple
        model object exists, per native grid cell.  Statistics are counted
        once per model track and mapped to the model's mean location.
        """
        radius = self._search_radius()
        shape = self.lat_nat.shape
        grid_mod_yes = np.zeros(shape)  # model tracks that were paired
        grid_mod_tot = np.zeros(shape)  # all model tracks
        lat_lim = (np.nanmin(self.lat_nat), np.nanmax(self.lat_nat))
        lon_lim = (np.nanmin(self.lon_nat), np.nanmax(self.lon_nat))

        for date in self.list_date:
            filename = self._track_file(
                date, self.load_model[0], 'simp_prop', self.end_fcst_hrs_ret
            )
            try:
                simp_prop_k = self._load_track(filename, 'simp_prop_k')
            except OSError:
                logger.info(filename + ' Not Found')
                continue
            if (
                np.mean(np.isnan(simp_prop_k)) >= 1
                or np.mean(simp_prop_k) == 0
            ):
                continue

            for mod_num in np.unique(simp_prop_k[0][simp_prop_k[0] > 0]):
                mod_use = np.where(simp_prop_k[0] == mod_num)[0]
                mean_lat = np.nanmean(simp_prop_k[4][mod_use])
                mean_lon = np.nanmean(simp_prop_k[5][mod_use])
                if _outside(mean_lat, *lat_lim) or _outside(
                    mean_lon, *lon_lim
                ):
                    continue
                paired = np.nanmean(simp_prop_k[1][mod_use]) > 0
                for yloc_l, xloc_l in self._neighbors(
                    *self._grid_index(mean_lat, mean_lon), radius
                ):
                    if paired:
                        grid_mod_yes[yloc_l, xloc_l] += 1
                    grid_mod_tot[yloc_l, xloc_l] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            self.grid_poygpmy = (grid_mod_yes / grid_mod_tot) * 100

    def gather_retro_biases(self, simp_prop):
        """
        Collect the retro differences near each real-time object.

        Parameters
        ----------
        simp_prop : list
            Object attributes of each operational model.

        Returns
        -------
        list
            ``pair_diff[model][row]``: (n, 10) array of analysis-minus-model
            differences (ydiff, xdiff, intensity 10/50/90, area, angle),
            forecast hour, diurnal hour and P(obs|model), or a (10, 1) NaN
            array when there are none.
        """
        pair_diff = []
        for model in range(len(self.load_model)):
            prop = simp_prop[model]
            diffs = []
            for ind_t in range(len(prop[0])):
                yloc, xloc = self._grid_index(prop[4][ind_t], prop[5][ind_t])
                t_idx = prop[2][ind_t]
                if t_idx > self.end_fcst_hrs_ret - self.hourly_sub:
                    fcst_hour_range = np.arange(
                        self.end_fcst_hrs_ret - self.hourly_sub * 2,
                        self.end_fcst_hrs_ret + 1,
                        1,
                    )
                else:
                    fcst_hour_range = np.arange(
                        t_idx * self.pre_acc - self.hourly_sub,
                        t_idx * self.pre_acc + self.hourly_sub + 1,
                        1,
                    )

                rows = []
                if self.conus_mask[yloc, xloc] == 1:
                    for rec in self.grid_pair[yloc][xloc]:
                        if rec[7] in fcst_hour_range:
                            rows.append(
                                [
                                    -rec[0],
                                    -rec[1],
                                    -rec[2],
                                    -rec[3],
                                    -rec[4],
                                    -rec[5],
                                    -rec[6],
                                    rec[7],
                                    rec[8],
                                    self.grid_poygpmy[yloc][xloc],
                                ]
                            )
                diffs.append(
                    np.array(rows) if rows else np.full((10, 1), np.nan)
                )
            pair_diff.append(diffs)
        return pair_diff

    def run_biaslookup(self):
        """
        Link real-time objects to retrospective biases and plot them:

        1) load the current model/ensemble MTD objects;
        2) load (or regenerate, every ``update_freq`` hours) the retro
           difference statistics and P(obs|model) for one model;
        3) attach the statistics to the operational tracks ('pair_diff');
        4) plot the tracks with their retro biases.

        The retro differences (model - analysis) are mapped to the analysis
        grid; their sign is flipped here and the difference between mapping
        to the model and analysis grid is assumed negligible.
        """
        self.create_grid()
        load_model_nc = [name + '.nc' for name in self.load_model]
        self.setup_data(self.end_date)

        biaslookup_file = (
            f'{self.track_path}/{self.grib_path_des}_biaslookup'
            f'_s{int(self.start_hrs)}'
            f'_e{int(self.start_hrs + self.end_fcst_hrs_ret)}'
            f'_h{self.pre_acc:.2f}_t{self.thresh}.npz'
        )
        try:
            age_hrs = (time.time() - os.stat(biaslookup_file).st_mtime) / 3600
        except OSError:
            age_hrs = np.inf

        if age_hrs >= self.update_freq:
            self.load_pair_prop()
            self.load_pair_POYgPMY()
            np.savez(
                biaslookup_file,
                grid_pair=self.grid_pair,
                grid_POYgPMY=self.grid_poygpmy,
            )
        else:
            with np.load(biaslookup_file, allow_pickle=True) as data:
                self.grid_pair = data['grid_pair']
                self.grid_poygpmy = data['grid_POYgPMY']

        n_hours = int(self.end_fcst_hrs - self.start_hrs)
        n_models = len(self.load_model)
        mtdfile_new = []
        simp_bin = np.zeros(
            (self.lat.shape[0], self.lat.shape[1], n_hours, n_models),
            dtype=np.int8,
        )
        simp_prop = []
        data_success = np.ones([n_hours, n_models])

        for model, name in enumerate(self.load_model):
            lag_inc = int(name[name.find('lag') + 3 : name.find('lag') + 6])
            model_init = self.end_date - datetime.timedelta(hours=lag_inc)
            mtdfile_new.append(
                f'mtd_{model_init:%Y%m%d}_h{model_init:%H}'
                f'_f{int(self.end_fcst_hrs) + lag_inc:02d}_{name}'
                f'_p{self.pre_acc:.2f}_t{self.thresh}'
            )
            track_load_path = pathlib.Path(
                self.track_path,
                f'{model_init.year:04d}',
                f'{model_init:%Y%m%d}',
            )

            (_, _, _, _, simp_bin_p, _, simp_prop_k, _, data_success_p) = (
                self.load_data_mtdv90(
                    track_load_path, mtdfile_new[model], model
                )
            )
            if np.isnan(np.nanmean(simp_prop_k)):
                raise ValueError('There is no operational model to track.')

            # Lagged members may cover fewer hours (their last forecast
            # hour is end_fcst_hrs); the remaining hours stay missing.
            hours_true = np.where(data_success_p == 0)[0]
            data_success[: len(data_success_p), model] = data_success_p
            if np.mean(np.isnan(simp_bin_p)) != 1:
                simp_bin[:, :, hours_true, model] = simp_bin_p
            simp_prop.append(simp_prop_k)

        pair_diff = self.gather_retro_biases(simp_prop)

        for region, bounds in zip(
            self.domain_sub, self.latlon_sub, strict=False
        ):
            args = (
                str(self.grib_path_des) + region,
                str(self.fig_path),
                bounds,
                self.pre_acc,
                self.hrs,
                self.thresh,
                self.end_date,
                data_success,
                load_model_nc,
                self.lat,
                self.lon,
                simp_bin,
                simp_prop,
                pair_diff,
            )
            if n_models == 1:
                mtd_plot_retro_prob(*args)
            else:
                mtd_plot_retro_disp_vect(*args, self.sigma)

        self.sweep(self.end_date, mtdfile_new)
