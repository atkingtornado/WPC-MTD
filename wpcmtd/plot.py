"""Figures for WPC-MTD object tracks and ensemble object probabilities.

Terminology: "simple" objects are unmerged and "cluster" objects merged;
"single" objects are unmatched and "pair" objects matched (model/analysis).

The plotting functions take ``simp_prop`` as a list (one entry per model) of
attribute arrays whose rows are OBJECT_ID, OBJECT_CAT, TIME_INDEX, AREA,
CENTROID_LAT, CENTROID_LON, AXIS_ANG, INTENSITY_10, INTENSITY_50 and
INTENSITY_90.  They temporarily convert TIME_INDEX to forecast hours in
place and restore it before returning.
"""

import bisect
import concurrent.futures
import contextlib
import datetime
import logging
import math
import multiprocessing.forkserver
import os
import shutil
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.io.shapereader as shapereader
import geojson
import geojsoncontour
import matplotlib as mpl
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy import interpolate
from scipy.ndimage import gaussian_filter

mpl.use('Agg')  # so plots can be made from cron
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)

# Colors for the 90th percentile object intensity.
INTENSITY_COLORS = [
    '#ffffff',
    '#02fd02',
    '#01c501',
    '#228b22',
    '#9acd32',
    '#fdf802',
    '#f8d826',
    '#e5bc00',
    '#fd9500',
    '#fd0000',
    '#bc0000',
]
SNOW_COLORS = [
    '#ffffff',
    '#95FF9F',
    '#0BC31D',
    '#006400',
    '#FAE94C',
    '#F5AD31',
    '#FF4500',
    '#CC0000',
    '#8B0000',
    '#4B0082',
    '#FF00FF',
]
MODEL_MARKERS = ['s', 'd', '*', '^', '<', '>', 'v', 'o', '8']
TEXT_BOX = dict(boxstyle='round', facecolor='wheat', alpha=1)
REGION_BOX = dict(boxstyle='round', facecolor='tan', alpha=1)
PLATE = ccrs.PlateCarree()

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def plot_map(lat, lon, ax):
    """
    Draw coastlines, borders, states and lat/lon gridlines and set the extent.

    Parameters
    ----------
    lat, lon : array_like
        2-D coordinate grids or [min, max] pairs.
    ax : cartopy.mpl.geoaxes.GeoAxes
        Axes to draw on.
    """
    lat = np.array(lat)
    lon = np.array(lon)

    ax.coastlines(edgecolor='grey', alpha=1, linewidth=0.5)
    ax.add_feature(cf.STATES, edgecolor='grey', alpha=1, linewidth=0.3)
    ax.add_feature(cf.BORDERS, edgecolor='grey', alpha=1, linewidth=0.3)

    if lon.ndim == 2:
        lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
        lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
        gridsize = np.ceil(np.ceil(lon_max) - np.floor(lon_min))
    else:
        lon_min, lon_max = lon[0], lon[1]
        lat_min, lat_max = lat[0], lat[1]
        gridsize = np.ceil(np.ceil(lon[1]) - np.floor(lon[0]))

    if gridsize >= 250:
        res = 40
    elif gridsize >= 200:
        res = 30
    elif gridsize >= 150:
        res = 20
    elif gridsize >= 100:
        res = 10
    else:
        res = 5
    xticks = np.arange(
        lon_min - lon_min % res, lon_max + lon_max % res + res, res
    )
    yticks = np.arange(
        lat_min - lat_min % res, lat_max + lat_max % res + res, res
    )

    gl = ax.gridlines(
        crs=PLATE,
        draw_labels=True,
        linewidth=0.3,
        color='black',
        linestyle='--',
        xlocs=xticks,
        ylocs=yticks,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=PLATE)


def prefetch_map_data():
    """Download any missing Natural Earth files before plotting in parallel."""
    for scale in ('110m', '50m', '10m'):
        for category, name in (
            ('physical', 'coastline'),
            ('cultural', 'admin_0_boundary_lines_land'),
            ('cultural', 'admin_1_states_provinces_lakes'),
        ):
            try:
                shapereader.natural_earth(
                    resolution=scale, category=category, name=name
                )
            except Exception as err:  # offline: cartopy retries when drawing
                warnings.warn(
                    f'Could not fetch Natural Earth {scale} {name}: {err}',
                    stacklevel=2,
                )


def _new_map():
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Mercator(), alpha=1)
    return fig, ax


def _finish_map(fig, ax, latlon_dims, path, dpi):
    plot_map(
        [latlon_dims[1], latlon_dims[3]], [latlon_dims[0], latlon_dims[2]], ax
    )
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def _blues(low, high=1.0):
    """Trimmed matplotlib 'Blues' colormap, white below the first level."""
    cmap = LinearSegmentedColormap.from_list(
        'name', plt.cm.Blues(np.linspace(low, high))
    )
    cmap.set_under('white')
    return cmap


def _is_model(load_data_nc):
    """True for models, False for the ST4 analysis."""
    return np.array(['ST4' not in name for name in load_data_nc])


def _uses_quarter_hours(load_data_nc):
    names = ''.join(load_data_nc)
    return 'NEWSe5min' in names or 'HRRRv415min' in names


def _time_index_to_hours(simp_prop, load_data_nc):
    """Convert the MTD TIME_INDEX row to forecast hours (in place)."""
    quarter = _uses_quarter_hours(load_data_nc)
    for prop in simp_prop[: len(load_data_nc)]:
        prop[2] = prop[2] / 4.0 + 0.25 if quarter else prop[2] + 1


def _hours_to_time_index(simp_prop, load_data_nc):
    """Undo `_time_index_to_hours` (in place)."""
    quarter = _uses_quarter_hours(load_data_nc)
    for prop in simp_prop[: len(load_data_nc)]:
        prop[2] = (prop[2] - 0.25) * 4 if quarter else prop[2] - 1


def _has_data(prop):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.mean(np.isnan(prop)) != 1


def _member_mean(simp_bin, members, hour):
    """
    Fraction of the selected members with an object at each grid point.

    Identical to ``np.nanmean`` over the members of a 0/1 float array, but
    computed from the integer object grid without creating float copies.
    NaN when no member is selected.
    """
    n_members = int(np.sum(members))
    if n_members == 0:
        return np.full(simp_bin.shape[:2], np.nan)
    return (
        np.count_nonzero(simp_bin[:, :, hour, members] > 0, axis=2) / n_members
    )


def _max_member_mean(simp_bin, members_by_hour):
    """Maximum over hours of `_member_mean` (NaN hours ignored)."""
    result = None
    for hour in range(simp_bin.shape[2]):
        mean = _member_mean(simp_bin, members_by_hour[hour], hour)
        result = mean if result is None else np.fmax(result, mean)
    return result


def _members_found(data_success):
    """Number of members (columns) with at least one loaded hour."""
    return int(np.sum(np.any(data_success == 0, axis=0)))


def _nan_where_missing(simp_bin, data_success, models):
    """Float 0/1 object grid of ``models`` with NaN where data are missing."""
    grid = (simp_bin[..., models] > 0).astype(float)
    nan_loc = (data_success[..., models] == 1).astype(float)
    nan_loc[nan_loc == 1] = np.nan
    return grid + nan_loc


def _model_markers(load_data_nc, markers=MODEL_MARKERS):
    """Legend names and marker symbols (one marker per model if few)."""
    if len(load_data_nc) <= 8:
        return list(load_data_nc), markers[: len(load_data_nc)]
    return [], ['d']


def _marker_for(model, load_data_nc, names, symbols, default=None):
    if len(load_data_nc) == len(names):
        return symbols[model]
    if len(symbols) == 1:
        return symbols[0]
    marker = default
    for name, symbol in zip(names, symbols, strict=True):
        if name in load_data_nc[model]:
            marker = symbol
    return marker


def _legend(
    ax, colors, labels, fontsize, loc, marker='o', markersize=7, color=None
):
    """Legend of point markers (becomes the axes legend)."""
    handles = [
        mpl.lines.Line2D(
            range(1),
            range(1),
            color=color or colors[i],
            linestyle='None',
            marker=marker if color is None else colors[i],
            markersize=markersize,
        )
        for i in range(len(labels))
    ]
    return ax.legend(
        handles,
        list(labels),
        fontsize=fontsize,
        numpoints=1,
        loc=loc,
        framealpha=1,
    )


def _intensity_legend(ax, colors, labels, fontsize=6, markersize=7):
    """Color legend kept on the axes when a second legend is added."""
    ax.add_artist(
        _legend(ax, colors, labels, fontsize, loc=3, markersize=markersize)
    )


def _marker_legend(ax, symbols, labels, fontsize=6, markersize=7):
    return _legend(
        ax, symbols, labels, fontsize, loc=4, markersize=markersize, color='k'
    )


def _object_style(is_model):
    """(IDs to keep, line color, line width, marker size, zorder)."""
    if is_model:
        return (lambda ids: ids[ids > 0]), 'k', 0.5, 30, 2
    return (lambda ids: ids[ids < 0]), 'b', 1.5, 60, 3


def _valid(curdate, hours, fmt='%Y%m%d%H%M'):
    """Valid time ``hours`` after ``curdate``, formatted."""
    return (curdate + datetime.timedelta(hours=hours)).strftime(fmt)


def _fig_file(fig_path_s, name):
    return f'{fig_path_s}/{name}'


def write_gif(frames, gif_path, delay_cs, thumbnail_path=None):
    """
    Write an animated GIF (``delay_cs`` hundredths of a second per frame,
    looping forever) and optionally copy it to a date-free thumbnail name.
    """
    frames = [frame for frame in frames if os.path.exists(frame)]
    if not frames:
        logger.warning(f'WARNING: no frames for {gif_path}')
        return
    images = [Image.open(frame).convert('RGB') for frame in frames]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    padded = []
    for image in images:
        if image.size != (width, height):
            canvas = Image.new('RGB', (width, height), 'white')
            canvas.paste(image, (0, 0))
            image = canvas
        padded.append(image)
    padded[0].save(
        gif_path,
        save_all=True,
        append_images=padded[1:],
        duration=delay_cs * 10,
        loop=0,
    )
    if thumbnail_path:
        shutil.copy2(gif_path, thumbnail_path)


# ----------------------------------------------------------------------
# Parallel execution
# ----------------------------------------------------------------------

_SHARED = {}


def _init_worker(shared):
    _SHARED.clear()
    _SHARED.update(shared)
    mpl.use('Agg')


def _call_with_shared(func, kwargs):
    return func(**_SHARED, **kwargs)


def plot_regions_in_parallel(jobs, max_workers, shared=None):
    """
    Run plotting jobs ``[(func, kwargs), ...]`` in worker processes.

    ``shared`` holds large arrays passed to every job; they are sent once
    per worker rather than once per job.  Separate processes are used
    because pyplot keeps global state and is not thread-safe.
    """
    shared = shared or {}
    if max_workers <= 1 or len(jobs) <= 1:
        for func, kwargs in jobs:
            func(**shared, **kwargs)
        return

    prefetch_map_data()
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(max_workers, len(jobs)),
            initializer=_init_worker,
            initargs=(shared,),
        ) as pool:
            futures = [
                pool.submit(_call_with_shared, func, kwargs)
                for func, kwargs in jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # re-raise worker errors
    finally:
        _stop_forkserver()


def _stop_forkserver():
    """
    Stop multiprocessing's fork server once the plotting workers are done.

    The fork server (how Python 3.14 starts worker processes on Linux)
    listens on a socket file in multiprocessing's temporary folder,
    ``$TMPDIR/pymp-*``, and is otherwise left running until after the
    main process exits.  The folder is removed at exit while the server
    still holds the socket, which on NFS turns the socket into a ``.nfs*``
    file instead of deleting it: the removal then fails with "Directory
    not empty" and leaves the folder behind.  Stopping the server here
    deletes the socket while nothing uses it.  ``_stop`` is private, so
    this does nothing if a Python version lacks it.
    """
    server = getattr(multiprocessing.forkserver, '_forkserver', None)
    stop = getattr(server, '_stop', None)
    if stop is not None:
        stop()


def split_hours(n_hours, n_chunks):
    """Split ``range(n_hours)`` into at most ``n_chunks`` contiguous chunks."""
    n_chunks = max(1, min(n_chunks, n_hours))
    size = math.ceil(n_hours / n_chunks)
    return [
        list(range(start, min(start + size, n_hours)))
        for start in range(0, n_hours, size)
    ]


# ----------------------------------------------------------------------
# Operational products
# ----------------------------------------------------------------------


def mtd_plot_all_fcst(
    grib_path_des,
    fig_path_s,
    latlon_dims,
    pre_acc,
    hrs,
    raw_thres,
    curdate,
    data_success,
    load_data_nc,
    lat,
    lon,
    simp_bin,
    simp_prop,
    sigma,
    hour_indices=None,
    summary=True,
    animate=True,
):
    """
    Ensemble object probability figures.

    1) Object probability (shading) with object tracks colored by intensity
       and marked by model; 2) the same colored by forecast hour; 3) an
       hourly series of 1) (plus an animated GIF).  Observed objects, if
       present, are outlined in black.  For the '_ALL' region the hourly
       probabilities and tracks are also written as GeoJSON.

    Parameters
    ----------
    grib_path_des : str
        Figure name prefix including the region, e.g.
        'MTD_QPF_HRRRTLE_EXT_NW'.
    fig_path_s : str
        Output directory.
    latlon_dims : list
        Plot bounds [WLON, SLAT, ELON, NLAT].
    pre_acc : float
        Accumulation interval (hours).
    hrs : numpy.ndarray
        Verification hours.
    raw_thres : float
        Object threshold (inches).
    curdate : datetime.datetime
        Cycle time.
    data_success : numpy.ndarray
        (hour, model): 0 loaded, 1 missing, NaN beyond the model's range.
    load_data_nc : list of str
        Model names with '.nc' appended.
    lat, lon : numpy.ndarray
        Grid coordinates.
    simp_bin : numpy.ndarray
        Simple object IDs (y, x, hour, model).
    simp_prop : list
        Object attributes of each model.
    sigma : float
        Gaussian smoothing of the probabilities (grid points).
    hour_indices : list of int, optional
        Hourly figures to make (default all).
    summary : bool
        Make the two all-hour summary figures.
    animate : bool
        Assemble the hourly figures into a GIF.
    """
    _time_index_to_hours(simp_prop, load_data_nc)
    try:
        _plot_all_fcst(
            grib_path_des,
            fig_path_s,
            latlon_dims,
            pre_acc,
            hrs,
            raw_thres,
            curdate,
            data_success,
            load_data_nc,
            lat,
            lon,
            simp_bin,
            simp_prop,
            sigma,
            hour_indices,
            summary,
            animate,
        )
    finally:
        _hours_to_time_index(simp_prop, load_data_nc)


def _plot_all_fcst(
    grib_path_des,
    fig_path_s,
    latlon_dims,
    pre_acc,
    hrs,
    raw_thres,
    curdate,
    data_success,
    load_data_nc,
    lat,
    lon,
    simp_bin,
    simp_prop,
    sigma,
    hour_indices,
    summary,
    animate,
):
    is_model = _is_model(load_data_nc)
    obs_models = [i for i in range(len(is_model)) if not is_model[i]]
    stamp = f'{curdate:%Y%m%d%H}'
    tag = f'_p{pre_acc:.2f}_t{raw_thres}'

    colorlist = INTENSITY_COLORS
    cmap_intensity = mpl.colors.ListedColormap(colorlist)
    names_bymod, symbols_bymod = _model_markers(load_data_nc)

    centroid_values = np.round(
        np.linspace(0, 1 * 10 * raw_thres, len(colorlist)), 2
    )
    leg_nam_byint = [
        '90th Object Percentile >=' + str(x) + '"' for x in centroid_values
    ]
    step = int(round(len(hrs) / len(colorlist)))
    leg_nam_bytime = [
        'Hours ' + str(hrs[i]) + ' - ' + str(hrs[i + step])
        for i in range(0, len(hrs) - step, step + 1)
    ]
    cmap_time = mpl.colors.ListedColormap(colorlist[0 : len(leg_nam_bytime)])
    cmap_prob = _blues(0.3)
    values = np.linspace(0, 1, 11)
    vmax_int = np.nanmax(centroid_values)

    def draw_obs_outline(ax, field):
        ax.contour(
            lon,
            lat,
            field > 0,
            levels=values,
            cmap=mpl.colors.ListedColormap(['#000000']),
            linewidths=2,
            transform=PLATE,
            transform_first=True,
        )

    def draw_tracks(ax, color_by_time):
        for model in range(len(load_data_nc)):
            marker = _marker_for(
                model, load_data_nc, names_bymod, symbols_bymod
            )
            if not _has_data(simp_prop[model]):
                continue
            prop = simp_prop[model]
            keep, linecolor, linewidth, markersize, zorder = _object_style(
                is_model[model]
            )
            for obj in keep(np.unique(prop[0])):
                ind = prop[0] == obj
                if color_by_time:
                    color, cmap, vmax, sc_lw, alpha_one = (
                        prop[2],
                        cmap_time,
                        hrs[-1],
                        0.25,
                        1,
                    )
                else:
                    color, cmap, vmax, sc_lw, alpha_one = (
                        prop[9],
                        cmap_intensity,
                        vmax_int,
                        None,
                        0.8,
                    )
                if len(prop[4][ind]) > 2:
                    ax.plot(
                        prop[5][ind],
                        prop[4][ind],
                        color='k',
                        linewidth=linewidth,
                        zorder=zorder,
                        transform=PLATE,
                    )
                    ax.scatter(
                        prop[5][ind],
                        prop[4][ind],
                        c=color[ind],
                        marker=marker,
                        vmin=0,
                        vmax=vmax,
                        s=markersize,
                        linewidth=sc_lw or linewidth,
                        cmap=cmap,
                        zorder=zorder,
                        alpha=1,
                        transform=PLATE,
                    )
                else:
                    ax.plot(
                        prop[5][ind][0],
                        prop[4][ind][0],
                        color='k' if color_by_time else linecolor,
                        linewidth=linewidth if color_by_time else 1,
                        zorder=zorder,
                        transform=PLATE,
                    )
                    ax.scatter(
                        prop[5][ind][0],
                        prop[4][ind][0],
                        c=color[ind][0],
                        marker=marker,
                        vmin=0,
                        vmax=vmax,
                        s=markersize,
                        linewidth=sc_lw or linewidth,
                        cmap=cmap,
                        zorder=zorder,
                        alpha=alpha_one,
                        transform=PLATE,
                    )

    def draw_model_legend(ax):
        if len(names_bymod) > 0:
            _marker_legend(ax, symbols_bymod, names_bymod)

    title = (
        'Ensemble Object Probabilities - Precip. >= '
        + str(raw_thres)
        + '" \n '
        f'{pre_acc:.2f} Hour Acc. Precip. - Initialized {curdate:%Y%m%d%H}'
    )

    if summary:
        # Only members with data at a given hour count towards its probability
        # (hours past a lagged member's last forecast hour are NaN).
        members_by_hour = is_model & (data_success == 0)
        ens_mean = _max_member_mean(simp_bin, members_by_hour)
        ens_mean[ens_mean > 0.95] = 0.95
        prob = gaussian_filter(ens_mean * 100, sigma)
        n_members = _members_found(data_success[:, is_model])
        members_text = f'{n_members} Members Found'
        valid_text = (
            f'Valid Time: {_valid(curdate, hrs[0])}'
            f' - {curdate + datetime.timedelta(hours=hrs[-1]):%Y%m%d%H%M}'
        )
        obs_all_hours = None
        if obs_models:
            obs_all_hours = np.squeeze(
                np.mean(
                    _nan_where_missing(simp_bin, data_success, obs_models),
                    axis=2,
                )
            )

        # 1) Probability, tracks colored by intensity, marked by model.
        fig, ax = _new_map()
        region_all = '_ALL' in grib_path_des
        if not region_all:  # keep the CONUS-wide figure simple
            ax.text(
                0.5,
                0.03,
                members_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='bottom',
                horizontalalignment='center',
                bbox=TEXT_BOX,
            )
            ax.text(
                0.5,
                0.97,
                valid_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='center',
                bbox=TEXT_BOX,
            )
        cs = ax.contourf(
            lon,
            lat,
            prob,
            levels=values * 100,
            extend='min',
            cmap=cmap_prob,
            antialiased=False,
            vmin=np.spacing(0.0),
            vmax=np.nanmax(values * 100),
            transform=PLATE,
            transform_first=True,
        )
        fig.colorbar(cs, ax=ax)
        if obs_models:
            draw_obs_outline(ax, obs_all_hours)
        draw_tracks(ax, color_by_time=False)
        if not region_all:
            _intensity_legend(ax, colorlist, leg_nam_byint)
            draw_model_legend(ax)
        ax.set_title(title, fontsize=13)
        if region_all:
            # Outline the regional sub-domains (the web page links to them).
            for xs, ys in (
                ([-110, -110, -110], [20, 36, 52]),
                ([-85, -85, -85], [20, 36, 52]),
                ([-130, -97, -64], [38, 38, 38]),
            ):
                ax.plot(
                    np.array(xs),
                    np.array(ys),
                    color='k',
                    linewidth=3,
                    zorder=3,
                    alpha=0.5,
                    transform=PLATE,
                )
            for x_pos, y_pos, label in (
                ((1.0 / 6.0) - 0.02, 0.95, 'North West'),
                (0.5, 0.95, 'North Central'),
                ((5.0 / 6.0) + 0.02, 0.95, 'North East'),
                ((1.0 / 6.0) - 0.02, 0.1, 'South West'),
                (0.5, 0.1, 'South Central'),
                ((5.0 / 6.0) + 0.02, 0.1, 'South East'),
            ):
                ax.text(
                    x_pos,
                    y_pos,
                    f'{label} Domain \n Click to Zoom in',
                    transform=ax.transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    zorder=5,
                    horizontalalignment='center',
                    bbox=REGION_BOX,
                )
        _finish_map(
            fig,
            ax,
            latlon_dims,
            _fig_file(
                fig_path_s,
                f'{grib_path_des}_TOTENS_objprob_bymodel_{stamp}{tag}.png',
            ),
            dpi=150,
        )

        # 2) Probability, tracks colored by forecast hour, marked by model.
        fig, ax = _new_map()
        ax.text(
            0.5,
            0.03,
            members_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='center',
            bbox=TEXT_BOX,
        )
        ax.text(
            0.5,
            0.97,
            valid_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='center',
            bbox=TEXT_BOX,
        )
        cs = ax.contourf(
            lon,
            lat,
            prob,
            levels=values * 100,
            extend='min',
            cmap=cmap_prob,
            antialiased=False,
            vmin=np.spacing(0.0),
            vmax=np.nanmax(values * 100),
            transform=PLATE,
            transform_first=True,
        )
        fig.colorbar(cs, ax=ax)
        if obs_models:
            draw_obs_outline(ax, obs_all_hours)
        draw_tracks(ax, color_by_time=True)
        _intensity_legend(ax, colorlist, leg_nam_bytime)
        draw_model_legend(ax)
        ax.set_title(title, fontsize=13)
        _finish_map(
            fig,
            ax,
            latlon_dims,
            _fig_file(
                fig_path_s,
                f'{grib_path_des}_TOTENS_objprob_bytime_{stamp}{tag}.png',
            ),
            dpi=150,
        )

    # 3) Hourly probability with the objects at that hour and their tracks.
    if hour_indices is None:
        hour_indices = range(len(hrs))
    for hr_count in hour_indices:
        _plot_hourly_prob(
            grib_path_des,
            fig_path_s,
            latlon_dims,
            pre_acc,
            hrs,
            raw_thres,
            curdate,
            data_success,
            load_data_nc,
            lat,
            lon,
            simp_bin,
            simp_prop,
            sigma,
            hr_count,
            is_model,
            obs_models,
            names_bymod,
            symbols_bymod,
            leg_nam_byint,
            vmax_int,
            cmap_intensity,
            cmap_prob,
            title,
        )

    if animate:
        animate_all_fcst(
            grib_path_des, fig_path_s, pre_acc, hrs, raw_thres, curdate
        )


def animate_all_fcst(
    grib_path_des, fig_path_s, pre_acc, hrs, raw_thres, curdate
):
    """GIF of the hourly `mtd_plot_all_fcst` figures (+ date-free copy)."""
    stem = (
        f'{grib_path_des}_TOTENS_objprob_byhour_{curdate:%Y%m%d%H}'
        f'_p{pre_acc:.2f}_t{raw_thres}'
    )
    frames = [
        _fig_file(fig_path_s, f'{stem}_f{hr_count + 1:02d}.png')
        for hr_count in range(len(hrs))
    ]
    write_gif(
        frames,
        _fig_file(fig_path_s, f'{stem}.gif'),
        delay_cs=100,
        thumbnail_path=_fig_file(
            fig_path_s,
            f'{grib_path_des}_TOTENS_objprob_byhour'
            f'_p{pre_acc:.2f}_t{raw_thres}.gif',
        ),
    )


def _plot_hourly_prob(
    grib_path_des,
    fig_path_s,
    latlon_dims,
    pre_acc,
    hrs,
    raw_thres,
    curdate,
    data_success,
    load_data_nc,
    lat,
    lon,
    simp_bin,
    simp_prop,
    sigma,
    hr_count,
    is_model,
    obs_models,
    names_bymod,
    symbols_bymod,
    leg_nam_byint,
    vmax_int,
    cmap_intensity,
    cmap_prob,
    title,
):
    """One hourly object probability figure (and GeoJSON for '_ALL')."""
    hr_inc = hrs[hr_count]
    stamp = f'{curdate:%Y%m%d%H}'
    name = (
        f'{grib_path_des}_TOTENS_objprob_byhour_{stamp}_p{pre_acc:.2f}'
        f'_t{raw_thres}_f{hr_count + 1:02d}'
    )
    region_all = '_ALL' in grib_path_des
    values = np.linspace(0, 1, 11)

    members = is_model & (data_success[hr_count, :] == 0)
    ens_mean = _member_mean(simp_bin, members, hr_count)
    ens_mean[ens_mean > 0.95] = 0.95

    fig, ax = _new_map()
    ax.text(
        0.5,
        0.03,
        f'{int(np.sum(members))} Members Found',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='bottom',
        horizontalalignment='center',
        bbox=TEXT_BOX,
    )
    ax.text(
        0.5,
        0.97,
        f'Valid Time: {_valid(curdate, hr_inc)}'
        f' - {pre_acc:.2f} Hour Accumulation ',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='center',
        bbox=TEXT_BOX,
    )
    # transform_first=False keeps the contour paths in lon/lat for the GeoJSON.
    cs = ax.contourf(
        lon,
        lat,
        gaussian_filter(ens_mean * 100, sigma),
        levels=values * 100,
        extend='min',
        cmap=cmap_prob,
        antialiased=False,
        vmin=np.spacing(0.0),
        vmax=np.nanmax(values * 100),
        transform=PLATE,
        transform_first=False,
    )
    if region_all:
        geojsoncontour.contourf_to_geojson(
            contourf=cs,
            geojson_filepath=_fig_file(
                fig_path_s, f'{name}_shadedprobs.geojson'
            ),
        )

    if obs_models:
        obs_now = _nan_where_missing(
            simp_bin[:, :, hr_count, :], data_success[hr_count, :], obs_models
        )
        ax.contour(
            lon,
            lat,
            np.squeeze(obs_now, axis=2) > 0,
            levels=values,
            cmap=mpl.colors.ListedColormap(['#000000']),
            linewidths=2,
            transform=PLATE,
            transform_first=True,
        )

    # Objects at this hour (colored by intensity) and their full tracks.
    features = []
    for model in range(len(load_data_nc)):
        marker = _marker_for(model, load_data_nc, names_bymod, symbols_bymod)
        if not _has_data(simp_prop[model]):
            continue
        prop = simp_prop[model]
        model_name = load_data_nc[model].replace('.nc', '')
        at_hour = prop[2] == hr_inc
        keep, linecolor, linewidth, markersize, zorder = _object_style(
            is_model[model]
        )
        for obj in keep(np.unique(prop[0][at_hour])):
            snap = np.asarray(prop[0] == obj) & np.asarray(prop[2] == hr_inc)
            track = np.asarray(prop[0] == obj)

            many = len(prop[5][snap]) > 2
            sel = snap if many else (np.flatnonzero(snap)[:1])
            ax.plot(
                prop[5][sel] if many else prop[5][sel][0],
                prop[4][sel] if many else prop[4][sel][0],
                color=linecolor,
                linewidth=linewidth,
                zorder=zorder,
                transform=PLATE,
            )
            ax.scatter(
                prop[5][sel] if many else prop[5][sel][0],
                prop[4][sel] if many else prop[4][sel][0],
                c=prop[9][sel] if many else prop[9][sel][0],
                marker=marker,
                vmin=0,
                vmax=vmax_int,
                s=markersize,
                linewidth=0.25,
                cmap=cmap_intensity,
                zorder=zorder,
                alpha=1 if many else 0.8,
                transform=PLATE,
            )
            first = np.flatnonzero(snap)[0]
            features.append(
                geojson.Feature(
                    geometry=geojson.Point(
                        [float(prop[5][first]), float(prop[4][first])]
                    ),
                    properties={
                        'model': model_name,
                        'object_id': prop[0][first],
                        'object_cat:': prop[1][first],
                        'time_index:': prop[2][first],
                        'area': prop[3][first],
                        'axis_ang': prop[6][first],
                        '10th_percentile': prop[7][first],
                        '50th_percentile': prop[8][first],
                        '90th_percentile': prop[9][first],
                    },
                )
            )

            if len(prop[5][track]) > 2:
                ax.plot(
                    prop[5][track],
                    prop[4][track],
                    color=linecolor,
                    linewidth=linewidth - 0.25,
                    zorder=zorder,
                    transform=PLATE,
                )
                coords = [
                    [float(x), float(y)]
                    for x, y in zip(
                        prop[5][track], prop[4][track], strict=True
                    )
                ]
            else:
                ax.plot(
                    prop[5][track][0],
                    prop[4][track][0],
                    color=linecolor,
                    linewidth=linewidth - 0.25,
                    zorder=zorder,
                    transform=PLATE,
                )
                coords = [[float(prop[5][track][0]), float(prop[4][track][0])]]
            features.append(
                geojson.Feature(
                    geometry=geojson.LineString(coords),
                    properties={'model': model_name},
                )
            )

    if region_all:
        with open(
            _fig_file(fig_path_s, f'{name}_tracks.geojson'), 'w'
        ) as fout:
            fout.write(geojson.dumps(geojson.FeatureCollection(features)))

    _intensity_legend(ax, INTENSITY_COLORS, leg_nam_byint)
    if len(names_bymod) > 0:
        _marker_legend(ax, symbols_bymod, names_bymod)
    fig.colorbar(cs, ax=ax)
    ax.set_title(title, fontsize=13)
    _finish_map(
        fig, ax, latlon_dims, _fig_file(fig_path_s, f'{name}.png'), dpi=150
    )


def mtd_plot_tle_fcst(
    grib_path_des,
    fig_path_s,
    latlon_dims,
    pre_acc,
    hrs,
    raw_thres,
    curdate,
    data_success,
    load_data_nc,
    lat,
    lon,
    simp_bin,
    simp_prop,
    sigma,
):
    """
    One figure per time-lagged ensemble (TLE): object probability of the
    TLE members (shading) and each member's object tracks, colored by
    intensity with a marker per lag.  Supports up to 9 members per TLE.
    Parameters are as for `mtd_plot_all_fcst`.
    """
    _time_index_to_hours(simp_prop, load_data_nc)
    try:
        _plot_tle_fcst(
            grib_path_des,
            fig_path_s,
            latlon_dims,
            pre_acc,
            hrs,
            raw_thres,
            curdate,
            data_success,
            load_data_nc,
            lat,
            lon,
            simp_bin,
            simp_prop,
            sigma,
        )
    finally:
        _hours_to_time_index(simp_prop, load_data_nc)


def _plot_tle_fcst(
    grib_path_des,
    fig_path_s,
    latlon_dims,
    pre_acc,
    hrs,
    raw_thres,
    curdate,
    data_success,
    load_data_nc,
    lat,
    lon,
    simp_bin,
    simp_prop,
    sigma,
):
    is_model = _is_model(load_data_nc)
    has_obs = np.sum(~is_model) > 0
    cmap_intensity = mpl.colors.ListedColormap(INTENSITY_COLORS)
    cmap_prob = _blues(0.3)
    centroid_values = np.round(
        np.linspace(0, 1 * 10 * raw_thres, len(INTENSITY_COLORS)), 2
    )
    vmax_int = np.nanmax(centroid_values)
    leg_nam_byint = [
        '90th Object Percentile >=' + str(x) + '"' for x in centroid_values
    ]
    if has_obs:
        leg_sym_bymem = ['s', 'd', 'o', '*', '^', '1', '2', '3', '4']
        mod_inc = 1  # the first marker belongs to the observation
    else:
        leg_sym_bymem = ['d', 'o', '*', '^', '1', '2', '3', '4']
        mod_inc = 0
    values = np.linspace(0, 1, 11)
    valid_text = (
        f'Valid Time: {_valid(curdate, hrs[0])}'
        f' - {curdate + datetime.timedelta(hours=hrs[-1]):%Y%m%d%H%M}'
    )

    # Group the members of each TLE (same name apart from the lag).
    short_names = [name[:-9] for name in load_data_nc]
    lags = np.array(
        [int(name[name.index('lag') + 3 : -3]) for name in load_data_nc]
    )
    groups, indices = np.unique(short_names, return_inverse=True)

    ind_count = 0
    for group in range(len(groups)):
        current = indices == group
        data_success_p = data_success[:, current]
        if not (np.sum(current) > 1 and np.any(data_success_p == 0)):
            continue  # not a TLE, or no data

        load_data_nc_p = [
            load_data_nc[i] for i in range(len(current)) if current[i]
        ]
        simp_prop_p = [simp_prop[i] for i in range(len(current)) if current[i]]
        lag_p = lags[current]

        # Newest (smallest lag) first.
        order = np.argsort(lag_p)[::-1]
        data_success_k = data_success_p[:, order]
        load_data_nc_k = [load_data_nc_p[i] for i in order]
        simp_prop_k = [
            simp_prop_p[i][:, simp_prop_p[i][0, :] > 0]
            if _has_data(simp_prop_p[i])
            else np.nan
            for i in order
        ]

        members_by_hour = current & (data_success == 0)
        ens_mean = _max_member_mean(simp_bin, members_by_hour)
        ens_mean[ens_mean > 0.95] = 0.95

        if has_obs:
            leg_nam_bymem = np.append(
                'ST4', [str(i) + ' Hours Lag' for i in lag_p]
            )
        else:
            leg_nam_bymem = np.array([str(i) + ' Hours Lag' for i in lag_p])

        fig, ax = _new_map()
        n_found = _members_found(data_success_k)
        ax.text(
            0.5,
            0.03,
            f'{load_data_nc_k[0][:-9]} - {n_found} Members Found',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='center',
            bbox=TEXT_BOX,
        )
        ax.text(
            0.5,
            0.97,
            valid_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='center',
            bbox=TEXT_BOX,
        )
        cs = ax.contourf(
            lon,
            lat,
            gaussian_filter(ens_mean * 100, sigma),
            levels=values * 100,
            extend='min',
            cmap=cmap_prob,
            antialiased=False,
            vmin=np.spacing(0.0),
            vmax=np.nanmax(values) * 100,
            transform=PLATE,
            transform_first=True,
        )

        leg_sym_use = []
        if has_obs:
            obs_models = list(np.flatnonzero(~is_model))
            obs_all_hours = np.squeeze(
                np.mean(
                    _nan_where_missing(simp_bin, data_success, obs_models),
                    axis=2,
                )
            )
            ax.contour(
                lon,
                lat,
                obs_all_hours > 0,
                levels=values,
                cmap=mpl.colors.ListedColormap(['#000000']),
                linewidths=2,
                transform=PLATE,
                transform_first=True,
            )

            obs_inc = obs_models[0]
            if _has_data(simp_prop[obs_inc]):
                prop = simp_prop[obs_inc]
                ids = np.unique(prop[0])
                for obj in ids[ids < 0]:
                    ind = prop[0] == obj
                    first = slice(None) if len(prop[5][ind]) > 2 else 0
                    ax.plot(
                        prop[5][ind][first],
                        prop[4][ind][first],
                        color='b',
                        linewidth=1.5,
                        zorder=3,
                        transform=PLATE,
                    )
                    ax.scatter(
                        prop[5][ind][first],
                        prop[4][ind][first],
                        c=prop[9][ind][first],
                        marker=leg_sym_bymem[obs_inc],
                        vmin=0,
                        vmax=vmax_int,
                        s=60,
                        linewidth=0.25,
                        cmap=cmap_intensity,
                        zorder=3,
                        alpha=0.8,
                        transform=PLATE,
                    )
                leg_sym_use.append(leg_sym_bymem[obs_inc])

        marker_inc = list(range(len(order) - 1, -1, -1))
        for model, prop in enumerate(simp_prop_k):
            marker = leg_sym_bymem[marker_inc[model] + mod_inc]
            if _has_data(prop):
                ids = np.unique(prop[0])
                for obj in ids[ids >= 0]:
                    ind = prop[0] == obj
                    first = slice(None) if len(prop[5][ind]) > 2 else 0
                    ax.plot(
                        prop[5][ind][first],
                        prop[4][ind][first],
                        color='k',
                        linewidth=0.5,
                        zorder=2,
                        transform=PLATE,
                    )
                    ax.scatter(
                        prop[5][ind][first],
                        prop[4][ind][first],
                        c=prop[9][ind][first],
                        marker=marker,
                        vmin=0,
                        vmax=vmax_int,
                        s=30,
                        linewidth=0.25,
                        cmap=cmap_intensity,
                        zorder=2,
                        alpha=0.8,
                        transform=PLATE,
                    )
            leg_sym_use.append(marker)

        _intensity_legend(ax, INTENSITY_COLORS, leg_nam_byint, markersize=5)
        n_legend = _members_found(data_success_k) + mod_inc
        ax.add_artist(
            _marker_legend(
                ax,
                leg_sym_use[:n_legend],
                leg_nam_bymem[:n_legend],
                markersize=5,
            )
        )

        fig.colorbar(cs, ax=ax)
        ax.set_title(
            'Objects/Max. Locations - Precip. >= ' + str(raw_thres) + '" \n '
            f'{pre_acc:.2f} Hour Acc. Precip. - '
            f'Initialized {curdate:%Y%m%d%H}',
            fontsize=13,
        )
        _finish_map(
            fig,
            ax,
            latlon_dims,
            _fig_file(
                fig_path_s,
                f'{grib_path_des}_TLE_obj_{curdate:%Y%m%d%H}'
                f'_p{pre_acc:.2f}_t{raw_thres}_group{ind_count}.png',
            ),
            dpi=150,
        )
        ind_count += 1


SNOW_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def _snow_outline_color(intensity):
    """Outline color for a snow object's 90th percentile intensity (inches)."""
    if not intensity >= 0:  # NaN or negative: no outline
        return None
    return SNOW_COLORS[bisect.bisect_right(SNOW_THRESHOLDS, intensity)]


def mtd_plot_all_snow_fcst(
    grib_path_des,
    fig_path_s,
    latlon_dims,
    pre_acc,
    hrs,
    raw_thres,
    curdate,
    data_success,
    load_data_nc,
    lat,
    lon,
    simp_bin,
    simp_prop,
    hour_indices=None,
    animate=True,
):
    """
    Hourly figures of every model's snow objects, outlined in a color for
    the object's 90th percentile intensity, plus an animated GIF.
    Parameters are as for `mtd_plot_all_fcst`.
    """
    _time_index_to_hours(simp_prop, load_data_nc)
    try:
        is_model = _is_model(load_data_nc)
        model_idx = np.flatnonzero(is_model)
        stamp = f'{curdate:%Y%m%d%H}'
        tag = f'_p{pre_acc:.2f}_t{raw_thres}'
        cmap = mpl.colors.ListedColormap(SNOW_COLORS)
        values = np.linspace(0, 0.55, 12)
        if hour_indices is None:
            hour_indices = range(len(hrs))

        for hr_count in hour_indices:
            hr_inc = hrs[hr_count]
            fig, ax = _new_map()
            n_found = int(
                np.sum((data_success[hr_count, is_model] == 0) * 1, axis=0)
            )
            ax.text(
                0.5,
                0.03,
                f'{n_found} Members Found',
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='bottom',
                horizontalalignment='center',
                bbox=TEXT_BOX,
            )
            ax.text(
                0.5,
                0.97,
                f'Valid Time: {_valid(curdate, hr_inc)}',
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='center',
                bbox=TEXT_BOX,
            )

            # Every object gets the same (all-white) fill, which only serves
            # the colorbar, plus an outline colored by its intensity.
            filled = None
            for position, model in enumerate(model_idx):
                ids_grid = simp_bin[:, :, hr_count, model].astype(float)
                if data_success[hr_count, model] == 1:
                    ids_grid += np.nan
                if np.sum(ids_grid) == 0:
                    ids_grid[0, 0] = 1  # guarantees a fill for the colorbar
                try:
                    prop = simp_prop[position]
                    inten = prop[9][prop[2] == hr_inc]
                    obj_ids = np.unique(ids_grid)
                except TypeError:
                    inten, obj_ids = [0], [0, 0]
                if len(inten) == 0:
                    inten, obj_ids = [0], [0, 0]

                for obj_count, intensity in enumerate(inten, start=1):
                    if obj_count >= len(obj_ids):
                        break
                    mask = ids_grid == obj_ids[obj_count] * 1
                    if filled is None:
                        with contextlib.suppress(Exception):
                            filled = ax.contourf(
                                lon,
                                lat,
                                mask * 0,
                                levels=values,
                                cmap=cmap,
                                antialiased=False,
                                vmin=np.nanmin(values),
                                vmax=np.nanmax(values),
                                transform=PLATE,
                                transform_first=True,
                            )
                    color = _snow_outline_color(intensity)
                    if color is None:
                        continue
                    with contextlib.suppress(Exception):
                        ax.contour(
                            lon,
                            lat,
                            mask,
                            antialiased=False,
                            colors=color,
                            linewidths=0.8,
                            transform=PLATE,
                            transform_first=True,
                        )

            if filled is not None:
                fig.colorbar(filled, ax=ax)
            ax.set_title(
                'All Snow Precip. Objects >= ' + str(raw_thres) + '" \n '
                f'{pre_acc:.2f} Hour Acc. Precip. at Hour {hr_inc:.2f} - '
                'Initialized '
                f'{curdate:%Y%m%d%H}',
                fontsize=13,
            )
            _finish_map(
                fig,
                ax,
                latlon_dims,
                _fig_file(
                    fig_path_s,
                    f'{grib_path_des}_snowobj_byhour_{stamp}{tag}'
                    f'_h{hr_count + 1:02d}.png',
                ),
                dpi=150,
            )

        if animate:
            animate_all_snow_fcst(
                grib_path_des, fig_path_s, pre_acc, hrs, raw_thres, curdate
            )
    finally:
        _hours_to_time_index(simp_prop, load_data_nc)


def animate_all_snow_fcst(
    grib_path_des, fig_path_s, pre_acc, hrs, raw_thres, curdate
):
    """GIF of the hourly `mtd_plot_all_snow_fcst` figures (+ copy)."""
    stem = (
        f'{grib_path_des}_snowobj_byhour_{curdate:%Y%m%d%H}'
        f'_p{pre_acc:.2f}_t{raw_thres}'
    )
    frames = [
        _fig_file(fig_path_s, f'{stem}_h{hr_count + 1:02d}.png')
        for hr_count in range(len(hrs))
    ]
    write_gif(
        frames,
        _fig_file(fig_path_s, f'{stem}.gif'),
        delay_cs=100,
        thumbnail_path=_fig_file(
            fig_path_s,
            f'{grib_path_des}_snowobj_byhour_p{pre_acc:.2f}_t{raw_thres}.gif',
        ),
    )


# ----------------------------------------------------------------------
# Retrospective / bias look-up products
# ----------------------------------------------------------------------


def _retro_setup(load_data_nc, simp_bin, data_success, raw_thres):
    """Common set-up of the bias look-up figures."""
    is_model = _is_model(load_data_nc)
    colorlist_mag = INTENSITY_COLORS
    values_mag = np.round(
        np.linspace(0, 1 * 10 * raw_thres, len(colorlist_mag)), 2
    )
    leg_nam_byint = [
        '90th Object Percentile >=' + str(x) + '"' for x in values_mag
    ]
    names_bymod, symbols_bymod = _model_markers(load_data_nc)
    return (
        is_model,
        colorlist_mag,
        values_mag,
        mpl.colors.ListedColormap(colorlist_mag),
        leg_nam_byint,
        names_bymod,
        symbols_bymod,
    )


def mtd_plot_retro_disp_vect(
    grib_path_des,
    fig_path_s,
    latlon_dims,
    pre_acc,
    hrs,
    thres,
    curdate,
    data_success,
    load_data_nc,
    lat,
    lon,
    simp_bin,
    simp_prop,
    pair_diff,
    sigma,
):
    """
    Ensemble object probability with each operational object's median
    retrospective displacement (arrow tip) and intensity adjustment (arrow
    color), from the bias look-up tables.  MJE. 20210409-20210803.

    Parameters are as for `mtd_plot_all_fcst`; ``pair_diff`` holds the
    retrospective model-minus-analysis differences for each [model][track].
    """
    _time_index_to_hours(simp_prop, load_data_nc)
    try:
        (
            is_model,
            colorlist_mag,
            values_mag,
            cmap_mag,
            leg_nam_byint,
            names_bymod,
            symbols_bymod,
        ) = _retro_setup(load_data_nc, simp_bin, data_success, thres)
        cmap_ens = _blues(0.3, 0.8)
        values_ens = np.linspace(0, 1, 11)
        stamp = f'{curdate:%Y%m%d%H}'
        tag = f'_p{pre_acc:.2f}_t{thres}'

        frames = []
        for hr_count, hr_inc in enumerate(hrs):
            members = is_model & (data_success[hr_count, :] == 0)
            ens_mean = _member_mean(simp_bin, members, hr_count)
            ens_mean[ens_mean > 0.95] = 0.95

            fig, ax = _new_map()
            cs = ax.contourf(
                lon,
                lat,
                gaussian_filter(ens_mean * 100, sigma),
                levels=values_ens * 100,
                extend='min',
                cmap=cmap_ens,
                antialiased=False,
                vmin=np.nanmin(values_ens) * 100,
                vmax=np.nanmax(values_ens) * 100,
                transform=PLATE,
                transform_first=True,
            )
            fig.colorbar(cs, ax=ax)

            for model in range(len(load_data_nc)):
                marker = _marker_for(
                    model, load_data_nc, names_bymod, symbols_bymod
                )
                if not _has_data(simp_prop[model]):
                    continue
                prop = simp_prop[model]
                keep, _, _, markersize, zorder = _object_style(is_model[model])
                markersize, zorder = (30, 2) if is_model[model] else (60, 3)
                linecolor = 'k' if is_model[model] else 'b'
                for obj in keep(np.unique(prop[0])):
                    snap = np.asarray(prop[0] == obj) & np.asarray(
                        prop[2] == hr_inc
                    )
                    track = np.asarray(prop[0] == obj)
                    # Skip objects absent at this hour or with short tracks.
                    if np.nansum(snap) == 0 or np.nansum(track) <= 5:
                        continue
                    latp = np.array([prop[4][snap][0]])
                    lonp = np.array([prop[5][snap][0]])
                    intp = np.array([prop[9][snap][0]])
                    retro = pair_diff[model][np.where(snap)[0][0]]

                    if (
                        len(retro) != 0 and retro.shape[0] > 10
                    ):  # enough retro samples
                        # km -> degrees on a copy (pair_diff is reused)
                        retro = np.array(retro, dtype=float)
                        retro[:, 0] = retro[:, 0] / 110.574
                        retro[:, 1] = retro[:, 1] / (
                            111.320 * math.cos(latp[0] * (math.pi / 180))
                        )
                        ydiff = np.nanmedian(retro[:, 0])
                        xdiff = np.nanmedian(retro[:, 1])
                        idiff = np.nanmedian(retro[:, 4])
                    else:
                        ydiff = xdiff = idiff = 0.0

                    ax.scatter(
                        lonp,
                        latp,
                        c=intp - np.nanmean(np.diff(values_mag)),
                        marker=marker,
                        vmin=np.nanmin(values_mag),
                        vmax=np.nanmax(values_mag),
                        s=markersize,
                        linewidth=0.25,
                        cmap=cmap_mag,
                        zorder=zorder + 1,
                        alpha=1,
                        transform=PLATE,
                    )
                    ax.plot(
                        prop[5][track],
                        prop[4][track],
                        color=linecolor,
                        linewidth=1,
                        zorder=zorder,
                        transform=PLATE,
                    )
                    # idiff is the negative of model - obs, so add it.
                    arrow_color = colorlist_mag[
                        np.argmax(values_mag >= intp[0] + idiff) - 1
                    ]
                    ax.arrow(
                        lonp[0],
                        latp[0],
                        xdiff,
                        ydiff,
                        facecolor=arrow_color,
                        width=0,
                        head_width=0.6,
                        head_length=0.2,
                        linewidth=0.4,
                        length_includes_head=True,
                        zorder=zorder + 1,
                        transform=PLATE,
                    )

            ax.text(
                0.5,
                0.03,
                f'{int(np.sum(members))} Members Found',
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='bottom',
                horizontalalignment='center',
                bbox=TEXT_BOX,
            )
            ax.text(
                0.5,
                0.97,
                f'Valid Time: {_valid(curdate, hr_inc)}'
                f' - {pre_acc:.2f} Hour Accumulation ',
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='center',
                bbox=TEXT_BOX,
            )
            _intensity_legend(ax, colorlist_mag, leg_nam_byint)
            if len(names_bymod) > 0:
                _marker_legend(ax, symbols_bymod, names_bymod)

            frame = _fig_file(
                fig_path_s,
                f'{grib_path_des}_retroplot3_{stamp}{tag}'
                f'_f{hr_count + 1:02d}.png',
            )
            _finish_map(fig, ax, latlon_dims, frame, dpi=200)
            frames.append(frame)

        write_gif(
            frames,
            _fig_file(
                fig_path_s, f'{grib_path_des}_retroplot3_{stamp}{tag}.gif'
            ),
            delay_cs=100,
            thumbnail_path=_fig_file(
                fig_path_s, f'{grib_path_des}_retroplot3_{tag[1:]}.gif'
            ),
        )
    finally:
        _hours_to_time_index(simp_prop, load_data_nc)


def mtd_plot_retro_prob(
    grib_path_des,
    fig_path_s,
    latlon_dims,
    pre_acc,
    hrs,
    thres,
    curdate,
    data_success,
    load_data_nc,
    lat,
    lon,
    simp_bin,
    simp_prop,
    pair_diff,
):
    """
    Probability that the observed object falls within 40 km of a point,
    given the operational model object, from the bias look-up tables:
    one figure per object and hour, and a CONUS-wide summed probability
    per hour.  MJE. 20210409-20220718.

    Parameters are as for `mtd_plot_retro_disp_vect` (without ``sigma``).
    """
    # A slightly larger binning radius than 40 km, smoothed a little, since
    # raw probabilities are noisy: 80 km Gaussian filter on a 1 degree grid.
    smooth_filter = 80
    grid_delta = 1.0
    levels_cntr = np.array(
        [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75]
    )
    min_elements = 8

    _time_index_to_hours(simp_prop, load_data_nc)
    try:
        (
            is_model,
            colorlist_mag,
            values_mag,
            cmap_mag,
            leg_nam_byint,
            names_bymod,
            symbols_bymod,
        ) = _retro_setup(load_data_nc, simp_bin, data_success, thres)
        cmap_ens = _blues(0.3, 0.8)
        cmap_prob = _blues(0.3)
        values_ens = np.linspace(0, 1, 11)
        stamp = f'{curdate:%Y%m%d%H}'
        tag = f'_p{pre_acc:.2f}_t{thres}'

        def regular_grid(dims):
            return np.meshgrid(
                np.linspace(
                    dims[0],
                    dims[2],
                    int(np.round((dims[2] - dims[0]) / 1, 2) + 1),
                ),
                np.linspace(
                    dims[3],
                    dims[1],
                    int(np.round((dims[3] - dims[1]) / 1, 2) + 1),
                ),
            )

        lon_prob, lat_prob = regular_grid(latlon_dims)
        grid_prob = np.zeros(
            (lon_prob.shape[0], lon_prob.shape[1], len(hrs), len(load_data_nc))
        )
        ind_snap_prev = None

        for model in range(len(load_data_nc)):
            marker = _marker_for(
                model, load_data_nc, names_bymod, symbols_bymod
            )
            if (
                not _has_data(simp_prop[model])
                or len(simp_prop[model][0]) == 0
            ):
                continue
            prop = simp_prop[model]
            keep, linecolor, _, _, _ = _object_style(is_model[model])
            linewidth, markersize, zorder = (
                (2, 30, 2) if is_model[model] else (3, 60, 3)
            )
            elements = keep(np.unique(prop[0]))
            model_label = load_data_nc[model][0:-13]
            grid_prob_full = np.zeros(
                (lon_prob.shape[0], lon_prob.shape[1], len(elements), len(hrs))
            )

            obj_count = 0
            for obj in elements:
                track = np.asarray(prop[0] == obj)
                hr_count = 0
                for hr_inc in range(len(hrs)):
                    if np.nansum(track) <= min_elements:
                        continue
                    # Skip hours outside the object's lifespan.
                    if not (
                        np.nanmin(prop[2][track])
                        <= hrs[hr_inc]
                        <= np.nanmax(prop[2][track])
                    ):
                        continue
                    if np.isnan(
                        np.nanmean(pair_diff[model][np.where(track)[0][0]])
                    ):
                        continue  # no retro statistics along the track

                    ens_mean = (simp_bin[:, :, hr_inc, model] > 0) * 1.0
                    snap = np.asarray(prop[0] == obj) & np.asarray(
                        prop[2] == hrs[hr_inc]
                    )

                    # Large objects: follow them; else fixed domain.
                    if (
                        np.nanmax(prop[5][track]) - np.nanmin(prop[5][track])
                        >= 25
                        or np.nanmax(prop[4][track])
                        - np.nanmin(prop[4][track])
                        >= 25
                    ):
                        use = (
                            snap
                            if np.nansum(snap) > 0 or ind_snap_prev is None
                            else ind_snap_prev
                        )
                        dims = np.array(
                            [
                                np.floor(np.nanmin(prop[5][use])) - 12,
                                np.floor(np.nanmin(prop[4][use])) - 7.5,
                                np.ceil(np.nanmax(prop[5][use])) + 12,
                                np.ceil(np.nanmax(prop[4][use])) + 7.5,
                            ]
                        )
                    else:
                        dims = np.array(
                            [
                                np.floor(np.nanmin(prop[5][track])) - 10,
                                np.floor(np.nanmin(prop[4][track])) - 5,
                                np.ceil(np.nanmax(prop[5][track])) + 10,
                                np.ceil(np.nanmax(prop[4][track])) + 5,
                            ]
                        )
                    lon_nat, lat_nat = regular_grid(dims)

                    try:
                        latp = np.array([prop[4][snap][0]])
                        lonp = np.array([prop[5][snap][0]])
                        intp = np.array([prop[9][snap][0]])
                        retro = pair_diff[model][np.where(snap)[0][0]]
                    except IndexError:
                        latp = lonp = intp = np.array([])
                        retro = []
                    retro_count = np.zeros(lat_nat.shape)

                    if (
                        len(retro) != 0
                        and retro.shape[0] > 4
                        and retro.shape[1] > 9
                    ):
                        # Displace the object by each retro difference.
                        latpts = latp + retro[:, 0] / 110.574
                        lonpts = lonp + retro[:, 1] / (
                            111.320 * math.cos(latp[0] * (math.pi / 180))
                        )
                        for j in range(lat_nat.shape[0] - 1):
                            for i in range(lat_nat.shape[1] - 1):
                                retro_count[j, i] = np.nansum(
                                    (latpts >= lat_nat[j + 1, i])
                                    & (latpts < lat_nat[j, i])
                                    & (lonpts >= lon_nat[j, i])
                                    & (lonpts < lon_nat[j, i + 1])
                                )
                        retro_count = retro_count / len(latpts)
                        poygm = np.array([np.nanmedian(retro[:, 9])])
                    else:
                        poygm = np.array([])

                    fig, ax = _new_map()
                    ax.contourf(
                        lon,
                        lat,
                        ens_mean * 100,
                        levels=values_ens * 100,
                        extend='min',
                        cmap=cmap_ens,
                        antialiased=False,
                        vmin=np.nanmin(values_ens) * 100,
                        vmax=np.nanmax(values_ens) * 100,
                        transform=PLATE,
                        transform_first=True,
                    )
                    ax.scatter(
                        lonp,
                        latp,
                        c=intp,
                        marker=marker,
                        vmin=np.nanmin(values_mag),
                        vmax=np.nanmax(values_mag),
                        s=markersize,
                        linewidth=0.25,
                        cmap=cmap_mag,
                        zorder=zorder + 1,
                        alpha=0.8,
                        transform=PLATE,
                    )
                    if len(poygm) > 0:
                        ax.text(
                            lonp[0] - 2,
                            latp[0] + 2,
                            'P = ' + str(int(poygm[0])),
                            fontsize=10,
                            bbox=dict(fc='white', ec='black'),
                            ha='center',
                            va='center',
                            transform=PLATE,
                        )
                    ax.plot(
                        prop[5][track],
                        prop[4][track],
                        color=linecolor,
                        linewidth=linewidth,
                        zorder=zorder,
                        transform=PLATE,
                    )

                    grid_prob_part = gaussian_filter(
                        retro_count,
                        ((smooth_filter * 2) / (111 * grid_delta)) / 2,
                    )
                    ctr = ax.contour(
                        lon_nat + 0.5,
                        lat_nat - 0.5,
                        grid_prob_part * 100,
                        levels=levels_cntr * 100,
                        colors='k',
                        linewidths=1,
                        transform=PLATE,
                        transform_first=True,
                    )
                    ax.clabel(
                        ctr,
                        ctr.levels,
                        inline=True,
                        inline_spacing=-5,
                        fmt='% 1.2f',
                        fontsize=8,
                    )
                    ax.text(
                        0.5,
                        0.03,
                        model_label,
                        transform=ax.transAxes,
                        fontsize=12,
                        verticalalignment='bottom',
                        horizontalalignment='center',
                        bbox=TEXT_BOX,
                    )
                    valid = curdate + datetime.timedelta(hours=hrs[hr_inc])
                    ax.text(
                        0.5,
                        0.97,
                        f'Valid Time: {valid:%Y-%m-%d at %H:%M UTC} - '
                        f'{pre_acc:.0f} Hour Acc. ',
                        transform=ax.transAxes,
                        fontsize=12,
                        verticalalignment='top',
                        horizontalalignment='center',
                        bbox=TEXT_BOX,
                    )
                    _intensity_legend(ax, colorlist_mag, leg_nam_byint)
                    _finish_map(
                        fig,
                        ax,
                        dims,
                        _fig_file(
                            fig_path_s,
                            f'{grib_path_des}_retroplot2_{stamp}{tag}_'
                            f'{load_data_nc[model][0:-3]}_obj{obj_count}'
                            f'_f{hr_inc + 1:02d}.png',
                        ),
                        dpi=200,
                    )

                    grid_prob_full[:, :, obj_count, hr_inc] = (
                        interpolate.griddata(
                            np.stack(
                                (lon_nat.flatten(), lat_nat.flatten()), axis=1
                            ),
                            grid_prob_part.flatten(),
                            (lon_prob, lat_prob),
                            method='linear',
                        )
                    )

                    if np.nansum(snap) > 0:
                        ind_snap_prev = snap
                    hr_count += 1
                if hr_count > 0:
                    obj_count += 1

            # Assume independence and sum over objects for each hour.
            grid_prob[:, :, :, model] = np.nansum(grid_prob_full, axis=2)

            # Simple temporal smoothing (weights sum to one).
            p = grid_prob[:, :, :, model]
            n = len(hrs)
            for h in range(n):
                if h == 0:
                    p[:, :, h] = (
                        p[:, :, h] * 0.6
                        + p[:, :, h + 1] * 0.3
                        + p[:, :, h + 2] * 0.1
                    )
                elif h == 1:
                    p[:, :, h] = (
                        p[:, :, h - 1] * 0.25
                        + p[:, :, h] * 0.4
                        + p[:, :, h + 1] * 0.25
                        + p[:, :, h + 2] * 0.1
                    )
                elif h == n - 2:
                    p[:, :, h] = (
                        p[:, :, h - 2] * 0.1
                        + p[:, :, h - 1] * 0.25
                        + p[:, :, h] * 0.4
                        + p[:, :, h + 1] * 0.25
                    )
                elif h == n - 1:
                    p[:, :, h] = (
                        p[:, :, h - 2] * 0.1
                        + p[:, :, h - 1] * 0.3
                        + p[:, :, h] * 0.6
                    )
                else:
                    p[:, :, h] = (
                        p[:, :, h - 2] * 0.1
                        + p[:, :, h - 1] * 0.2
                        + p[:, :, h] * 0.4
                        + p[:, :, h + 1] * 0.2
                        + p[:, :, h + 2] * 0.1
                    )

            # P(object within 40 km of a point), summed over objects.
            values = np.linspace(0.01, 0.15, num=15)
            for hr_inc in range(len(hrs)):
                fig, ax = _new_map()
                cs = ax.contourf(
                    lon_prob,
                    lat_prob,
                    p[:, :, hr_inc] * 100,
                    levels=values * 100,
                    extend='both',
                    cmap=cmap_prob,
                    antialiased=False,
                    vmin=np.spacing(0.0),
                    vmax=np.nanmax(values) * 100,
                    transform=PLATE,
                    transform_first=True,
                )
                fig.colorbar(cs, ax=ax)
                ax.text(
                    0.5,
                    0.03,
                    model_label,
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='bottom',
                    horizontalalignment='center',
                    bbox=TEXT_BOX,
                )
                valid = curdate + datetime.timedelta(hours=hrs[hr_inc])
                ax.text(
                    0.5,
                    0.97,
                    f'Valid Time: {valid:%Y-%m-%d at %H:%M UTC} - '
                    f'{pre_acc:.0f} Hour Acc. ',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    horizontalalignment='center',
                    bbox=TEXT_BOX,
                )
                _finish_map(
                    fig,
                    ax,
                    latlon_dims,
                    _fig_file(
                        fig_path_s,
                        f'{grib_path_des}_retroplot1_{stamp}{tag}_'
                        f'{load_data_nc[model][0:-3]}_f{hr_inc + 1:02d}.png',
                    ),
                    dpi=200,
                )
    finally:
        _hours_to_time_index(simp_prop, load_data_nc)


def mtd_plot_retro(
    grib_path_des,
    fig_path_s,
    latlon_dims,
    pre_acc,
    hrs,
    raw_thres,
    curdate,
    data_success,
    load_data_nc,
    lat,
    lon,
    fcst,
    obs,
    clus_bin,
    pair_prop,
):
    """
    Hourly paired model/analysis objects and raw precipitation for a single
    retrospective run, to compare the tracker with "by eye" results.
    MJE. 20171205.

    Parameters
    ----------
    fcst, obs : numpy.ndarray
        Raw forecast / observed precipitation (y, x, hour).
    clus_bin : numpy.ndarray
        Cluster object IDs (y, x, hour, model).
    pair_prop : list
        Paired cluster attributes of each model.
    Other parameters are as for `mtd_plot_all_fcst`.
    """
    is_model = np.array(
        ['ST4' not in name and 'MRMS' not in name for name in load_data_nc]
    )

    # With one model and one analysis, both are in the model's pair_prop.
    if np.nanmean(is_model) == 0.5 and np.mean(np.isnan(pair_prop[0])) == 1:
        pair_prop[0] = pair_prop[1] + 0

    for model in range(len(load_data_nc)):
        pair_prop[model][2] = (pair_prop[model][2] + 1) * pre_acc

    # Hours with any missing data are skipped.
    missing = np.mean(data_success, axis=1) > 0
    clus = (clus_bin > 0).astype(float)
    clus[:, :, missing, :] = np.nan

    stamp = f'{curdate:%Y%m%d%H%M}'
    tag = f'_p{pre_acc:.2f}_t{raw_thres}'
    colorlist = INTENSITY_COLORS
    cmap_intensity = mpl.colors.ListedColormap(colorlist)
    names_bymod, symbols_bymod = _model_markers(
        load_data_nc, MODEL_MARKERS[:8]
    )
    leg_nam_byint = [
        '>=' + str(x) + '"'
        for x in np.round(
            np.linspace(0, 1 * 10 * raw_thres, len(colorlist)), 2
        )
    ]
    cmap_raw = mpl.colormaps['cool'].copy()
    cmap_raw.set_under('white')

    def draw_objects(
        ax, model, keep, linecolor, linewidth, markersize, zorder, hr_inc, vmax
    ):
        prop = pair_prop[model]
        marker = _marker_for(model, load_data_nc, names_bymod, symbols_bymod)
        ids = np.unique(prop[0][prop[2] == hr_inc])
        for obj in keep(ids):
            snap = np.asarray(prop[0] == obj) & np.asarray(prop[2] == hr_inc)
            track = np.asarray(prop[0] == obj)
            if len(prop[5][snap]) > 2:
                ax.plot(
                    prop[5][snap],
                    prop[4][snap],
                    c=linecolor,
                    linewidth=linewidth,
                    zorder=zorder,
                    transform=PLATE,
                )
                ax.scatter(
                    prop[5][snap],
                    prop[4][snap],
                    c=prop[9][snap],
                    marker=marker,
                    vmin=0,
                    vmax=vmax,
                    s=markersize,
                    linewidth=0.25,
                    cmap=cmap_intensity,
                    zorder=zorder,
                    alpha=1,
                    transform=PLATE,
                )
            else:
                ax.plot(
                    prop[5][snap][0],
                    prop[4][snap][0],
                    c=linecolor,
                    linewidth=linewidth,
                    zorder=zorder,
                    transform=PLATE,
                )
                ax.scatter(
                    prop[5][snap][0],
                    prop[4][snap][0],
                    c=prop[9][snap][0],
                    marker=marker,
                    vmin=0,
                    vmax=vmax,
                    s=markersize,
                    linewidth=0.25,
                    cmap=cmap_intensity,
                    zorder=zorder,
                    alpha=0.8,
                    transform=PLATE,
                )
            first = slice(None) if len(prop[5][track]) > 2 else 0
            ax.plot(
                prop[5][track][first],
                prop[4][track][first],
                color=linecolor,
                linewidth=linewidth - 0.25,
                zorder=zorder,
                transform=PLATE,
            )

    def legends(ax):
        _intensity_legend(ax, colorlist, leg_nam_byint, fontsize=7)
        _marker_legend(ax, symbols_bymod, names_bymod, fontsize=7)

    def valid_box(ax, hr_inc):
        ax.text(
            0.5,
            0.97,
            f'Valid Time: {_valid(curdate, hr_inc)}'
            f' - {pre_acc:.2f} Hour Accumulation ',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='center',
            bbox=TEXT_BOX,
        )

    # 1) Object outlines, intensity and type for model and analysis.
    frames = []
    for hr_count, hr_inc in enumerate(hrs):
        fig, ax = _new_map()
        if not missing[hr_count]:
            valid_box(ax, hr_inc)
            for select, color in ((is_model, 'grey'), (~is_model, 'black')):
                if np.sum(clus[:, :, hr_count, select]) > 0:
                    with contextlib.suppress(Exception):
                        ax.contour(
                            lon,
                            lat,
                            np.squeeze(
                                np.round(clus[:, :, hr_count, select], 2)
                            ),
                            colors=color,
                            linewidths=1,
                            zorder=1,
                            transform=PLATE,
                            transform_first=True,
                        )
        for model in range(len(load_data_nc)):
            if not _has_data(pair_prop[model]):
                continue
            if is_model[model]:
                draw_objects(
                    ax,
                    model,
                    lambda ids: ids[ids > 0],
                    'k',
                    0.5,
                    25,
                    2,
                    hr_inc,
                    1.0,
                )
            else:
                draw_objects(
                    ax,
                    model,
                    lambda ids: ids[ids < 0],
                    'b',
                    1.5,
                    50,
                    3,
                    hr_inc,
                    1.0,
                )
        legends(ax)
        ax.set_title(
            'Ensemble Object Probabilities - Precip. >= '
            + str(raw_thres)
            + '" \n '
            f'{pre_acc:.2f} Hour Acc. Precip. - '
            f'Initialized {curdate:%Y%m%d%H}',
            fontsize=13,
        )
        frame = _fig_file(
            fig_path_s,
            f'{grib_path_des}_TOTENS_objshape_byhour_{stamp}{tag}'
            f'_f{hr_count + 1:02d}.png',
        )
        _finish_map(fig, ax, latlon_dims, frame, dpi=150)
        frames.append(frame)
    write_gif(
        frames,
        _fig_file(
            fig_path_s,
            f'{grib_path_des}_TOTENS_objshape_byhour_{stamp}{tag}.gif',
        ),
        delay_cs=100,
    )

    # 2-3) Raw forecast / observed precipitation with that side's objects.
    for label, raw, select, keep, title in (
        ('fcst', fcst, is_model, lambda ids: ids[ids > 0], 'Model Object'),
        ('obs', obs, ~is_model, lambda ids: ids[ids < 0], 'Observed Object'),
    ):
        loc = np.flatnonzero(select)[0]
        frames = []
        for hr_count, hr_inc in enumerate(hrs):
            values = np.linspace(0, 0.3, 11)
            fig, ax = _new_map()
            if not missing[hr_count]:
                valid_box(ax, hr_inc)
                if np.nansum(raw[:, :, hr_count]) > 0:
                    cs = ax.contourf(
                        lon,
                        lat,
                        np.round(raw[:, :, hr_count], 2),
                        levels=values,
                        extend='min',
                        cmap=cmap_raw,
                        antialiased=False,
                        vmin=raw_thres,
                        vmax=np.nanmax(values),
                        transform=PLATE,
                        transform_first=True,
                    )
                    fig.colorbar(cs, ax=ax)
            if _has_data(pair_prop[loc]):
                draw_objects(
                    ax, loc, keep, 'b', 1.5, 50, 3, hr_inc, np.nanmax(values)
                )
            legends(ax)
            ax.set_title(
                f'{title} and Raw Precipitation - Precip. >= '
                + str(raw_thres)
                + '" \n '
                f'{pre_acc:.2f} Hour Acc. Precip. - '
                f'Initialized {curdate:%Y%m%d%H%M}',
                fontsize=13,
            )
            frame = _fig_file(
                fig_path_s,
                f'{grib_path_des}_{label}_precipAobjs_byhour_{stamp}{tag}'
                f'_f{hr_count + 1:02d}.png',
            )
            _finish_map(fig, ax, latlon_dims, frame, dpi=150)
            frames.append(frame)
        write_gif(
            frames,
            _fig_file(
                fig_path_s,
                f'{grib_path_des}_{label}_precipAobjs_byhour_{stamp}{tag}.gif',
            ),
            delay_cs=50,
        )

    for model in range(len(load_data_nc)):
        pair_prop[model][2] = (pair_prop[model][2] * (1 / pre_acc)) - 1
