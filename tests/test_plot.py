"""Tests for wpcmtd.plot helpers (and a small end-to-end figure)."""

import datetime
import multiprocessing
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from PIL import Image

from wpcmtd import plot


def legacy_ensemble_mean(simp_bin, data_success, members):
    """The original computation: float copy with NaN for missing data."""
    grid = (simp_bin > 0).astype(float)
    nan_loc = (data_success == 1).astype(float)
    nan_loc[nan_loc == 1] = np.nan
    grid = grid + nan_loc
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmax(np.nanmean(grid[:, :, :, members], axis=3), axis=2)


def test_member_means_match_legacy_nan_computation():
    rng = np.random.default_rng(0)
    simp_bin = rng.integers(0, 4, size=(20, 30, 12, 5)).astype(np.int8)
    data_success = rng.choice(
        [0.0, 1.0, np.nan], size=(12, 5), p=[0.7, 0.2, 0.1]
    )
    data_success[3, :] = 1  # an hour with no data at all
    members = np.array([True, True, False, True, True])

    expected = legacy_ensemble_mean(simp_bin, data_success, members)
    actual = plot._max_member_mean(simp_bin, members & ~(data_success == 1))
    np.testing.assert_array_equal(actual, expected)

    hour = 5
    sel = members & (data_success[hour] == 0)
    expected_hour = np.nanmean(
        (simp_bin[:, :, hour, sel] > 0).astype(float), axis=2
    )
    np.testing.assert_array_equal(
        plot._member_mean(simp_bin, sel, hour), expected_hour
    )


def test_members_found_ignores_beyond_range_hours():
    # Lagged members have NaN (not 1) for hours past their last forecast.
    data_success = np.array([[0, 0, 0], [0, 0, np.nan], [1, 0, np.nan]])
    assert plot._members_found(data_success) == 3
    assert plot._members_found(np.ones((3, 2))) == 0


def test_member_mean_without_members_is_nan():
    assert np.isnan(
        plot._member_mean(np.zeros((2, 2, 1, 2)), np.array([False, False]), 0)
    ).all()


def legacy_snow_color(obj):
    """The original if/elif chain."""
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    if 0.00 <= obj < thresholds[0]:
        return plot.SNOW_COLORS[0]
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= obj < thresholds[i + 1]:
            return plot.SNOW_COLORS[i + 1]
    if obj >= 0.50:
        return plot.SNOW_COLORS[-1]
    return None


@pytest.mark.parametrize(
    'value', list(np.round(np.arange(0, 0.8, 0.001), 3)) + [np.nan, -1]
)
def test_snow_outline_color(value):
    assert plot._snow_outline_color(value) == legacy_snow_color(value)


def test_split_hours():
    assert plot.split_hours(36, 5) == [
        list(range(0, 8)),
        list(range(8, 16)),
        list(range(16, 24)),
        list(range(24, 32)),
        list(range(32, 36)),
    ]
    assert plot.split_hours(3, 10) == [[0], [1], [2]]
    assert sum(plot.split_hours(60, 32), []) == list(range(60))


def test_time_index_round_trip():
    names = ['HRRRv415min_lag00.nc']
    prop = [np.array([[1.0, 2.0]] * 10)]
    plot._time_index_to_hours(prop, names)
    np.testing.assert_array_equal(prop[0][2], [0.5, 0.75])
    plot._hours_to_time_index(prop, names)
    np.testing.assert_array_equal(prop[0][2], [1.0, 2.0])


def test_write_gif(tmp_path):
    frames = []
    for i, color in enumerate(['red', 'green', 'blue']):
        path = tmp_path / f'f{i}.png'
        Image.new('RGB', (40 + i, 30), color).save(
            path
        )  # sizes may differ slightly
        frames.append(str(path))
    plot.write_gif(
        frames,
        tmp_path / 'a.gif',
        delay_cs=100,
        thumbnail_path=tmp_path / 'b.gif',
    )
    with Image.open(tmp_path / 'a.gif') as gif:
        assert gif.n_frames == 3
        assert gif.info['duration'] == 1000
        assert gif.info['loop'] == 0
    assert (tmp_path / 'b.gif').exists()


@pytest.mark.filterwarnings('ignore')
def test_plot_all_fcst_smoke(tmp_path):
    """Figures, GeoJSON and GIF for a tiny synthetic two-member ensemble."""
    lon, lat = np.meshgrid(np.linspace(-100, -90, 41), np.linspace(35, 42, 29))
    hrs = np.arange(1.0, 4.0)
    simp_bin = np.zeros(lat.shape + (3, 2), dtype=np.int8)
    simp_bin[10:15, 10:20, :, 0] = 1
    simp_bin[12:18, 15:25, 1:, 1] = 1
    data_success = np.zeros((3, 2))
    prop = np.full((10, 3), np.nan)
    prop[0] = 1
    prop[1] = 1
    prop[2] = [0, 1, 2]
    prop[3] = 50
    prop[4] = [38.0, 38.2, 38.4]
    prop[5] = [-96.0, -95.5, -95.0]
    prop[6] = 0
    prop[7:10] = [[0.1] * 3, [0.2] * 3, [0.4] * 3]
    simp_prop = [prop.copy(), prop.copy()]
    names = ['HRRRv4_TLE_lag01.nc', 'HRRRv4_TLE_lag00.nc']
    curdate = datetime.datetime(2026, 9, 27, 12)

    common = dict(
        fig_path_s=str(tmp_path),
        latlon_dims=[-100, 35, -90, 42],
        pre_acc=1.0,
        hrs=hrs,
        raw_thres=0.1,
        curdate=curdate,
        data_success=data_success,
        load_data_nc=names,
        lat=lat,
        lon=lon,
        simp_bin=simp_bin,
        simp_prop=simp_prop,
    )
    plot.mtd_plot_all_fcst(grib_path_des='TEST_ALL', sigma=1, **common)
    plot.mtd_plot_tle_fcst(grib_path_des='TEST_ALL', sigma=1, **common)

    names_out = {p.name for p in tmp_path.iterdir()}
    stem = 'TEST_ALL_TOTENS_objprob_byhour_2026092712_p1.00_t0.1'
    assert f'{stem}_f01.png' in names_out and f'{stem}_f03.png' in names_out
    assert f'{stem}_f02_tracks.geojson' in names_out
    assert f'{stem}.gif' in names_out
    assert 'TEST_ALL_TLE_obj_2026092712_p1.00_t0.1_group0.png' in names_out
    # The time index is restored after plotting.
    np.testing.assert_array_equal(simp_prop[0][2], [0, 1, 2])


def _synthetic_case():
    lon, lat = np.meshgrid(np.linspace(-100, -90, 41), np.linspace(35, 42, 29))
    prop = np.full((10, 6), np.nan)
    prop[0] = [1, 1, 1, -1, -1, -1]
    prop[1] = [1, 1, 1, -1, -1, -1]
    prop[2] = [0, 1, 2, 0, 1, 2]
    prop[3] = 50
    prop[4] = [38.0, 38.2, 38.4, 38.1, 38.3, 38.5]
    prop[5] = [-96.0, -95.5, -95.0, -96.1, -95.6, -95.1]
    prop[6] = 0
    prop[7:10] = [[0.1] * 6, [0.2] * 6, [0.3] * 6]
    return lat, lon, prop


@pytest.mark.filterwarnings('ignore')
def test_plot_snow_and_retro_smoke(tmp_path):
    lat, lon, prop = _synthetic_case()
    hrs = np.arange(1.0, 4.0)
    curdate = datetime.datetime(2026, 1, 10, 12)
    bins = np.zeros(lat.shape + (3, 2), dtype=np.int8)
    bins[10:15, 10:20, :, 0] = 1
    bins[12:18, 12:22, :, 1] = 1

    plot.mtd_plot_all_snow_fcst(
        'SNOW_ALL',
        str(tmp_path),
        [-100, 35, -90, 42],
        1.0,
        hrs,
        0.1,
        curdate,
        np.zeros((3, 2)),
        ['HRRRv4_TLE_lag06.nc', 'HRRRv4_TLE_lag00.nc'],
        lat,
        lon,
        bins,
        [prop[:, :3].copy(), prop[:, :3].copy()],
    )
    assert len(list(tmp_path.glob('SNOW_ALL_snowobj_byhour_*_h0?.png'))) == 3
    assert list(tmp_path.glob('SNOW_ALL_snowobj_byhour_p1.00_t0.1.gif'))

    raw = np.where(bins[:, :, :, 0] > 0, 0.5, 0.0)
    pair_prop = [np.full((10, 1), np.nan), prop.copy()]
    plot.mtd_plot_retro(
        'RETRO',
        str(tmp_path),
        [-100, 35, -90, 42],
        1.0,
        hrs,
        0.1,
        curdate,
        np.zeros((3, 2)),
        ['HRRRv4_TLE_lag00.nc', 'ST4_lag00.nc'],
        lat,
        lon,
        raw,
        raw,
        bins,
        pair_prop,
    )
    for kind in ('TOTENS_objshape', 'fcst_precipAobjs', 'obs_precipAobjs'):
        assert len(list(tmp_path.glob(f'RETRO_{kind}_byhour_*_f0?.png'))) == 3
        assert list(tmp_path.glob(f'RETRO_{kind}_byhour_*.gif'))


@pytest.mark.filterwarnings('ignore')
def test_plot_bias_lookup_smoke(tmp_path):
    lat, lon, prop = _synthetic_case()
    prop[0] = prop[1] = 1  # model objects only, one long track
    prop = np.concatenate([prop] * 2, axis=1)
    prop[2] = np.arange(12) % 6
    hrs = np.arange(1.0, 7.0)  # the temporal smoothing needs >= 5 hours
    curdate = datetime.datetime(2026, 9, 27, 12)
    bins = np.zeros(lat.shape + (6, 1), dtype=np.int8)
    bins[10:15, 10:20, :, 0] = 1
    rng = np.random.default_rng(1)
    retro = np.column_stack(
        [
            rng.normal(0, 50, (30, 2)),
            rng.normal(0, 0.1, (30, 5)),
            np.ones((30, 2)),
            np.full(30, 60.0),
        ]
    )
    pair_diff = [[retro.copy() for _ in range(prop.shape[1])]]
    args = (
        'BIAS_ALL',
        str(tmp_path),
        [-100, 35, -90, 42],
        1.0,
        hrs,
        0.1,
        curdate,
        np.zeros((6, 1)),
        ['HRRRv4_TLE_lag00.nc'],
        lat,
        lon,
        bins,
        [prop.copy()],
        pair_diff,
    )
    plot.mtd_plot_retro_disp_vect(*args, 1)
    assert len(list(tmp_path.glob('BIAS_ALL_retroplot3_*_f0?.png'))) == 6
    np.testing.assert_array_equal(pair_diff[0][0], retro)  # not modified
    plot.mtd_plot_retro_prob(*args)
    assert list(tmp_path.glob('BIAS_ALL_retroplot1_*.png'))


FORKSERVER_SCRIPT = """
import multiprocessing
import multiprocessing.util
import os

from wpcmtd import plot


def job(scale, i):
    return scale * i


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    plot.plot_regions_in_parallel(
        [(job, dict(i=i)) for i in range(4)], 2, dict(scale=2)
    )
    print(os.listdir(multiprocessing.util.get_temp_dir()))
"""


@pytest.mark.skipif(
    'forkserver' not in multiprocessing.get_all_start_methods(),
    reason='no fork server on this platform',
)
def test_parallel_plotting_leaves_no_forkserver_socket(tmp_path):
    # multiprocessing removes its temp folder at exit; a fork server still
    # bound to its socket there makes that fail on NFS ("Directory not
    # empty"), so the folder must be empty once the workers are done.
    script = tmp_path / 'run_pool.py'
    script.write_text(FORKSERVER_SCRIPT)
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env=dict(
            os.environ,
            TMPDIR=str(tmp_path),
            # import the same wpcmtd as this test
            PYTHONPATH=os.path.dirname(os.path.dirname(plot.__file__)),
        ),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '[]'
    assert 'Traceback' not in result.stderr
