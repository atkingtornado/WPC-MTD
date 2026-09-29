"""Tests for wpcmtd.mtd that do not need MET or model data."""

import datetime
import json
import pathlib

import numpy as np
import pytest
from netCDF4 import Dataset

from wpcmtd import mtd
from wpcmtd.mtd import WPCMTD

DATA = pathlib.Path(__file__).parent / 'data'


def make_mtd(tmp_path, **overrides):
    """WPCMTD with the thepro_hrrrtle.py settings and local paths."""
    kwargs = dict(
        met_path=tmp_path / 'met',
        temp_dir=tmp_path / 'tmp',
        fig_path=tmp_path / 'figs',
        grib_path=tmp_path / 'archive',
        config_path=tmp_path / 'config',
        track_path=tmp_path / 'tracks',
        latlon_dims=[-126, 23, -65, 50],
        beg_date=datetime.datetime(2026, 9, 27, 12),
        end_date=datetime.datetime(2026, 9, 27, 12),
        init_inc=1,
        mtd_mode='Operational',
        load_model=[
            'HRRRv4_TLE_lag12',
            'HRRRv4_TLE_lag06',
            'HRRRv4_TLE_lag00',
        ],
        load_qpe=['None'],
        snow_mask=False,
        pre_acc=1.0,
        season_sub=list(range(1, 13)),
        grib_path_des='MTD_QPF_HRRRTLE_EXT',
        thresh=0.1,
        plot_allhours=False,
        end_fcst_hrs=36.0,
        conv_radius=6,
        min_volume=700,
        ti_thresh=0.7,
        sigma=2,
        domain_sub=['_ALL', '_NW'],
        latlon_sub=[[-130, 20, -64, 52], [-130, 36, -100, 52]],
        transfer_to_prod=False,
        save_probs_to_file=False,
    )
    kwargs.update(overrides)
    return WPCMTD(**kwargs)


def write_static_grid(archive):
    """A small lat/lon grid covering CONUS, like STATIC.nc."""
    lon, lat = np.meshgrid(np.linspace(-135, -55, 81), np.linspace(15, 60, 46))
    (archive / 'static').mkdir(parents=True)
    with Dataset(archive / 'static' / 'STATIC.nc', 'w') as nc:
        nc.createDimension('lat', lat.shape[0])
        nc.createDimension('lon', lat.shape[1])
        nc.createVariable('lat', 'f4', ('lat', 'lon'))[:] = lat
        nc.createVariable('lon', 'f4', ('lat', 'lon'))[:] = lon
    return lat, lon


@pytest.mark.parametrize(
    'hours, expected',
    [
        (1, 'A010000'),
        (0.25, 'A001500'),
        (6, 'A060000'),
        (1 / 12, 'A000500'),
        (2 / 60, 'A000200'),
    ],
)
def test_accum_str(hours, expected):
    assert mtd._accum_str(hours) == expected


@pytest.mark.parametrize(
    'name, n_runs, acc',
    [
        ('HRRRv4_TLE_lag06', 25, 1),
        ('HRRRv415min_lag01', 25, 0.25),
        ('HIRESNAM_lag06', 5, 1),
        ('FV3LAM_lag00', 25, 1),
        ('HIRESWARW_lag12', 3, 1),
        ('ST4_lag00', 25, 1),
        ('NSSL-ARW-CTL_lag00', 2, 1),
        ('HRRRe60min_mem01_lag00', 25, 1),
    ],
)
def test_run_cycle(name, n_runs, acc):
    run_times, acc_int = mtd._run_cycle(
        name, 1.0, datetime.datetime(2026, 9, 27, 12)
    )
    assert len(run_times) == n_runs
    assert acc_int == pytest.approx(acc)


def test_run_cycle_st4_six_hourly():
    run_times, acc_int = mtd._run_cycle(
        'ST4_lag00', 6.0, datetime.datetime(2026, 9, 27)
    )
    assert (len(run_times), acc_int) == (5, 6)


def test_run_cycle_rejects_old_nam_and_unknown_models():
    with pytest.raises(ValueError, match='NAM'):
        mtd._run_cycle('HIRESNAM_lag00', 1.0, datetime.datetime(2016, 1, 1))
    with pytest.raises(ValueError, match='Unrecognized'):
        mtd._run_cycle('FOO_lag00', 1.0, datetime.datetime(2026, 1, 1))


def test_validation(tmp_path):
    with pytest.raises(ValueError, match='Operational'):
        make_mtd(tmp_path, load_qpe=['ST4_lag00'])
    with pytest.raises(ValueError, match='Retro'):
        make_mtd(tmp_path, mtd_mode='Retro', load_qpe=['ST4_lag00'])
    with pytest.raises(ValueError, match='NEWSe'):
        make_mtd(
            tmp_path, load_model=['NEWSe60min_mem01_lag00'], snow_mask=True
        )


def test_snow_tracks_go_to_spt(tmp_path):
    assert (
        make_mtd(tmp_path, snow_mask=True).track_path
        == tmp_path / 'tracks' / 'SPT'
    )


def test_hours_and_dates(tmp_path):
    obj = make_mtd(tmp_path, pre_acc=0.25, end_fcst_hrs=15.0)
    assert obj.hrs[0] == 0.25 and obj.hrs[-1] == 15.0 and len(obj.hrs) == 60
    assert list(obj.list_date) == [datetime.datetime(2026, 9, 27, 12)]

    retro = make_mtd(
        tmp_path,
        mtd_mode='Retro',
        load_model=['HRRRv4_TLE_lag00'],
        load_qpe=['ST4_lag00'],
        beg_date=datetime.datetime(2026, 9, 26, 0),
        end_date=datetime.datetime(2026, 9, 26, 5),
    )
    assert len(retro.list_date) == 6
    assert retro.load_data == ['HRRRv4_TLE_lag00', 'ST4_lag00']
    assert list(retro.ismodel) == [1.0, 0.0]


def test_setup_data(tmp_path):
    lat, lon = write_static_grid(tmp_path / 'archive')
    obj = make_mtd(tmp_path)
    obj.setup_data(datetime.datetime(2026, 9, 27, 12))
    assert obj.offset_fcsthr == [12, 6, 0]
    assert obj.init_yrmondayhr == ['2026092700', '2026092706', '2026092712']
    assert obj.acc_int == [1, 1, 1]
    assert obj.grib_path_temp.is_dir()
    assert obj.fig_path == tmp_path / 'figs' / '2026092712'
    # The analysis grid is cropped to latlon_dims +/- 3 degrees.
    assert obj.lat.min() >= 20 - 1 and obj.lat.max() <= 53 + 1
    assert obj.lon.min() >= -129 - 1 and obj.lon.max() <= -62 + 1

    # A second cycle does not nest the figure directory.
    obj.setup_data(datetime.datetime(2026, 9, 27, 13))
    assert obj.fig_path == tmp_path / 'figs' / '2026092713'


def test_nam_offset_when_cycle_is_between_runs(tmp_path):
    write_static_grid(tmp_path / 'archive')
    obj = make_mtd(tmp_path, load_model=['HIRESNAM_lag00', 'HIRESNAM_lag06'])
    obj.setup_data(datetime.datetime(2026, 9, 27, 13))
    assert obj.offset_fcsthr == [1, 7]
    assert obj.init_yrmondayhr == ['2026092712', '2026092706']


def test_mtd_output_prefix_uses_valid_time(tmp_path):
    """MTD names its output after the valid time of the first input."""
    obj = make_mtd(tmp_path)
    run = mtd._ModelRun(
        index=0,
        name='HIRESNAM_lag00',
        temp_dir=tmp_path,
        hrs_all=np.arange(1.0, 37.0),
        steps=[],
        grib_names=[],
        last_fcst_hr_str='37',
    )
    hour_success = np.zeros(36, dtype=bool)
    hour_success[2:] = True
    prefix = obj._mtd_output_prefix(
        datetime.datetime(2026, 9, 27, 13), run, hour_success
    )
    assert prefix == 'mtd_20260927_160000V'

    run.hrs_all = np.arange(0.25, 15.01, 0.25)
    prefix = obj._mtd_output_prefix(
        datetime.datetime(2026, 9, 27, 23), run, np.ones(60, dtype=bool)
    )
    assert prefix == 'mtd_20260927_231500V'


def test_read_mtd_2d(tmp_path):
    header = (
        'VERSION MODEL DESC FCST_LEAD OBJECT_ID OBJECT_CAT TIME_INDEX AREA '
        'CENTROID_X CENTROID_Y CENTROID_LAT CENTROID_LON AXIS_ANG '
        'INTENSITY_10\n'
    )
    rows = [
        ('F001', 'CF001', 0, 388, 42.76, -71.43, 3.81),
        ('F001', 'CF001', 1, 313, 42.80, -71.42, 4.01),
        ('F002', 'NA', 0, 50, 30.00, -90.00, -12.5),
        ('O001', 'CO001', 0, 400, 42.70, -71.50, 5.00),
        ('CF001', 'CF001', 0, 388, 42.76, -71.43, 3.81),
        ('CO001', 'CO001', 0, 400, 42.70, -71.50, 5.00),
        ('CF012', 'CF123', 3, 12345, 35.10, -95.20, 80.0),
    ]
    lines = [header] + [
        f'V12.2.2 WRF NA 010000 {i} {c} {t} {a} 1.0 2.0 {la} {lo} {ang} 0.5\n'
        for i, c, t, a, la, lo, ang in rows
    ]
    path = tmp_path / 'x_2d.txt'
    path.write_text(''.join(lines))

    simp, pair = mtd._read_mtd_2d(path)
    assert simp.shape == (10, 4) and pair.shape == (10, 3)
    np.testing.assert_array_equal(simp[0], [1, 1, 2, -1])
    np.testing.assert_array_equal(simp[1], [1, 1, np.nan, -1])
    np.testing.assert_array_equal(simp[2], [0, 1, 0, 0])
    np.testing.assert_array_equal(simp[3], [388, 313, 50, 400])
    np.testing.assert_array_equal(simp[6], [3.81, 4.01, -12.5, 5.0])
    np.testing.assert_array_equal(pair[0], [1, -1, 12])
    np.testing.assert_array_equal(pair[1], [1, -1, 123])
    assert np.isnan(simp[7:]).all()


def test_domain_slices():
    lon, lat = np.meshgrid(np.linspace(-135, -55, 81), np.linspace(15, 60, 46))
    rows, cols = mtd._domain_slices(lat, lon, [-126, 23, -65, 50])
    # The padded box [20, 53) x [-129, -62) plus one low-side row/column.
    assert (lat[rows, 0].min(), lat[rows, 0].max()) == (19, 51)
    assert (lon[0, cols].min(), lon[0, cols].max()) == (-130, -64)


def test_port_data_figs_commands_match_original(tmp_path, monkeypatch):
    expected = json.loads((DATA / 'port_data_figs_commands.json').read_text())
    for label, overrides in (
        ('qpf', {}),
        ('subhourly', dict(load_model=['HRRRv415min_lag00'])),
        ('snow', dict(snow_mask=True)),
    ):
        commands = []
        monkeypatch.setattr(
            mtd, '_run_shell', lambda cmd, out=commands: out.append(cmd) or 0
        )
        obj = make_mtd(tmp_path, grib_path_des='MTD_QPF_TEST', **overrides)
        obj.fig_path = '/figs/2026092712'
        obj.port_data_figs(datetime.datetime(2026, 9, 27, 12))
        assert commands == expected[label], label


def test_perf_timer():
    timer = mtd.PerfTimer()
    with timer.section('a'):
        pass
    timer.add('a', 1.0)
    assert timer.counts['a'] == 2 and timer.timers['a'] >= 1.0
    timer.reset()
    assert timer.timers == {}
