"""Tests for wpcmtd.utils (expected values come from the original code)."""

import datetime
import json
import pathlib

import pytest

from wpcmtd import utils

DATA = pathlib.Path(__file__).parent / 'data'


@pytest.mark.parametrize(
    'case',
    json.loads((DATA / 'load_data_str_cases.json').read_text()),
    ids=lambda case: case[0],
)
def test_load_data_str_matches_original(case):
    *args, expected = case
    assert utils.load_data_str('/arch', *args) == [expected]


def test_load_data_str_unknown_model():
    with pytest.raises(ValueError, match='Unrecognized'):
        utils.load_data_str(
            '/arch', 'NOTAMODEL_lag00', '2026092712', 1, '01', '00'
        )


@pytest.mark.parametrize(
    'subhourly, name', [(False, 'hourly'), (True, 'subhourly')]
)
def test_gen_mtdconfig_matches_original(tmp_path, subhourly, name):
    func = utils.gen_mtdconfig_15m if subhourly else utils.gen_mtdconfig
    path = func(tmp_path / 'config' / 'c', 0.1, 6, 700, 0.7, 'A010000')
    expected = (DATA / f'MTDConfig_{name}.expected').read_text()
    assert path.read_text() == expected


def test_gen_mtdconfig_met_version(tmp_path):
    path = utils.gen_mtdconfig(
        tmp_path / 'c', 0.05, 2, 700, 0.7, 'A001500', met_version='V12.2'
    )
    text = path.read_text()
    assert 'version        = "V12.2";' in text
    assert 'conv_thresh       = >=1.27;' in text
    assert 'name  = "A001500";' in text


@pytest.mark.parametrize(
    'start, end, inc, expected',
    [
        (
            datetime.datetime(2026, 9, 27, 12),
            datetime.datetime(2026, 9, 27, 12),
            1,
            (
                datetime.datetime(2026, 9, 27, 12),
                datetime.datetime(2026, 9, 27, 13),
            ),
        ),
        (
            datetime.datetime(2026, 9, 27, 1),
            datetime.datetime(2026, 9, 27, 20),
            6,
            (
                datetime.datetime(2026, 9, 27, 6),
                datetime.datetime(2026, 9, 28, 0),
            ),
        ),
    ],
)
def test_adjust_date_range(start, end, inc, expected):
    assert utils.adjust_date_range(start, end, inc) == expected


def test_adjust_date_range_rejects_reversed_dates():
    with pytest.raises(ValueError):
        utils.adjust_date_range(
            datetime.datetime(2026, 9, 28), datetime.datetime(2026, 9, 25), 1
        )


def test_haversine():
    # One degree of latitude is ~111.2 km on a 6371 km sphere.
    assert utils.haversine2_distance((40, -100), (41, -100)) == pytest.approx(
        111.19, abs=0.01
    )
    assert utils.haversine2_distance((40, -100), (40, -100)) == 0
