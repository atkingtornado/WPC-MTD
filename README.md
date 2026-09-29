# WPC-MTD

Object-based precipitation tracking for WPC products using the MET
[Mode Time Domain (MTD)](https://met.readthedocs.io/en/latest/Users_Guide/mode-td.html)
tool. `WPCMTD` converts and regrids model (and optionally Stage IV / MRMS)
precipitation with MET, tracks precipitation objects through time with
`mtd`, and turns the output into track files, ensemble object probability
figures, GeoJSON and animated GIFs.

The driver scripts that run the operational products (thepro, spt,
thepro-subhourly, anapro) live in the separate `WPC-MTD-SCRIPTS` repository.

## Installation

With Miniconda (or Anaconda/Miniforge):

```bash
conda env create -f environment.yml -n mtd_products
conda activate mtd_products
```

`environment.yml` pins the tested stack (Python 3.14, NumPy 2.5, matplotlib
3.11, Cartopy 0.26, ...) from **conda-forge only**: its `nodefaults` entry
keeps Miniconda's default Anaconda channel out, which lacks `geojson` and has
older geospatial packages. pip installs only `geojsoncontour` (not packaged
for conda) and this package, in editable mode.

* Linux requires glibc 2.28 or newer (RHEL 8+; check with `ldd --version`).
  On glibc 2.17 (RHEL/CentOS 7), change `matplotlib=3.11` to
  `matplotlib=3.10` in `environment.yml`.
* Use the libmamba solver (the default since conda 23.10). If an older conda
  is still on the classic solver, a solve can take 20+ minutes: run
  `conda install -n base conda-libmamba-solver` or pass `--solver libmamba`.

To install into an existing environment instead: `pip install -e .`

MET is not a Python package: install it separately and pass the directory
holding `pcp_combine`, `regrid_data_plane`, `gen_vx_mask` and `mtd` as
`met_path`.

## Usage

```python
import datetime
from pathlib import Path

from wpcmtd.mtd import WPCMTD

cycle = datetime.datetime(2026, 9, 27, 12)
mtd = WPCMTD(
    met_path=Path('/opt/MET/METPlus4_0/bin/'),
    temp_dir=Path('/scratch/mtd/data/'),
    fig_path=Path('/scratch/mtd/figs/'),
    grib_path=Path('/wpc/dev/archive/'),
    config_path=Path('/scratch/mtd/config/'),
    track_path=Path('/scratch/mtd/tracks/'),
    latlon_dims=[-126, 23, -65, 50],
    beg_date=cycle,
    end_date=cycle,
    init_inc=1,
    mtd_mode='Operational',
    load_model=['HRRRv4_TLE_lag12', 'HRRRv4_TLE_lag06', 'HRRRv4_TLE_lag00'],
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
mtd.run_mtd()
```

* `mtd_mode='Operational'` tracks the latest cycle of one or more models or
  ensemble members and makes figures; `'Retro'` pairs one model with QPE
  (`load_qpe=['ST4_lag00']`) over `beg_date`..`end_date` and saves gzipped NPZ
  track files used by the bias look-up (`WPCMTDBiasLookup`).
* Model names end in `_lagHH`; see `wpcmtd.utils.load_data_str` for the
  supported models and the archive layout under `grib_path`.
* `grib_path` must also contain `static/STATIC.nc` (lat/lon of the analysis
  grid) and `temp/ensmean_sample_NODELETE` (the grid everything is regridded
  to).

Optional keyword-only settings:

| Setting | Default | Purpose |
| --- | --- | --- |
| `max_workers` | CPU count, capped at 8 | Concurrent MET processes and plotting processes. Use 1 for fully serial runs. |
| `met_version` | `'V10.0'` | Version written to the generated MTD config; must match the MET release's major.minor version (e.g. `'V12.2'` for MET 12.2). |

Log messages (including a timing summary at the end of each run) go to
stdout.

## Development

```bash
pytest                # unit tests (no MET or data needed)
ruff check .          # lint (PEP 8 via pycodestyle rules, pyflakes, naming, ...)
ruff format .         # formatting (79 columns)
```

### End-to-end tests with real data

`devtools/` contains the tools used to check the package end to end against
public data, without access to the WPC archive:

* `fetch_test_data.py` downloads just the needed GRIB2 messages of HRRR
  (AWS open data) and Stage IV (NCEP NOMADS) into the archive layout the
  package expects. The static grid files can be created with MET, e.g.
  `regrid_data_plane <any HRRR file> G227 grid.nc -field 'name="APCP"; level="A1";'`,
  then copying `grid.nc` to `temp/ensmean_sample_NODELETE` and its `lat`/`lon`
  to `static/STATIC.nc`.
* `run_scenario.py` runs local copies of the production configurations
  (`hrrrtle`, `spt_hrrrtle`, `subhourly`, `hrrr`, `retro`) against any
  checkout of the package and records the MET commands, plot inputs and
  timing.
* `compare_runs.py` compares two such runs (plot inputs, MTD commands, track
  files, probability NetCDF, GeoJSON and figures pixel by pixel).

On an external or network drive, set `HDF5_USE_FILE_LOCKING=FALSE` if HDF5
reports locking errors.
