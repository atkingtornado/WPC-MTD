"""Helper functions for WPC-MTD: file naming, MET config generation, grids."""

import datetime
import math
import pathlib
import string

import numpy as np
from netCDF4 import Dataset
from scipy import interpolate

# Default MET version written to generated config files.  MET refuses to run
# a config whose major.minor version does not match the executable.
MET_VERSION = 'V10.0'

MTD_CONFIG_TEMPLATE = string.Template("""\
model = "WRF";
desc = "NA";
obtype = "ANALYS";
regrid = {
    to_grid    = NONE;
    vld_thresh = 0.5;
    method     = NEAREST;
    width      = 1;
}

grid_res = 4;

fcst = {
   field = {
      name  = "$acc_str";
      level = "(*,*)";
   }

   conv_radius       = $conv_radius; // in grid squares
   conv_thresh       = >=$conv_thresh;
}

obs = fcst;
min_volume = $min_volume;
weight = {
   space_centroid_dist  = $space_centroid_weight;
   time_centroid_delta  = 2.0;
   speed_delta          = 0.0;
   direction_diff       = $direction_diff_weight;
   volume_ratio         = 1.0;
   axis_angle_diff      = 0.0;
   start_time_delta     = 3.0;
   end_time_delta       = 1.0;
}

interest_function = {
   space_centroid_dist = (
      (   0.0, 1.0 )
      $space_centroid_mid
      $space_centroid_zero
   );

   time_centroid_delta = (
      ( -5.0, 0.0 )
      ( -3.0, 0.5 )
      ( -1.0, 0.8 )
      (  0.0, 1.0 )
      (  1.0, 0.8 )
      (  3.0, 0.5 )
      (  5.0, 0.0 )
   );

   speed_delta = (
      ( -10.0, 0.0 )
      (  -5.0, 0.5 )
      (   0.0, 1.0 )
      (   5.0, 0.5 )
      (  10.0, 0.0 )
   );

   direction_diff = (
      (   0.0, 1.0 )
      (  90.0, 0.0 )
      ( 180.0, 0.0 )
   );

   volume_ratio = (
      (  0.0, 0.0 )
      (  0.5, 0.5 )
      (  1.0, 1.0 )
      (  1.5, 0.5 )
      (  2.0, 0.0 )
   );
   axis_angle_diff = (
      (  0.0, 1.0 )
      ( 30.0, 1.0 )
      ( 90.0, 0.0 )
   );

   start_time_delta = (
      (  0.0, 1.0 )
      (  3.0, 0.5 )
      (  5.0, 0.0 )
   );

   end_time_delta = (
      (  0.0, 1.0 )
      (  3.0, 0.5 )
      (  5.0, 0.0 )
   );

}

total_interest_thresh = $total_interest_thresh;

nc_output = {
   latlon       = true;
   raw          = true;
   object_id    = true;
   cluster_id   = true;
}

txt_output = {
   attributes_2d   = true;
   attributes_3d   = true;
}

output_prefix  = "";

version        = "$met_version";

""")

# Matching settings that differ between hourly and sub-hourly tracking.
_HOURLY_MATCHING = dict(
    space_centroid_weight='6.0',
    direction_diff_weight='2.0',
    space_centroid_mid='( 100.0, 0.5 )',
    space_centroid_zero='( 200.0, 0.0 )',
)
_SUBHOURLY_MATCHING = dict(
    space_centroid_weight='12.0',
    direction_diff_weight='0.0',
    space_centroid_mid='(  50.0, 0.5 )',
    space_centroid_zero='( 150.0, 0.0 )',
)


def haversine2_distance(origin, destination):
    """
    Great-circle distance (km) between two latitude/longitude pairs.

    Copied from https://gist.github.com/rochacbruno/2883505 by MJE. 20171012.

    Parameters
    ----------
    origin : (float, float)
        Origin (lat, lon) pair in degrees.
    destination : (float, float)
        Destination (lat, lon) pair in degrees.

    Returns
    -------
    float
        Distance between the two points in km.
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(
        math.radians(lat1)
    ) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def read_CONUS_mask(grib_path, latlon_dims, grid_delta):  # noqa: N802 (public name)
    """
    Read the ASCII CONUS mask (originally created in MATLAB) and its grid,
    and interpolate the mask to a new regular lat/lon grid.

    Parameters
    ----------
    grib_path : pathlib.Path
        Directory containing ``ERO/static``.
    latlon_dims : list
        Domain bounds [WLON, SLAT, ELON, NLAT].
    grid_delta : float
        Grid spacing (degrees lat/lon) of the new grid.

    Returns
    -------
    conus_mask, lat_new, lon_new : numpy.ndarray
        Mask (0 = not CONUS) and the coordinates of the new grid.
    """
    # Mask is only defined over -129.8 to -60.2 W and 20.2 to 50.0 N.
    if (
        latlon_dims[0] < -129.8
        or latlon_dims[2] > -60.2
        or latlon_dims[1] < 20.2
        or latlon_dims[3] > 50.0
    ):
        raise ValueError('User Specified lat/lon outside of boundary')

    static_dir = pathlib.Path(grib_path, 'ERO', 'static')
    lat = np.loadtxt(static_dir / 'CONUS_lat.txt')
    lon = np.loadtxt(static_dir / 'CONUS_lon.txt')
    conus_mask = np.loadtxt(static_dir / 'CONUSmask.txt')

    # New grid from the user specified boundaries and resolution.
    nlon = int(
        np.round((latlon_dims[2] - latlon_dims[0]) / grid_delta, 2) + 1.0
    )
    nlat = int(
        np.round((latlon_dims[3] - latlon_dims[1]) / grid_delta, 2) + 1.0
    )
    lon_new, lat_new = np.meshgrid(
        np.linspace(latlon_dims[0], latlon_dims[2], nlon),
        np.linspace(latlon_dims[3], latlon_dims[1], nlat),
    )

    conus_mask = interpolate.griddata(
        np.stack((lon.flatten(), lat.flatten()), axis=1),
        conus_mask.flatten(),
        (lon_new, lat_new),
        method='linear',
    )

    if grid_delta == 0.09:
        # Make-shift mask that filters out excessive coastal ST4 > FFG regions.
        ffg_file = pathlib.Path(
            grib_path,
            'ERO_verif',
            'static',
            'ST4gFFG_s2017072212_e2017082212_vhr12.nc',
        )
        with Dataset(ffg_file, 'r') as nc:
            temp = (np.array(nc.variables['ST4gFFG'][:]) <= 0.20) * 1
            lat = np.array(nc.variables['lat'][:])
            lon = np.array(nc.variables['lon'][:])
        lat = np.tile(lat, (len(lon), 1)).T
        lon = np.tile(lon, (len(lat), 1))
        temp = interpolate.griddata(
            np.stack((lon.flatten(), lat.flatten()), axis=1),
            temp.flatten(),
            (lon_new, lat_new),
        )
        conus_mask = conus_mask * temp

    return conus_mask, lat_new, lon_new


def adjust_date_range(start_date, end_date, init_inc):
    """
    Snap the start/end dates to the model initialization increment.

    Parameters
    ----------
    start_date, end_date : datetime.datetime
        Requested date range.
    init_inc : int
        Hours between model initializations.

    Returns
    -------
    (datetime.datetime, datetime.datetime)
        Adjusted start and end dates.
    """
    start_date = start_date + datetime.timedelta(
        hours=(init_inc - start_date.hour % init_inc)
        * (start_date.hour % init_inc > 0)
    )
    end_date = (
        end_date
        + datetime.timedelta(hours=init_inc)
        - datetime.timedelta(hours=end_date.hour % init_inc)
    )

    if (end_date - start_date).days < 0:
        raise ValueError("'end_date' is too close to 'start_date'")

    return start_date, end_date


def gen_mtdconfig(
    save_filename,
    raw_thres,
    conv_radius,
    min_volume,
    total_interest_thresh,
    acc_str,
    subhourly=False,
    met_version=MET_VERSION,
):
    """
    Write an MTD config file and return its path.

    Parameters
    ----------
    save_filename : pathlib.Path
        Output file; parent directories are created as needed.
    raw_thres : float
        Precipitation threshold (inches); converted to mm for MET.
    conv_radius : int
        Convolution radius (grid squares).
    min_volume : int
        Minimum object volume (grid squares x time steps).
    total_interest_thresh : float
        Total interest threshold for matching.
    acc_str : str
        Name of the accumulation field, e.g. ``A010000``.
    subhourly : bool
        Use the matching weights tuned for sub-hourly (15-minute) tracking.
    met_version : str
        MET version (``Vmajor.minor``) written to the config.
    """
    matching = _SUBHOURLY_MATCHING if subhourly else _HOURLY_MATCHING
    conf_data = MTD_CONFIG_TEMPLATE.substitute(
        acc_str=acc_str,
        conv_radius=conv_radius,
        conv_thresh=float(raw_thres) * 25.4,
        min_volume=int(min_volume),
        total_interest_thresh=total_interest_thresh,
        met_version=met_version,
        **matching,
    )

    save_filename = pathlib.Path(save_filename)
    save_filename.parent.mkdir(parents=True, exist_ok=True)
    save_filename.write_text(conf_data)
    return save_filename


def gen_mtdconfig_15m(
    save_filename,
    raw_thres,
    conv_radius,
    min_volume,
    total_interest_thresh,
    acc_str,
    met_version=MET_VERSION,
):
    """Write an MTD config tuned for sub-hourly data (see `gen_mtdconfig`)."""
    return gen_mtdconfig(
        save_filename,
        raw_thres,
        conv_radius,
        min_volume,
        total_interest_thresh,
        acc_str,
        subhourly=True,
        met_version=met_version,
    )


def _ens_member(load_data, strip_zero=True):
    """Member number following 'mem' in a model name ('mem05' -> '5')."""
    loc = load_data.find('mem')
    mem = load_data[loc + 3 : loc + 5]
    if strip_zero and int(mem) < 10:
        return mem[1:]
    return mem


def load_data_str(
    grib_path, load_data, init_yrmondayhr, acc_int, fcst_hr_str, fcst_min_str
):
    """
    Return the archive path (without ``.gz``) of one model/analysis file.

    Parameters
    ----------
    grib_path : pathlib.Path or str
        Root of the GRIB archive.
    load_data : str
        Model/analysis name, e.g. ``HRRRv4_TLE_lag06`` or ``ST4_lag00``.
    init_yrmondayhr : str
        Initialization time, ``YYYYMMDDHH``.
    acc_int : float
        Accumulation interval of the source data (hours).
    fcst_hr_str, fcst_min_str : str
        Two-digit forecast hour and minute.

    Returns
    -------
    list of str
        Single-element list containing the file path.
    """
    grib_path = str(grib_path)
    name = load_data[:-2]  # drop the two lag digits
    ymd = init_yrmondayhr[:-2]
    hh = init_yrmondayhr[-2:]
    init = datetime.datetime.strptime(ymd, '%Y%m%d')
    valid = init + datetime.timedelta(hours=int(fcst_hr_str) + int(hh))
    valid_min = init + datetime.timedelta(
        hours=int(fcst_hr_str) + float(fcst_min_str) / 60 + int(hh)
    )
    nssl = f'{grib_path}/NSSL/{ymd}/wrf4nssl'
    nssl_suffix = f'{ymd}{hh}.f{fcst_hr_str}.mod2'

    # Order matters: the first matching rule wins (e.g. 'MRMS' must be tested
    # after 'MRMS2min' and 'MRMS15min').
    if 'ST4' in name:
        # Analyses are stored by valid time rather than initialization time.
        path = (
            f'{grib_path}/ST4/{valid:%Y%m}/ST4.{valid:%Y%m%d%H}.'
            f'{int(acc_int):02d}h'
        )
    elif 'MRMS2min' in name:
        path = (
            f'{grib_path}/MRMS/{valid_min:%Y%m%d}/MRMS_PrecipRate_00.00_'
            f'{valid_min:%Y%m%d-%H%M}00.grib2'
        )
    elif 'MRMS15min' in name:
        path = (
            f'{grib_path}/MRMS_15min/{valid_min:%Y%m%d}/'
            f'MRMS_{valid_min:%Y%m%d-%H%M}00.grib2'
        )
    elif 'MRMS' in name:
        path = (
            f'{grib_path}/MRMS/{valid_min:%Y%m%d}/'
            f'MRMS_{valid_min:%Y%m%d-%H%M}00.grib2'
        )
    elif 'HIRESNAM' in name:
        # Also matches 'HIRESNAMP' (NAM nest parallel), as it always has.
        path = (
            f'{grib_path}/HIRESNAM/{ymd}/nam.t{hh}z.conusnest.hiresf'
            f'{fcst_hr_str}.tm00.grib2.mod'
        )
    elif 'HIRESWARW' in name and '2' not in name:
        path = (
            f'{grib_path}/HIRESW/{ymd}/hiresw.t{hh}z.arw_5km.f{fcst_hr_str}'
            '.conus.grib2.mod'
        )
    elif 'HIRESWARW2' in name:
        path = (
            f'{grib_path}/HIRESW/{ymd}/hiresw.t{hh}z.arw_5km.f{fcst_hr_str}'
            '.conusmem2.grib2.mod'
        )
    elif 'HIRESWFV3' in name:
        path = (
            f'{grib_path}/HIRESW/{ymd}/hiresw.t{hh}z.fv3_5km.f{fcst_hr_str}'
            '.conus.grib2.mod'
        )
    elif 'HRRRv2' in name:
        path = (
            f'{grib_path}/HRRRv2/{ymd}/hrrr.t{hh}z.wrfsfcf{fcst_hr_str}'
            '.grib2.mod'
        )
    elif 'HRRRv3' in name:
        path = f'{grib_path}/HRRRv3/{ymd}/hrrr_2d_{ymd}{hh}{fcst_hr_str}.grib2'
    elif 'HRRRv4' in name and '15min' not in name:
        path = f'{grib_path}/HRRRv4/{ymd}/hrrr_2d_{ymd}{hh}{fcst_hr_str}.grib2'
    elif 'NSSL-OP' in name:
        path = f'{nssl}_{nssl_suffix}'
    elif 'NSSL-ARW-CTL' in name:
        path = f'{nssl}_arw_ctl_{nssl_suffix}'
    elif 'NSSL-ARW-N1' in name:
        path = f'{nssl}_arw_n1_{nssl_suffix}'
    elif 'NSSL-ARW-P1' in name:
        path = f'{nssl}_arw_p1_{nssl_suffix}'
    elif 'NSSL-ARW-P2' in name:
        path = f'{nssl}_arw_p2_{nssl_suffix}'
    elif 'NSSL-NMB-CTL' in name:
        path = f'{nssl}_nmb_ctl_{nssl_suffix}'
    elif 'NSSL-NMB-N1' in name:
        path = f'{nssl}_nmb_n1_{nssl_suffix}'
    elif 'NSSL-NMB-P1' in name:
        path = f'{nssl}_nmb_p1_{nssl_suffix}'
    elif 'NSSL-NMB-P2' in load_data:
        path = f'{nssl}_nmb_p2_{nssl_suffix}'
    elif 'HRAM3E' in name:
        path = (
            f'{grib_path}/WWE2017/{ymd}/hram3e_{ymd}{hh}f0{fcst_hr_str}.grib2'
        )
    elif 'HRAM3G' in name:
        path = (
            f'{grib_path}/WWE2017/{ymd}/hram3g_{ymd}{hh}f0{fcst_hr_str}.grib2'
        )
    elif 'NEWSe5min' in name:
        init_time = datetime.datetime.strptime(init_yrmondayhr, '%Y%m%d%H')
        path = (
            f'{grib_path}/NEWSe/NEWSe_Min/{init_time:%Y%m%d}/'
            f'NEWSe_{init_time:%Y%m%d}_i{init_time:%H%M}'
            f'_m{_ens_member(load_data)}_f{valid_min:%H%M}.nc'
        )
    elif 'NEWSe60min' in name:
        path = (
            f'{grib_path}/NEWSe/NEWSe_Hour/{ymd}/NEWSe_{ymd}_i{hh}'
            f'_m{_ens_member(load_data)}_f{fcst_hr_str}.grb2'
        )
    elif 'HRRRe60min' in name:
        path = (
            f'{grib_path}/HRRRe/{ymd}/'
            f'mem{_ens_member(load_data, strip_zero=False)}'
            f'_hrrr_ens_{init_yrmondayhr}{fcst_hr_str}.grib2'
        )
    elif 'FV3LAM' in name:
        prefix = load_data[: load_data.find('_')]
        if prefix not in ('FV3LAM', 'FV3LAMdax', 'FV3LAMda'):
            raise ValueError(f'Unrecognized FV3LAM variant: {load_data}')
        path = (
            f'{grib_path}/{prefix}/{ymd}/{prefix.lower()}.t{hh}z'
            f'.conus.testbed.f{fcst_hr_str}.grib2.mod'
        )
    elif 'HRRRv415min' in name:
        path = (
            f'{grib_path}/HRRRv4_SubHourly/{ymd}/hrrr_2d_{init_yrmondayhr}'
            f'{fcst_hr_str}{fcst_min_str}.grib2'
        )
    else:
        raise ValueError(f'Unrecognized model/analysis name: {load_data}')

    return [path]
