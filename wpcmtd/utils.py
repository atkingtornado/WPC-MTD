import os
import datetime
import pathlib
import numpy as np
from scipy import interpolate
import math

def haversine2_distance(origin, destination):
    """
    Haversine formula (distance between latitude longitude pairs) in Python.
    Copied from https://gist.github.com/rochacbruno/2883505 by MJE. 20171012.

    Parameters
    ----------
    origin : [float]
        List containing origin lat/lon pair.
    destination : [float]
        List containing origin lat/lon pair.

    Returns
    -------
    d : float
      Distance between lat/lon pairs
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d


def read_CONUS_mask(grib_path,latlon_dims,grid_delta):
    """
    Reads an ASCII CONUS mask originally created in MATLAB, along with
    the latitude-longitude grid coordinates, and interpolates the data to a new grid.

    Parameters
    ----------
    grib_path : Pathlib.path
        directory where data is stored.
    latlon_dims : list
        latitude/longitude dimensions for plotting [WLON,SLAT,ELON,NLAT]
    grid_delta : int
      grid resolution increment for interpolation (degrees lat/lon)

    Returns
    -------
    CONUS_mask : numpy.array
      data containing the CONUSmask (0 = not CONUS) 
    lat_new : numpy.array
      latitude corresponding to CONUS data
    lon_new : numpy.array
      longitude corresponding to CONUS data
    """

    #Error if specified lat/lon dims are outside CONUS (-129.8 to -60.2 W and 20.2 to 50.0 N)
    if latlon_dims[0] < -129.8 or latlon_dims[2] > -60.2 or latlon_dims[1] < 20.2 or latlon_dims[3] > 50.0:
        raise ValueError('User Specified lat/lon outside of boundary')

    lat_count = 0  
    fobj = open(str(grib_path)+'/ERO/static/CONUS_lat.txt', 'r')
    for line in fobj:
        if lat_count == 0:
            lat = [float(i) for i in line.split()] 
        else:
            lat = np.vstack((lat,[float(i) for i in line.split()]))
        lat_count += 1
        
    lon_count = 0  
    fobj = open(str(grib_path)+'/ERO/static/CONUS_lon.txt', 'r')
    for line in fobj:
        if lon_count == 0:
            lon = [float(i) for i in line.split()]
        else:
            lon = np.vstack((lon,[float(i) for i in line.split()]))
        lon_count += 1
        
    mask_count = 0  
    fobj = open(str(grib_path)+'/ERO/static/CONUSmask.txt', 'r')
    for line in fobj:
        if mask_count == 0:
            CONUSmask = [float(i) for i in line.split()]
        else:
            CONUSmask = np.vstack((CONUSmask,[float(i) for i in line.split()]))
        mask_count += 1

    #Create new grid from user specified boundaries and resolution
    [lon_new,lat_new] = np.meshgrid(np.linspace(latlon_dims[0],latlon_dims[2],int(np.round((latlon_dims[2]-latlon_dims[0])/grid_delta,2)+1.0)), \
        np.linspace(latlon_dims[3],latlon_dims[1],int(np.round((latlon_dims[3]-latlon_dims[1])/grid_delta,2)+1.0)))

    #print('lon')
    #print(lon)
    #print('lon flatten and lat flatten')
    #print(np.stack((lon.flatten(),lat.flatten()),axis=1))
    #print('conus mask flatten')
    #print(CONUSmask.flatten())
    #print('lon_new')
    #print(lon_new)
    #Interpolate delx and dely from static grid to user specified grid
    CONUSmask = interpolate.griddata(np.stack((lon.flatten(),lat.flatten()),axis=1),CONUSmask.flatten(),(lon_new,lat_new),method='linear')

    if grid_delta == 0.09: #Load a make-shift mask that filters out excessive coastal ST4>FFG regions
        #f         = Dataset(grib_path+'/ERO_verif/static/ST4gFFG_s2017072212_e2017082212_vhr12.nc', "a", format="NETCDF4")
        f         = Dataset(str(grib_path)+'/ERO_verif/static/ST4gFFG_s2017072212_e2017082212_vhr12.nc', "r", format="NETCDF4")
        temp      = (np.array(f.variables['ST4gFFG'][:])<=0.20)*1
        lat       = np.array(f.variables['lat'][:])*1
        lon       = np.array(f.variables['lon'][:])*1
        lat       = np.tile(lat,(len(lon),1)).T
        lon       = np.tile(lon,(len(lat),1))
        f.close()
        
        temp      = interpolate.griddata(np.stack((lon.flatten(),lat.flatten()),axis=1),temp.flatten(),(lon_new,lat_new))
        
        CONUSmask = CONUSmask * temp
     
    return(CONUSmask,lat_new,lon_new)


def adjust_date_range(start_date, end_date, init_inc):
    """
    Adjust beginning and end dates to proper ranges given initialization increment 
    (init_inc, in hours; e.g. total hours between model init. cycles)

    Parameters
    ----------
    start_date : datetime.datetime object
        Start date of model data to be loaded.
    end_date : datetime.datetime object
        End date of model data to be loaded.
    
    Returns
    -------
    start_date : datetime.datetime object
        Adjusted start date.
    end_date : datetime.datetime object
        Adjusted end date.
    """
    start_date = start_date + datetime.timedelta(hours = (init_inc-start_date.hour%init_inc)*(start_date.hour%init_inc>0))
    end_date = end_date + datetime.timedelta(hours = init_inc) - datetime.timedelta(hours = end_date.hour%init_inc)

    #Throw an error if 'end_date' is before 'start_date'
    if (end_date - start_date).days < 0:
        raise ValueError("'end_date' is too close to 'start_date'")

    return (start_date, end_date)

def gen_mtdconfig(save_filename,raw_thres,conv_radius,min_volume,total_interest_thresh,acc_str):
    #Create the content for the CONFIG file
    conf_data = 'model = "WRF"; \n' \
      + 'desc = "NA";\n' \
      + 'obtype = "ANALYS"; \n' \
      + 'regrid = {\n' \
      + '    to_grid    = NONE;\n' \
      + '    vld_thresh = 0.5;\n' \
      + '    method     = NEAREST;\n' \
      + '    width      = 1;\n' \
      + '}\n\n' \
      + 'grid_res = 4;\n\n' \
      + 'fcst = {\n' \
      + '   field = {\n' \
      + '      name  = "'+acc_str+'";\n' \
      + '      level = "(*,*)";\n' \
      + '   }\n\n' \
      + '   conv_radius       = '+str(conv_radius)+'; // in grid squares\n' \
      + '   conv_thresh       = >='+str(float(raw_thres)*25.4)+';\n' \
      + '}\n\n' \
      + 'obs = fcst;\n' \
      + 'min_volume = '+str(int(min_volume))+';\n' \
      + 'weight = {\n' \
      + '   space_centroid_dist  = 6.0;\n' \
      + '   time_centroid_delta  = 2.0;\n' \
      + '   speed_delta          = 0.0;\n' \
      + '   direction_diff       = 2.0;\n' \
      + '   volume_ratio         = 1.0;\n' \
      + '   axis_angle_diff      = 0.0;\n' \
      + '   start_time_delta     = 3.0;\n' \
      + '   end_time_delta       = 1.0;\n' \
      + '}\n\n' \
      + 'interest_function = {\n' \
      + '   space_centroid_dist = (\n' \
      + '      (   0.0, 1.0 )\n' \
      + '      ( 100.0, 0.5 )\n' \
      + '      ( 200.0, 0.0 )\n' \
      + '   );\n\n' \
      + '   time_centroid_delta = (\n' \
      + '      ( -5.0, 0.0 )\n' \
      + '      ( -3.0, 0.5 )\n' \
      + '      ( -1.0, 0.8 )\n' \
      + '      (  0.0, 1.0 )\n' \
      + '      (  1.0, 0.8 )\n' \
      + '      (  3.0, 0.5 )\n' \
      + '      (  5.0, 0.0 )\n' \
      + '   );\n\n' \
      + '   speed_delta = (\n' \
      + '      ( -10.0, 0.0 )\n' \
      + '      (  -5.0, 0.5 )\n' \
      + '      (   0.0, 1.0 )\n' \
      + '      (   5.0, 0.5 )\n' \
      + '      (  10.0, 0.0 )\n' \
      + '   );\n\n' \
      + '   direction_diff = (\n' \
      + '      (   0.0, 1.0 )\n' \
      + '      (  90.0, 0.0 )\n' \
      + '      ( 180.0, 0.0 )\n' \
      + '   );\n\n' \
      + '   volume_ratio = (\n' \
      + '      (  0.0, 0.0 )\n' \
      + '      (  0.5, 0.5 )\n' \
      + '      (  1.0, 1.0 )\n' \
      + '      (  1.5, 0.5 )\n' \
      + '      (  2.0, 0.0 )\n' \
      + '   );\n' \
      + '   axis_angle_diff = (\n' \
      + '      (  0.0, 1.0 )\n' \
      + '      ( 30.0, 1.0 )\n' \
      + '      ( 90.0, 0.0 )\n' \
      + '   );\n\n' \
      + '   start_time_delta = (\n' \
      + '      (  0.0, 1.0 )\n' \
      + '      (  3.0, 0.5 )\n' \
      + '      (  5.0, 0.0 )\n' \
      + '   );\n\n' \
      + '   end_time_delta = (\n' \
      + '      (  0.0, 1.0 )\n' \
      + '      (  3.0, 0.5 )\n' \
      + '      (  5.0, 0.0 )\n' \
      + '   );\n\n' \
      + '}\n\n' \
      + 'total_interest_thresh = '+str(total_interest_thresh)+';\n\n' \
      + 'nc_output = {\n' \
      + '   latlon       = true;\n' \
      + '   raw          = true;\n' \
      + '   object_id    = true;\n' \
      + '   cluster_id   = true;\n' \
      + '}\n\n' \
      + 'txt_output = {\n' \
      + '   attributes_2d   = true;\n' \
      + '   attributes_3d   = true;\n' \
      + '}\n\n' \
      + 'output_prefix  = "";\n\n' \
      + 'version        = "V10.0";\n\n'
       

    pathlib.Path(save_filename.parent).mkdir(parents=True, exist_ok=True)

    f=open(str(save_filename), "w")
    f.write(conf_data)
    f.close()
   
    return(save_filename)

def gen_mtdconfig_15m(save_filename,raw_thres,conv_radius,min_volume,total_interest_thresh,acc_str):
    #Create the content for the CONFIG file
    conf_data = 'model = "WRF"; \n' \
      + 'desc = "NA";\n' \
      + 'obtype = "ANALYS"; \n' \
      + 'regrid = {\n' \
      + '    to_grid    = NONE;\n' \
      + '    vld_thresh = 0.5;\n' \
      + '    method     = NEAREST;\n' \
      + '    width      = 1;\n' \
      + '}\n\n' \
      + 'grid_res = 4;\n\n' \
      + 'fcst = {\n' \
      + '   field = {\n' \
      + '      name  = "'+acc_str+'";\n' \
      + '      level = "(*,*)";\n' \
      + '   }\n\n' \
      + '   conv_radius       = '+str(conv_radius)+'; // in grid squares\n' \
      + '   conv_thresh       = >='+str(float(raw_thres)*25.4)+';\n' \
      + '}\n\n' \
      + 'obs = fcst;\n' \
      + 'min_volume = '+str(int(min_volume))+';\n' \
      + 'weight = {\n' \
      + '   space_centroid_dist  = 12.0;\n' \
      + '   time_centroid_delta  = 2.0;\n' \
      + '   speed_delta          = 0.0;\n' \
      + '   direction_diff       = 0.0;\n' \
      + '   volume_ratio         = 1.0;\n' \
      + '   axis_angle_diff      = 0.0;\n' \
      + '   start_time_delta     = 3.0;\n' \
      + '   end_time_delta       = 1.0;\n' \
      + '}\n\n' \
      + 'interest_function = {\n' \
      + '   space_centroid_dist = (\n' \
      + '      (   0.0, 1.0 )\n' \
      + '      (  50.0, 0.5 )\n' \
      + '      ( 150.0, 0.0 )\n' \
      + '   );\n\n' \
      + '   time_centroid_delta = (\n' \
      + '      ( -5.0, 0.0 )\n' \
      + '      ( -3.0, 0.5 )\n' \
      + '      ( -1.0, 0.8 )\n' \
      + '      (  0.0, 1.0 )\n' \
      + '      (  1.0, 0.8 )\n' \
      + '      (  3.0, 0.5 )\n' \
      + '      (  5.0, 0.0 )\n' \
      + '   );\n\n' \
      + '   speed_delta = (\n' \
      + '      ( -10.0, 0.0 )\n' \
      + '      (  -5.0, 0.5 )\n' \
      + '      (   0.0, 1.0 )\n' \
      + '      (   5.0, 0.5 )\n' \
      + '      (  10.0, 0.0 )\n' \
      + '   );\n\n' \
      + '   direction_diff = (\n' \
      + '      (   0.0, 1.0 )\n' \
      + '      (  90.0, 0.0 )\n' \
      + '      ( 180.0, 0.0 )\n' \
      + '   );\n\n' \
      + '   volume_ratio = (\n' \
      + '      (  0.0, 0.0 )\n' \
      + '      (  0.5, 0.5 )\n' \
      + '      (  1.0, 1.0 )\n' \
      + '      (  1.5, 0.5 )\n' \
      + '      (  2.0, 0.0 )\n' \
      + '   );\n' \
      + '   axis_angle_diff = (\n' \
      + '      (  0.0, 1.0 )\n' \
      + '      ( 30.0, 1.0 )\n' \
      + '      ( 90.0, 0.0 )\n' \
      + '   );\n\n' \
      + '   start_time_delta = (\n' \
      + '      (  0.0, 1.0 )\n' \
      + '      (  3.0, 0.5 )\n' \
      + '      (  5.0, 0.0 )\n' \
      + '   );\n\n' \
      + '   end_time_delta = (\n' \
      + '      (  0.0, 1.0 )\n' \
      + '      (  3.0, 0.5 )\n' \
      + '      (  5.0, 0.0 )\n' \
      + '   );\n\n' \
      + '}\n\n' \
      + 'total_interest_thresh = '+str(total_interest_thresh)+';\n\n' \
      + 'nc_output = {\n' \
      + '   latlon       = true;\n' \
      + '   raw          = true;\n' \
      + '   object_id    = true;\n' \
      + '   cluster_id   = true;\n' \
      + '}\n\n' \
      + 'txt_output = {\n' \
      + '   attributes_2d   = true;\n' \
      + '   attributes_3d   = true;\n' \
      + '}\n\n' \
      + 'output_prefix  = "";\n\n' \
      + 'version        = "V10.0";\n\n'
    
    pathlib.Path(save_filename.parent).mkdir(parents=True, exist_ok=True)

    f=open(str(save_filename), "w")
    f.write(conf_data)
    f.close()

    return(save_filename)

def load_data_str(grib_path, load_data, init_yrmondayhr, acc_int, fcst_hr_str, fcst_min_str):

#####Comment out when not testing function#################
#load_data       = load_data[0]
#init_yrmondayhr = init_yrmondayhr[0]
#acc_int         = acc_int[0]
###########################################################

    #Default list of model strings
    data_default=['ST4'     ,'MRMS2min'    ,'MRMS15min'       ,'MRMS'              ,'HIRESNAM'           ,'HIRESNAMP'         ,'HIRESWARW'         , \
        'HIRESWARW2'        ,'HIRESWFV3'   ,'HRRRv2'          ,'HRRRv3'            ,'HRRRv4'             ,'NSSL-OP'           , \
        'NSSL-ARW-CTL'      ,'NSSL-ARW-N1' ,'NSSL-ARW-P1'     ,'NSSL-ARW-P2'       ,'NSSL-NMB-CTL'       ,'NSSL-NMB-N1'       , \
        'NSSL-NMB-P1'       ,'NSSL-NMB-P2' ,'HRAM3E'          ,'HRAM3G'            ,'NEWSe5min'          ,'NEWSe60min'        , \
        'HRRRe60min'        ,'FV3LAM'      ,'HRRRv415min']  #Array of total model strings
    grib_path = str(grib_path)
    #Create proper string for loading of model data
    if data_default[0] in load_data[:-2]: #Observation/analysis for ST4
        #ST4 is treated a bit differently, since the analysis time changes, hence the date and folder
        acc_int_str = '{:02d}'.format(int(acc_int))
        newtime= datetime.datetime(int(init_yrmondayhr[0:4]),int(init_yrmondayhr[4:6]),int(init_yrmondayhr[6:8]))+ \
            datetime.timedelta(hours=int(fcst_hr_str)+int(init_yrmondayhr[-2:]))
        data_str = [grib_path+'/ST4/'+newtime.strftime('%Y%m%d')+'/ST4.'+newtime.strftime('%Y%m%d')+ \
            newtime.strftime('%H')+'.'+acc_int_str+'h']
    elif data_default[1] in load_data[:-2]: #Observation/analysis for MRMS 2-min data
        #MRMS is treated a bit differently, since the analysis time changes, hence the date and folder
        newtime= datetime.datetime(int(init_yrmondayhr[0:4]),int(init_yrmondayhr[4:6]),int(init_yrmondayhr[6:8]))+ \
            datetime.timedelta(hours=int(fcst_hr_str)+float(fcst_min_str)/60+int(init_yrmondayhr[-2:]))
        data_str = [grib_path+'/MRMS/'+newtime.strftime('%Y%m%d')+'/MRMS_PrecipRate_00.00_'+newtime.strftime('%Y%m%d')+ \
            '-'+newtime.strftime('%H')+newtime.strftime('%M')+'00.grib2']
    elif data_default[2] in load_data[:-2]: #Observation/analysis for MRMS 15-min data
        #MRMS is treated a bit differently, since the analysis time changes, hence the date and folder
        newtime= datetime.datetime(int(init_yrmondayhr[0:4]),int(init_yrmondayhr[4:6]),int(init_yrmondayhr[6:8]))+ \
            datetime.timedelta(hours=int(fcst_hr_str)+float(fcst_min_str)/60+int(init_yrmondayhr[-2:]))
        data_str = [grib_path+'/MRMS_15min/'+newtime.strftime('%Y%m%d')+'/MRMS_'+newtime.strftime('%Y%m%d')+'-'+newtime.strftime('%H')+ \
            newtime.strftime('%M')+'00.grib2']
    elif data_default[3] in load_data[:-2]: #Observation/analysis for MRMS 1-hour data
        #MRMS is treated a bit differently, since the analysis time changes, hence the date and folder
        newtime= datetime.datetime(int(init_yrmondayhr[0:4]),int(init_yrmondayhr[4:6]),int(init_yrmondayhr[6:8]))+ \
            datetime.timedelta(hours=int(fcst_hr_str)+float(fcst_min_str)/60+int(init_yrmondayhr[-2:]))
        data_str = [grib_path+'/MRMS/'+newtime.strftime('%Y%m%d')+'/MRMS_'+newtime.strftime('%Y%m%d')+'-'+newtime.strftime('%H')+ \
            newtime.strftime('%M')+'00.grib2']
    elif data_default[4] in load_data[:-2] and not data_default[2] in load_data[:-2]:   #NAM NEST
        data_str = [grib_path+'/HIRESNAM/'+init_yrmondayhr[:-2]+'/nam.t'+init_yrmondayhr[-2:]+'z.conusnest.hiresf'+fcst_hr_str+'.tm00.grib2.mod']
    elif data_default[5] in load_data[:-2]: #NAM NEST PARALLEL
        data_str = [grib_path+'/HIRESNAMP/'+init_yrmondayhr[:-2]+'/nam_conusnest_'+init_yrmondayhr[:-2]+init_yrmondayhr[-2:]+'f0'+fcst_hr_str+'.grib2']
    elif data_default[6] in load_data[:-2] and '2' not in load_data[:-2]: #ARW CONUS
        data_str = [grib_path+'/HIRESW/'+init_yrmondayhr[:-2]+'/hiresw.t'+init_yrmondayhr[-2:]+'z.arw_5km.f'+fcst_hr_str+'.conus.grib2.mod']
    elif data_default[7] in load_data[:-2]: #ARW CONUS Member 2
        data_str = [grib_path+'/HIRESW/'+init_yrmondayhr[:-2]+'/hiresw.t'+init_yrmondayhr[-2:]+'z.arw_5km.f'+fcst_hr_str+'.conusmem2.grib2.mod']
    elif data_default[8] in load_data[:-2]: #FV3 CONUS
        data_str = [grib_path+'/HIRESW/'+init_yrmondayhr[:-2]+'/hiresw.t'+init_yrmondayhr[-2:]+'z.fv3_5km.f'+fcst_hr_str+'.conus.grib2.mod']
    elif data_default[9] in load_data[:-2]: #HRRRv2
        data_str = [grib_path+'/HRRRv2/'+init_yrmondayhr[:-2]+'/hrrr.t'+init_yrmondayhr[-2:]+'z.wrfsfcf'+fcst_hr_str+'.grib2.mod']
    elif data_default[10] in load_data[:-2]: #HRRRv3
        data_str = [grib_path+'/HRRRv3/'+init_yrmondayhr[:-2]+'/hrrr_2d_'+init_yrmondayhr[:-2]+init_yrmondayhr[-2:]+fcst_hr_str+'.grib2']
    elif data_default[11] in load_data[:-2] and not '15min' in load_data[:-2]: #HRRRv4
        data_str = [grib_path+'/HRRRv4/'+init_yrmondayhr[:-2]+'/hrrr_2d_'+init_yrmondayhr[:-2]+init_yrmondayhr[-2:]+fcst_hr_str+'.grib2']  
    elif data_default[12] in load_data[:-2]: #NSSL CONTROL
        data_str = [grib_path+'/NSSL/'+init_yrmondayhr[:-2]+'/wrf4nssl_'+init_yrmondayhr[:-2]+init_yrmondayhr[-2:]+'.f'+fcst_hr_str+'.mod2']
    elif data_default[13] in load_data[:-2]: #NSSL ARW CTL
        data_str = [grib_path+'/NSSL/'+init_yrmondayhr[:-2]+'/wrf4nssl_arw_ctl_'+init_yrmondayhr[:-2]+init_yrmondayhr[-2:]+'.f'+fcst_hr_str+'.mod2']
    elif data_default[14] in load_data[:-2]: #NSSL ARW N1
        data_str = [grib_path+'/NSSL/'+init_yrmondayhr[:-2]+'/wrf4nssl_arw_n1_'+init_yrmondayhr[:-2]+init_yrmondayhr[-2:]+'.f'+fcst_hr_str+'.mod2']
    elif data_default[15] in load_data[:-2]: #NSSL ARW P1
        data_str = [grib_path+'/NSSL/'+init_yrmondayhr[:-2]+'/wrf4nssl_arw_p1_'+init_yrmondayhr[:-2]+init_yrmondayhr[-2:]+'.f'+fcst_hr_str+'.mod2']
    elif data_default[16] in load_data[:-2]: #NSSL ARW P2
        data_str = [grib_path+'/NSSL/'+init_yrmondayhr[:-2]+'/wrf4nssl_arw_p2_'+init_yrmondayhr[:-2]+init_yrmondayhr[-2:]+'.f'+fcst_hr_str+'.mod2']
    elif data_default[17] in load_data[:-2]: #NSSL NMB CTL
        data_str = [grib_path+'/NSSL/'+init_yrmondayhr[:-2]+'/wrf4nssl_nmb_ctl_'+init_yrmondayhr[:-2]+init_yrmondayhr[-2:]+'.f'+fcst_hr_str+'.mod2']
    elif data_default[18] in load_data[:-2]: #NSSL NMB N1
        data_str = [grib_path+'/NSSL/'+init_yrmondayhr[:-2]+'/wrf4nssl_nmb_n1_'+init_yrmondayhr[:-2]+init_yrmondayhr[-2:]+'.f'+fcst_hr_str+'.mod2']
    elif data_default[19] in load_data[:-2]: #NSSL NMB P1
        data_str = [grib_path+'/NSSL/'+init_yrmondayhr[:-2]+'/wrf4nssl_nmb_p1_'+init_yrmondayhr[:-2]+init_yrmondayhr[-2:]+'.f'+fcst_hr_str+'.mod2']
    elif data_default[20] in load_data: #NSSL NMB P2
        data_str = [grib_path+'/NSSL/'+init_yrmondayhr[:-2]+'/wrf4nssl_nmb_p2_'+init_yrmondayhr[:-2]+init_yrmondayhr[-2:]+'.f'+fcst_hr_str+'.mod2']
    elif data_default[21] in load_data[:-2]: #HRAM3E
        data_str = [grib_path+'/WWE2017/'+init_yrmondayhr[:-2]+'/hram3e_'+init_yrmondayhr[:-2]+init_yrmondayhr[-2:]+'f0'+fcst_hr_str+'.grib2']
    elif data_default[22] in load_data[:-2]: #HRAM3G
        data_str = [grib_path+'/WWE2017/'+init_yrmondayhr[:-2]+'/hram3g_'+init_yrmondayhr[:-2]+init_yrmondayhr[-2:]+'f0'+fcst_hr_str+'.grib2']
    elif data_default[23] in load_data[:-2]: #NEWSe5min
        initime = datetime.datetime(int(init_yrmondayhr[0:4]),int(init_yrmondayhr[4:6]),int(init_yrmondayhr[6:8]),int(init_yrmondayhr[8::]))
        newtime = datetime.datetime(int(init_yrmondayhr[0:4]),int(init_yrmondayhr[4:6]),int(init_yrmondayhr[6:8]))+ \
            datetime.timedelta(hours=int(fcst_hr_str)+float(fcst_min_str)/60+int(init_yrmondayhr[-2:]))

        if int(load_data[load_data.find('mem')+3:load_data.find('mem')+5]) < 10:
            ens_mem = load_data[load_data.find('mem')+4:load_data.find('mem')+5]
        else:
            ens_mem = load_data[load_data.find('mem')+3:load_data.find('mem')+5]

        data_str = [grib_path+'/NEWSe/NEWSe_Min/'+initime.strftime('%Y%m%d')+'/NEWSe_'+initime.strftime('%Y%m%d')+ \
            '_i'+initime.strftime('%H')+initime.strftime('%M')+'_m'+ens_mem+'_f'+newtime.strftime('%H')+newtime.strftime('%M')+ \
            '.nc']
    elif data_default[24] in load_data[:-2]: #NEWSe60min
        if int(load_data[load_data.find('mem')+3:load_data.find('mem')+5]) < 10:
            ens_mem = load_data[load_data.find('mem')+4:load_data.find('mem')+5]
        else:
            ens_mem = load_data[load_data.find('mem')+3:load_data.find('mem')+5]
        data_str = [grib_path+'/NEWSe/NEWSe_Hour/'+init_yrmondayhr[:-2]+'/NEWSe_'+init_yrmondayhr[:-2]+'_i'+init_yrmondayhr[-2:]+'_m'+ens_mem+'_f'+fcst_hr_str+'.grb2']
    elif data_default[25] in load_data[:-2]: #HRRRe60min
        ens_mem = load_data[load_data.find('mem')+3:load_data.find('mem')+5]
        data_str = [grib_path+'/HRRRe/'+init_yrmondayhr[:-2]+'/mem'+ens_mem+'_hrrr_ens_'+init_yrmondayhr+fcst_hr_str+'.grib2']
    elif data_default[26] in load_data[:-2]: #FV3LAM
        if load_data[0:load_data.find('_')] == 'FV3LAM':
            data_str = [grib_path+'/FV3LAM/'+init_yrmondayhr[:-2]+'/fv3lam.t'+init_yrmondayhr[-2:]+'z.conus.testbed.f'+fcst_hr_str+'.grib2.mod']
        elif load_data[0:load_data.find('_')] == 'FV3LAMdax':
            data_str = [grib_path+'/FV3LAMdax/'+init_yrmondayhr[:-2]+'/fv3lamdax.t'+init_yrmondayhr[-2:]+'z.conus.testbed.f'+fcst_hr_str+'.grib2.mod']
        elif load_data[0:load_data.find('_')] == 'FV3LAMda':
            data_str = [grib_path+'/FV3LAMda/'+init_yrmondayhr[:-2]+'/fv3lamda.t'+init_yrmondayhr[-2:]+'z.conus.testbed.f'+fcst_hr_str+'.grib2.mod']
    elif data_default[27] in load_data[:-2]: #HRRRv415min
        data_str = [grib_path+'/HRRRv4_SubHourly/'+init_yrmondayhr[:-2]+'/hrrr_2d_'+init_yrmondayhr+fcst_hr_str+fcst_min_str+'.grib2']

    return data_str