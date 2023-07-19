import datetime
import pathlib

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