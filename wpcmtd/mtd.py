import dataclasses
import pathlib
import datetime
import numpy as np
from netCDF4 import Dataset
from .utils import adjust_date_range

@dataclasses.dataclass
class WPCMTD:
    """
    This class is responsible for configuring all the neccecary components 
    required for Model Evaluation Tools (MET) Mode Time Domain (MTD) verification.
    """

    met_path:       pathlib.PosixPath # Location of MET software
    temp_dir:       pathlib.PosixPath # Location where all temporary files are stored
    fig_path:       pathlib.PosixPath # Location for figure generation
    grib_path:      pathlib.PosixPath # Location of model GRIB files
    latlon_dims:    [float] # Latitude/longitude dimensions for plotting [WLON,SLAT,ELON,NLAT]
    beg_date:       datetime.datetime # Beginning time of retrospective runs
    end_date:       datetime.datetime # Ending time of retrospective runs
    init_inc:       int # Initialization increment (hours; e.g. total hours between model init. cycles)
    mtd_mode:       str # 'Operational', 'Retro', or 'Both'
    load_model:     [str] # list of models to load, ex.  ['HRRRv4_TLE_lag12'   ,'HRRRv4_TLE_lag06'  ,'HRRRv4_TLE_lag00']
    load_qpe:       [str] # 'ST4','MRMS',or 'NONE' (NOTE: NEWS-E and HRRRe DOES NOT WORK FOR ANYTHING OTHER THAN LAG = 00)
    snow_mask:      bool  # Keep only regions that are snow (True) or no masking (False)
    pre_acc:        float # Precipitation accumulation total
    season_sub:     [int] # Seasonally subset retro data to match oper data (array of months to keep)
    grib_path_des:  str   # An appended string label for the temporary directory name
    thresh:         float # Precip. threshold for defining objects (inches) (specifiy single value)
    end_fcst_hrs:   float # Last forecast hour to be considered

    grib_path_temp: pathlib.PosixPath = None
    lat:            float = None
    lon:            float = None
    offset_fcsthr:  [float] = dataclasses.field(default_factory=list) 
    acc_int:        [int] = dataclasses.field(default_factory=list) 


    # Perform post-initalization checks and property updates based on initial values
    def __post_init__(self):
        # Adjust beginning and end dates to proper ranges given initialization increment 
        (beg_date, end_date) = adjust_date_range(self.beg_date,self.end_date, self.init_inc)
        self.beg_date = beg_date
        self.end_date = end_date

        # Perform some sanity checks

        # If mtd_mode='Operational' then there should be no verification and only one cycle analyzed
        if self.mtd_mode == 'Operational':
            if self.load_qpe[0] != 'None':
                raise ValueError('If mtd_mode = \'Operational\' then load_qpe = \'None\'')
        elif self.mtd_mode == 'Retro':
            self.fig_path = self.fig_path+'RETRO/'
            if len(self.load_model) > 1:
                raise ValueError('If mtd_mode = \'Retro\' then only one model specification at a time')
        elif self.mtd_mode == 'Both':
            strip = [model[0:model.find('TLE')] for model in load_model]
            for model in strip:
                if model != self.load_model[0][0:self.load_model[0].find('TLE')]:
                    raise ValueError("'load_model' must be a single model or a TLE.")

        #NEWSe not contain snow information
        if 'NEWSe' in ''.join(self.load_model) and self.snow_mask == True:
            raise ValueError('NEWSe does not contain the appropriate snow mask information')

        #If using NEWSe/MRMS, only load 1 hour increment or less
        if ('NEWSe' in ''.join(self.load_model) or 'MRMS' in self.load_qpe) and self.pre_acc > 1:
            raise ValueError('NEWSe or MRMS data can only Have a precipitation accumulation interval of 1 hour or less')


    @property
    def load_data(self):
        #Determine if models are being compared to QPE, or if the models are solo
        if self.load_qpe[0].lower() != 'none':
            return np.append(self.load_model,self.load_qpe[0])
        else:
            return self.load_model
    
    @property
    def ismodel(self):
        #Logical matrix = 1 if model and = 0 if obs
        ismodel = []
        for mod in self.load_data:
            if 'ST4' in mod or 'MRMS' in mod:
                ismodel = np.append(ismodel,0)
            else:
                ismodel = np.append(ismodel,1)
        return ismodel

    @property
    def list_date(self):
        #Create datetime list of all model initializations to be loaded
        if self.mtd_mode == 'Operational':
            return [self.beg_date]
        else:
            delta = self.end_date - self.beg_date
            total_incs = int(((delta.days * 24) + (delta.seconds / 3600) - 1) / self.init_inc)
            list_date = []
            for i in range(total_incs+1):
                if (self.beg_date + datetime.timedelta(hours = i * self.init_inc)).month in self.season_sub:
                    list_date = np.append(list_date,self.beg_date + datetime.timedelta(hours = i * self.init_inc))
            return list_date
    @property
    def hrs(self):
        #Set the proper forecast hours load depending on the initialization
        hrs = np.arange(0.0,self.end_fcst_hrs+0.01,self.pre_acc)
        #Remove partial accumulation periods
        hrs = hrs[hrs > 0]
        return hrs
     

    def setup_data(self, curr_date):
        #Create an array of partial strings to match the load data array specified by the user
        model_default=['ST4','MRMS_','MRMS15min','MRMS2min','NAM','HIRESW','HRRR','NSSL-OP','NSSL-ARW','NSSL-NMB','HRAM','NEWSe5min','NEWSe60min', \
            'HRRRe60min','FV3LAM','HRRRv415min']

        #Convert numbers to strings of proper length
        yrmondayhr=curr_date.strftime('%Y%m%d%H')
        
        #Use list comprehension to cat on '.nc' to model list
        load_data_nc = [x + '.nc' for x in self.load_data]

        #Given the initalization hour, choose the most recent run for each model, archive initialization times
        init_yrmondayhr=[]
        for model in range(len(self.load_data)): #Through the models
            if model_default[0] in self.load_data[model]:          #ST4 (Depends on specified acc. interval)
                if (self.pre_acc == 6 or self.pre_acc == 12 or self.pre_acc == 18 or self.pre_acc == 24): # 6 hour acc. (24 hr acc. takes to long operationally)
                    run_times=np.linspace(0,24,5) #Time in UTC data is available
                    self.acc_int=np.append(self.acc_int,6)  #Hourly interval of data
                else:                                         #1 hour acc.
                    run_times=np.linspace(0,24,25)
                    self.acc_int=np.append(self.acc_int,1)
            elif model_default[1] in self.load_data[model]:       #MRMS available in 1 hour output
                if (self.pre_acc == 6 or self.pre_acc == 12 or self.pre_acc == 18 or self.pre_acc == 24): # 6 hour acc. (24 hr acc. takes to long operationally)
                    run_times=np.linspace(0,24,5) #Time in UTC data is available
                    self.acc_int=np.append(self.acc_int,6)  #Hourly interval of data
                else:                                         #1 hour acc.
                    run_times=np.linspace(0,24,25)
                    self.acc_int=np.append(self.acc_int,1)
            elif model_default[2]  in self.load_data[model]:       #MRMS available in 15 min output
                run_times=np.linspace(0,24,4*24+1)
                self.acc_int=np.append(self.acc_int,15.0/60.0)
            elif model_default[3]  in self.load_data[model]:       #MRMS available in 2 min output
                run_times=np.linspace(0,24,30*24+1)
                self.acc_int=np.append(self.acc_int,2.0/60.0)
            elif model_default[4]  in self.load_data[model]:       #NAMNEST AND NAMNESTP
                run_times=np.linspace(0,24,5)
                #Accumulation interval depends on date for NAM and also user specificed precipitation interval
                if pygrib.datetime_to_julian(curdate) >= pygrib.datetime_to_julian(datetime.datetime(2017,3,21,6,0)):
                    self.acc_int=np.append(self.acc_int,1)
                else:
                    #If desired interval (pre_acc) is one but we are using earlier NAM data, throw error
                    raise ValueError('Specified Interval of 1 Hour Impossible with NAM before 2017032106')
            elif model_default[5]  in self.load_data[model]:       #HIRES ARW/FV3
                run_times=np.linspace(0,24,3)
                self.acc_int=np.append(self.acc_int,1)
            elif model_default[6]  in self.load_data[model] and \
                '15min' not in self.load_data[model]:              #HRRR or HRRRp
                run_times=np.linspace(0,24,25)
                self.acc_int=np.append(self.acc_int,1)
            elif model_default[7]  in self.load_data[model]:       #NSSL Operational
                run_times=np.linspace(0,24,3)
                self.acc_int=np.append(self.acc_int,1)
            elif model_default[8]  in self.load_data[model] \
                or model_default[9]  in self.load_data[model]:     #NSSL Ensemble
                run_times=np.linspace(0,24,2)
                self.acc_int=np.append(self.acc_int,1)
            elif model_default[11]  in self.load_data[model]:      #NEWSe5min (NEWSe at 5 minute output)
                run_times=np.linspace(0,24,25)
                self.acc_int=np.append(self.acc_int,1.0/12.0)
            elif model_default[12]  in self.load_data[model]:      #NEWSe60min (NEWSe at 60 minute output)
                run_times=np.linspace(0,24,25)
                self.acc_int=np.append(self.acc_int,1)
            elif model_default[13]  in self.load_data[model]:      #HRRRe60min (HRRRe at 60 minute output)
                run_times=np.linspace(0,24,3)
                self.acc_int=np.append(self.acc_int,1)
            elif model_default[14]  in self.load_data[model]:      #FV3LAM (FV3LAM at 60 minute output)
                run_times=np.linspace(0,24,25)
                self.acc_int=np.append(self.acc_int,1)
            elif model_default[15]  in self.load_data[model]:      #HRRR15min (HRRR at 15 minute output)
                run_times=np.linspace(0,24,25)
                self.acc_int=np.append(self.acc_int,1.0/4.0)

            #Calculate lag from user specified string
            lag = float(self.load_data[model][self.load_data[model].index("lag")+3::])
            
            #Check to see if lag makes sense given frequency of initialization, and calculate strings
            if lag%(24.0 / (float(len(run_times))-1)) == 0: 
                #Determine the most recent model initialization offset
                init_offset = curr_date.hour - run_times[np.argmax(run_times>curr_date.hour)-1]
                #Combine the lag and offset to determine total offset
                self.offset_fcsthr=np.append(self.offset_fcsthr,init_offset+lag)
                #Determine yrmondayhr string from above information
                curdate_temp=curr_date-datetime.timedelta(hours=int(self.offset_fcsthr[model]))
                init_yrmondayhr=np.append(init_yrmondayhr,curdate_temp.strftime('%Y%m%d%H'))
            else: #If lag is misspecified, compute with no lag
                lag = 0
                init_offset = curr_date.hour - run_times[np.argmax(run_times>curr_date.hour)-1]
                self.offset_fcsthr=np.append(self.offset_fcsthr,init_offset+lag)
                curdate_temp=curr_date-datetime.timedelta(hours=int(self.offset_fcsthr[model]))
                init_yrmondayhr=np.append(init_yrmondayhr,curdate_temp.strftime('%Y%m%d%H'))
        # Create temporary directory
        self.grib_path_temp=pathlib.Path(self.temp_dir,yrmondayhr+'_p_'+str(self.pre_acc)+'_t'+str(self.thresh)+'_'+self.grib_path_des)
        self.grib_path_temp.mkdir(parents=True, exist_ok=True)

        # Remove old files from temp directory
        for path in self.grib_path_temp.glob("**/*"):
            if path.is_file():
                path.unlink()

        self.fig_path = pathlib.Path(self.fig_path, yrmondayhr)
        self.fig_path.mkdir(parents=True, exist_ok=True)

        #Load static data file and initialize needed variables for later
        f = Dataset(pathlib.Path(self.grib_path,'static','STATIC.nc'), "a", format="NETCDF4")
        lat=f.variables['lat'][:]
        lon=f.variables['lon'][:]
        f.close()
        #Remove gridded data not in the lat/lon grid specifications
        data_subset=(lat >= self.latlon_dims[1]-3) & (lat < self.latlon_dims[3]+3) & (lon >= self.latlon_dims[0]-3) & (lon < self.latlon_dims[2]+3)*1
        x_elements_beg=np.argmax(np.diff(data_subset,axis=1)==1, axis=1)
        x_elements_end=np.argmax(np.diff(data_subset,axis=1)==-1, axis=1)
        y_elements_beg=np.argmax(np.diff(data_subset,axis=0)==1, axis=0)
        y_elements_end=np.argmax(np.diff(data_subset,axis=0)==-1, axis=0)
        x_elements_beg=x_elements_beg[x_elements_beg > 0]
        x_elements_end=x_elements_end[x_elements_end > 0]
        y_elements_beg=y_elements_beg[y_elements_beg > 0]
        y_elements_end=y_elements_end[y_elements_end > 0]
        x_beg=np.min(x_elements_beg)
        x_end=np.max(x_elements_end)
        y_beg=np.min(y_elements_beg)
        y_end=np.max(y_elements_end)
        
        self.lat=lat[y_beg:y_end,x_beg:x_end]
        self.lon=lon[y_beg:y_end,x_beg:x_end]
    
    def run_mtd(self):
        for curr_date in self.list_date:
            self.setup_data(curr_date)

            # Initialize variables determining successfully downloaded models
            MTDfile_old        = ['dummyname'  for j in range(0,len(self.load_data))]
            MTDfile_new        = ['dummyname'  for j in range(0,len(self.load_data))]
            simp_bin           = np.zeros((self.lat.shape[0], self.lat.shape[1], len(self.hrs), len(self.load_data)),dtype=np.int8)
            clus_bin           = np.zeros((self.lat.shape[0], self.lat.shape[1], len(self.hrs), len(self.load_data)),dtype=np.int8)
            simp_prop          = [[] for i in range(0,len(self.load_data))]
            pair_prop          = [[] for i in range(0,len(self.load_data))]
            data_success       = np.ones([len(self.hrs), len(self.load_data)])
            mem                = []
            
            data_name_nc_all = []

            for model in range(len(self.load_data)): #Through the 1 model and observation
            
                #Create APCP string (e.g. pre_acc = 1, APCP_str_beg = 'A010000'; pre_acc = 0.25, APCP_str_beg = 'A001500')
                time_inc_beg = datetime.timedelta(hours=self.acc_int[model])
                time_inc_end = datetime.timedelta(hours=self.pre_acc)
                APCP_str_beg = 'A'+'{:02d}'.format(int(time_inc_beg.seconds/(3600)))+'{:02d}'.format(int(((time_inc_beg.seconds \
                     % 3600)/60 * (time_inc_beg.seconds >= 3600)) + ((time_inc_beg.seconds < 3600)*(time_inc_beg.seconds / 60))))+'00'
                APCP_str_end = 'A'+'{:02d}'.format(int(time_inc_end.seconds/(3600)))+'{:02d}'.format(int(((time_inc_end.seconds \
                     % 3600)/60 * (time_inc_end.seconds >= 3600)) + ((time_inc_end.seconds < 3600)*(time_inc_end.seconds / 60))))+'00'

                # #Given the variables, create proper config files
                # if 'NEWSe' in ''.join(self.load_model) or '15min' in ''.join(self.load_model):
                #     config_name = METConFigGenV100.set_MTDConfig_15min(CON_PATH,'MTDConfig_USE_SAMP',thres,conv_radius,min_volume,end_fcst_hrs,ti_thresh,APCP_str_end)
                # else:
                #     config_name = METConFigGenV100.set_MTDConfig(CON_PATH,'MTDConfig_USE_SAMP',thres,conv_radius,min_volume,end_fcst_hrs,ti_thresh,APCP_str_end)

                # #Isolate member number if there are multiple members
                # if 'mem' in load_data_nc[model]:
                #     mem = np.append(mem,'_m'+load_data_nc[model][load_data_nc[model].find('mem')+3:load_data_nc[model].find('mem')+5])
                # else:
                #     mem = np.append(mem,'')
                    
                # #Determine lag from string specification
                # mod_lag = int(load_data[model][load_data[model].find('lag')+3:load_data[model].find('lag')+5])

                # #NEWSe 60min files only store accumulated precip. Must load previous hour in instances of lag
                # if 'NEWSe60min' in load_data[model] and mod_lag > 0:
                #     hrs_all = np.arange(hrs[0]-1,hrs[-1]+pre_acc,pre_acc)
                #     data_success       = np.concatenate((data_success,np.ones([1, len(load_data)])),axis = 0)
                # else: 
                #     hrs_all = np.arange(hrs[0],hrs[-1]+pre_acc,pre_acc)

                # while 1: #In operational mode, wait for a while to make sure all data comes in

                #     #When snow_mask = True, there is no ST4 analysis mask, so ues the model CSNOW (will utitlize MRMS later)
                #     try:
                #         data_name_grib_prev  = data_name_grib
                #     except NameError:
                #         data_name_grib_prev  = []
                    
                #     fcst_hr_count = 0
                #     data_name_grib        = []
                #     data_name_nc          = []
                #     data_name_nc_prev   = []
                #     for fcst_hr in hrs_all: #Through the forecast hours
        
                #         #Create proper string of last hour loaded to read in MTD file
                #         last_fcst_hr_str = '{:02d}'.format(int(fcst_hr+offset_fcsthr[model]))
        
                #         #Sum through accumulated precipitation interval
                #         sum_hr_count = 0
                #         data_name_temp = []
                #         data_name_temp_part = []
        
                #         for sum_hr in np.arange(pre_acc-acc_int[model],-acc_int[model],-acc_int[model]):
                        
                #             #range(int(pre_acc) - int(acc_int[model]),-1,int(-acc_int[model])):
                            
                #             #Determine the forecast hour and min to load given current forecast hour, summing location and offset
                #             fcst_hr_load = float(fcst_hr) + offset_fcsthr[model] - float(round(sum_hr,2))
                #             fcst_hr_str  = '{:02d}'.format(int(fcst_hr_load))
                #             fcst_min_str = '{:02d}'.format(int(round((fcst_hr_load-int(fcst_hr_load))*60)))
        
                #             #Determine the end date for the summation of precipitation accumulation
                #             curdate_ahead = curdate+datetime.timedelta(hours=fcst_hr_load - offset_fcsthr[model])
        
                #             #Use function to load proper data filename string
                #             data_name_temp = np.append(data_name_temp,METLoadEnsemble.loadDataStr(GRIB_PATH, load_data[model], init_yrmondayhr[model], acc_int[model], fcst_hr_str, fcst_min_str))
        
                #             #Find location of last slash to isolate file name from absolute path
                #             for i in range(0,len(data_name_temp[sum_hr_count])):
                #                 if '/' in data_name_temp[sum_hr_count][i]:
                #                     str_search = i + 1
                #             data_name_temp_part = np.append(data_name_temp_part, data_name_temp[sum_hr_count][str_search::])
        
                #             #Copy the needed files to a temp directory and note successes
                #             output=os.system("cp "+data_name_temp[sum_hr_count]+".gz "+GRIB_PATH_TEMP)
                #             sum_hr_count += 1
                #         #END loop through accumulated precipitation
                        
                #         #Create the string for the time increment ahead
                #         yrmonday_ahead = str(curdate_ahead.year)+'{:02d}'.format(int(curdate_ahead.month))+'{:02d}'.format(int(curdate_ahead.day))
                #         hrmin_ahead = '{:02d}'.format(int(curdate_ahead.hour))+'{:02d}'.format(int(curdate_ahead.minute))
        
                #         #Gunzip the file 
                #         output = os.system("gunzip "+GRIB_PATH_TEMP+"/*.gz")
                        
                #         #Create archive of last hour grib file name within the summed increment
                #         data_name_grib     = np.append(data_name_grib, data_name_temp_part)

                #         ##########?) USE MET PCP_COMBINE TO CONVERT DATA TO NETCDF AND SUM PRECIPITATION WHEN APPLICABLE############
                #         METLoadEnsemble.pcp_combine(MET_PATH,GRIB_PATH_TEMP,pre_acc,acc_int[model],model,load_data[model],load_data_nc[model],\
                #             data_name_grib,load_qpe[0],init_yrmondayhr[model],yrmonday_ahead,hrmin_ahead,fcst_hr,APCP_str_beg,\
                #             APCP_str_end,fcst_hr_count)

                #         #For NEWSe/WoF the domain changes so determine domain here
                #         if mtd_mode == 'Operational':
                #             if 'NEWSe' in load_model[model] and not 'latlon_dims_keep' in locals():
                #                 try:
                #                     f = Dataset(GRIB_PATH_TEMP+"/"+load_data_nc[model], "a", format="NETCDF4")
                #                     latlon_sub = [[np.nanmin(np.array(f.variables['lon'][:]))-0.5,np.nanmin(np.array(f.variables['lat'][:]))-0.1,\
                #                         np.nanmax(np.array(f.variables['lon'][:])),np.nanmax(np.array(f.variables['lat'][:]))+0.5]]
                #                     domain_sub = ['ALL']
                #                     f.close()
                #                 except:
                #                     pass
                        
                #         #Regrid to ST4 grid using regrid_data_plane
                #         data_success[fcst_hr_count,model] = os.system(MET_PATH+"/regrid_data_plane "+GRIB_PATH_TEMP+"/"+ \
                #             load_data_nc[model]+" "+GRIB_PATH+"/temp/ensmean_sample_NODELETE "+GRIB_PATH_TEMP+"/"+load_data_nc[model]+"2 "+ \
                #             "-field 'name=\""+APCP_str_end+"\"; level=\""+APCP_str_end+ \
                #             "\";' -name "+APCP_str_end)

                #         #Remove original netcdf files
                #         output=os.system("rm -rf "+GRIB_PATH_TEMP+"/"+load_data_nc[model])
        
                #         #Rename netcdf file
                #         data_name_nc = np.append(data_name_nc,GRIB_PATH_TEMP+"/"+load_data_nc[model][0:-3]+"_f"+fcst_hr_str+fcst_min_str+".nc")
                #         output=os.system("mv "+GRIB_PATH_TEMP+"/"+load_data_nc[model]+"2 "+data_name_nc[fcst_hr_count])
                            
                #         #Construct an array of filenames in the previous model portion of the loop when in retro mode
                #         if (mtd_mode == 'Retro') and (load_model[0] not in load_data[model]):
                #             data_name_nc_prev = np.append(data_name_nc_prev,GRIB_PATH_TEMP+"/"+load_data[0]+"_f"+fcst_hr_str+fcst_min_str+".nc")
                #         else:
                #             data_name_nc_prev = np.append(data_name_nc_prev,'null')
        
                #         METLoadEnsemble.snow_mask(MET_PATH,GRIB_PATH,GRIB_PATH_TEMP,load_data[model],load_qpe[0],data_name_nc,data_name_grib,\
                #             data_name_grib_prev,APCP_str_end,fcst_hr_count,snow_mask)
                       
                #         #Apply a regional mask if wanted
                #         METLoadEnsemble.reg_mask(MET_PATH,mtd_mode,load_data[model],load_qpe[0],data_name_nc[fcst_hr_count],\
                #             data_name_nc_prev[fcst_hr_count],APCP_str_end,reg_mask_file)
     
                #         fcst_hr_count += 1
                #     #END through forecast hour

                #     #Remove original grib files
                #     for files in data_name_grib:
                #         output = os.system('rm -rf '+GRIB_PATH_TEMP+'/'+files+'*')
                      
                #     #Determine which hours are successfully loaded for each model
                #     if (mtd_mode == 'Retro'):
                #         hour_success = (data_success[:,[i for i in range(0,len(load_data)) if load_data[i] == load_qpe[0]][0]] + \
                #             data_success[:,[i for i in range(0,len(load_data)) if load_data[i] == load_model[0]][0]]) == 0
                #     elif (mtd_mode == 'Operational'): #No OBS, compare model to itself 
                #         hour_success = data_success[:,model] == 0
                    
                #     #If not all of the data is in during operational mode, then pause for 2 minutes and try again. Repeat for one hour
                #     if np.nanmean(data_success[:,model]) != 0 and mtd_mode == 'Operational':
                #         if (datetime.datetime.now() - datetime_now).seconds > ops_check:
                #             print('Missing Model Data; Plotting What We Have')
                #             break
                #         else:
                #             print('There is missing Model Data; Pausing')
                #             time.sleep(120)
                #     elif np.nanmean(data_success[:,model]) == 0 and mtd_mode == 'Operational':
                #         break
                #     else:
                #         break 

                # #END while check for all model data  
                # print("HERE 2")
                
                # #Create variable name for default MTD file and renamed MTD file
                # if ((mtd_mode == 'Retro' and load_qpe[0] in load_data[model]) or (mtd_mode == 'Operational')):
                #     #Find first tracked forecast hour for mtd output label
                #     hours_MTDlabel    = (((np.argmax(hour_success == 1)+1)*pre_acc)+int(load_data[model][-2:])+int(init_yrmondayhr[model][-2:]))
                #     curdate_MTDlabel  = curdate + datetime.timedelta(hours = ((np.argmax(hour_success == 1)+1)*pre_acc))
                #     yrmonday_MTDlabel = str(curdate_MTDlabel.year)+'{:02d}'.format(int(curdate_MTDlabel.month))+'{:02d}'.format(int(curdate_MTDlabel.day))

                #     if hours_MTDlabel >= 24:
                #         hours_MTDlabel = hours_MTDlabel - 24
                #     mins_MTDlabel     = '{:02d}'.format(int(((hours_MTDlabel - int(hours_MTDlabel))*60)))
                #     hours_MTDlabel    = '{:02d}'.format(int(hours_MTDlabel))

                #     MTDfile_old[model] = 'mtd_'+yrmonday_MTDlabel+'_'+hours_MTDlabel+mins_MTDlabel+'00V'
                # MTDfile_new[model] = 'mtd_'+init_yrmondayhr[model][:-2]+'_h'+init_yrmondayhr[model][-2:]+'_f'+last_fcst_hr_str+'_'+ \
                #     load_data_nc[model][0:-3]+'_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(thres)

                # print("HERE 3")
                # #Remove old MTD output files if they exist
                # if os.path.isfile(GRIB_PATH_TEMP+'/'+MTDfile_new[model]): 
                #     os.remove(GRIB_PATH_TEMP+'/'+MTDfile_new[model])    
     
                # #In retro mode, if there is no data in the model, then quit this attempt
                # if mtd_mode == 'Retro' and load_data[model] != load_qpe[0] and np.nanmean(data_success[:,model]) == 1:
                #     print('skipped')
                #     break
                
                # print("HERE 4")
                # #Run MTD: 1) if QPE exists compare model to obs, otherwise 2) run mtd in single mode   
                # output = []
                # if mtd_mode == 'Retro' and load_qpe[0] in load_data[model]:
                #     print(MET_PATH+'/mtd -fcst '+' '.join(data_name_nc_prev[hour_success])+ \
                #         ' -obs '+' '.join(data_name_nc[hour_success])+' -config '+config_name+' -outdir '+GRIB_PATH_TEMP)
                #     mtd_success = os.system(MET_PATH+'/mtd -fcst '+' '.join(data_name_nc_prev[hour_success])+ \
                #         ' -obs '+' '.join(data_name_nc[hour_success])+' -config '+config_name+' -outdir '+GRIB_PATH_TEMP)
                # elif mtd_mode == 'Operational' and load_qpe[0] not in load_data[model]: #No QPE, compare model to itself
                #     mtd_success = os.system(MET_PATH+'/mtd -single '+' '.join(data_name_nc[hour_success])+' -config '+ \
                #         config_name+' -outdir '+GRIB_PATH_TEMP)
                
                # #Matrix to gather all netcdf file strings
                # data_name_nc_all = np.append(data_name_nc_all,data_name_nc)

                # #Rename cluster file and 2d text file (QPE is handled later)
                # if ((mtd_mode == 'Retro' and load_qpe[0] in load_data[model]) or (mtd_mode == 'Operational')):
                #     output = os.system('mv '+GRIB_PATH_TEMP+'/'+MTDfile_old[model]+'_obj.nc '+ GRIB_PATH_TEMP+'/'+MTDfile_new[model]+ \
                #         '_obj.nc ')
                #     output = os.system('mv '+GRIB_PATH_TEMP+'/'+MTDfile_old[model]+'_2d.txt '+ GRIB_PATH_TEMP+'/'+MTDfile_new[model]+ \
                #         '_2d.txt ')
                #     output = os.system('rm -rf '+GRIB_PATH_TEMP+'/'+MTDfile_old[model]+'_3d_ss.txt')
                #     output = os.system('rm -rf '+GRIB_PATH_TEMP+'/'+MTDfile_old[model]+'_3d_sc.txt')
                #     output = os.system('rm -rf '+GRIB_PATH_TEMP+'/'+MTDfile_old[model]+'_3d_ps.txt')
                #     output = os.system('rm -rf '+GRIB_PATH_TEMP+'/'+MTDfile_old[model]+'_3d_pc.txt')
