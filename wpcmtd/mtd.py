import os
import dataclasses
import pathlib
import datetime
import numpy as np
import time
from netCDF4 import Dataset
from .utils import adjust_date_range, gen_mtdconfig_15m, gen_mtdconfig, load_data_str
from .plot import mtd_plot_retro, mtd_plot_all_fcst, mtd_plot_tle_fcst, mtd_plot_all_snow_fcst


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
    config_path:    pathlib.PosixPath # Location of MET config files
    track_path:     pathlib.PosixPath # Location to store NPZ track files
    latlon_dims:    [float]           # Latitude/longitude dimensions for plotting [WLON,SLAT,ELON,NLAT]
    beg_date:       datetime.datetime # Beginning time of retrospective runs
    end_date:       datetime.datetime # Ending time of retrospective runs
    init_inc:       int               # Initialization increment (hours; e.g. total hours between model init. cycles)
    mtd_mode:       str               # 'Operational', 'Retro', or 'Both'
    load_model:     [str]             # list of models to load, ex.  ['HRRRv4_TLE_lag12'   ,'HRRRv4_TLE_lag06'  ,'HRRRv4_TLE_lag00']
    load_qpe:       [str]             # 'ST4','MRMS',or 'NONE' (NOTE: NEWS-E and HRRRe DOES NOT WORK FOR ANYTHING OTHER THAN LAG = 00)
    snow_mask:      bool              # Keep only regions that are snow (True) or no masking (False)
    pre_acc:        float             # Precipitation accumulation total
    season_sub:     [int]             # Seasonally subset retro data to match oper data (array of months to keep)
    grib_path_des:  str               # An appended string label for the temporary directory name
    thresh:         float             # Precip. threshold for defining objects (inches) (specifiy single value)
    end_fcst_hrs:   float             # Last forecast hour to be considered
    conv_radius:    int               # Radius of the smoothing (grid squares)
    min_volume:     int               # Area threshold (grid squares) to keep an object
    ti_thresh:      float             # Total interest threshold for determining matches
    plot_allhours:  bool              # Plot every track figure hour-by-hour (good for comparing tracker to eye)
    sigma:          int               # Gaussian Filter for smoothing ensemble probabilities in some plots (grid points)
    domain_sub:     [str]             # Sub-domain names for plotting
    latlon_sub:     [[int]]           # Sub-domain lat/lon sizes for plotting

    transfer_to_prod: bool = False
    grib_path_temp: pathlib.PosixPath = None
    lat:            float = None
    lon:            float = None
    offset_fcsthr:  [float] = dataclasses.field(default_factory=list) 
    acc_int:        [int] = dataclasses.field(default_factory=list) 
    init_yrmondayhr:[] = dataclasses.field(default_factory=list) 
    reg_mask_file:  [] = dataclasses.field(default_factory=list) 
    ops_check:      float = 60*60*0.0   


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
            self.fig_path = str(self.fig_path)+'RETRO/'
            if len(self.load_model) > 1:
                raise ValueError('If mtd_mode = \'Retro\' then only one model specification at a time')
        elif self.mtd_mode == 'Both':
            strip = [model[0:model.find('TLE')] for model in self.load_model]
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
        """
        Pre-processing step to load the CAM model data and/or ST4 obs located 
        at /export/hpc-lw-dtbdev5/wpc_cpgffh/gribs on hpc-lw-dtbdev5. Since most
        CAM data is initialized at different intervals, this function does most 
        of the grunt work to load the proper strings, variables, and create the proper folders.

        When adding another ensemble member, take the following steps:
        1) Ensure the new model string name matches a partial string name in 'model_default,'
        if not then add a partial string to model_default
        2) Add the frequency of initalizations in UTC to 'run_times' and 'acc_int'

        Parameters
        ----------
        curr_date : datetime.datetime object
            Start date of model data to be loaded.
        
        Returns
        -------
            None
        """

        #Create an array of partial strings to match the load data array specified by the user
        model_default=['ST4','MRMS_','MRMS15min','MRMS2min','NAM','HIRESW','HRRR','NSSL-OP','NSSL-ARW',\
            'NSSL-NMB','HRAM','NEWSe5min','NEWSe60min','HRRRe60min','FV3LAM','HRRRv415min']

        #Convert numbers to strings of proper length
        yrmondayhr=curr_date.strftime('%Y%m%d%H')

        #Given the initalization hour, choose the most recent run for each model, archive initialization times
        self.init_yrmondayhr=[]
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
                if pygrib.datetime_to_julian(curr_date) >= pygrib.datetime_to_julian(datetime.datetime(2017,3,21,6,0)):
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
                self.init_yrmondayhr=np.append(self.init_yrmondayhr,curdate_temp.strftime('%Y%m%d%H'))
            else: #If lag is misspecified, compute with no lag
                lag = 0
                init_offset = curr_date.hour - run_times[np.argmax(run_times>curr_date.hour)-1]
                self.offset_fcsthr=np.append(self.offset_fcsthr,init_offset+lag)
                curdate_temp=curr_date-datetime.timedelta(hours=int(self.offset_fcsthr[model]))
                self.init_yrmondayhr=np.append(self.init_yrmondayhr,curdate_temp.strftime('%Y%m%d%H'))
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


    def pcp_combine(self, model, load_data_nc, data_name_grib, yrmonday_ahead, \
        hrmin_ahead, fcst_hr, APCP_str_beg, APCP_str_end, fcst_hr_count):
        """
        Uses MET pcp_combine to sum the various model/analysis data tot he precipitation
        intervals specified in pre_acc. This is the first MET step & involves converting
        grib2 files to MET-friendly netcdf files.

        Parameters
        ----------
        model : int
            model counter (can include analysis)
        load_data_nc : string 
            netCDF string of data to be loaded
        data_name_grib : string
            grib name string of data to be loaded
        yrmonday_ahead : string  
            yymmdd at the forecast hour
        hrmin_ahead : string
            hhmm at the forecast hour
        fcst_hr : int
            number of forecast hour
        APCP_str_beg : string
            grib variable name to be loaded (beginning)
        APCP_str_end : string
            grib variable name to be loaded (end)

        
        Returns
        -------
            None
        """

        #Create variable name for pcp_combine
        if self.load_qpe[0] in self.load_data:
            pcp_combine_str = '00000000_000000'
        else:
            pcp_combine_str = self.init_yrmondayhr[model][:-2]+'_'+self.init_yrmondayhr[model][-2:]+'0000'

        #Create variables names for pcp_combine
        if 'ST4' in self.load_data:
            pcp_combine_str_beg = '00000000_000000'
        else:
            pcp_combine_str_beg = self.init_yrmondayhr[model][:-2]+'_'+self.init_yrmondayhr[model][-2::]+'0000 '
        pcp_combine_str_end = yrmonday_ahead+'_'+hrmin_ahead+'00 '

        #Use MET pcp_combine to sum, except for 1 hr acc from NAM, which contains either 1, 2, 3 hr acc. precip
        os.chdir(self.grib_path_temp)
        if 'HIRESNAM' in self.load_data and self.pre_acc == 1:
            if ((fcst_hr - 1) % 3) == 0:   #Every 3rd hour contains 1 hr acc. precip
                output = os.system(str(self.met_path)+'/pcp_combine -sum '+pcp_combine_str_beg+' '+APCP_str_beg[1::]+' '+ \
                    pcp_combine_str_end+APCP_str_end[1::]+' '+str(self.grib_path_temp)+'/'+load_data_nc+' -pcpdir '+ \
                    str(self.grib_path_temp)+'/ -name "'+APCP_str_end+'"')
            elif ((fcst_hr - 2) % 3) == 0: #2nd file contains 2 hr  acc. precip
                output = os.system(str(self.met_path)+'/pcp_combine -subtract '+str(self.grib_path_temp)+'/'+data_name_grib[fcst_hr_count]+' 020000 '+ \
                    str(self.grib_path_temp)+'/'+data_name_grib[fcst_hr_count-1]+' 010000 '+str(self.grib_path_temp)+'/'+load_data_nc+ \
                    ' -name "'+APCP_str_end+'"')
            elif ((fcst_hr - 3) % 3) == 0: #3rd file contains 3 hr  acc. precip
                output = os.system(str(self.met_path)+'/pcp_combine -subtract '+str(self.grib_path_temp)+'/'+data_name_grib[fcst_hr_count]+' 030000 '+ \
                    str(self.grib_path_temp)+'/'+data_name_grib[fcst_hr_count-1]+' 020000 '+str(self.grib_path_temp)+'/'+load_data_nc+ \
                    ' -name "'+APCP_str_end+'"')
        elif 'HIRESNAM' in self.load_data and self.pre_acc % 3 == 0: #NAM increments divisible by 3
            output = os.system(str(self.met_path)+'/pcp_combine -sum '+pcp_combine_str_beg+' 030000 '+ \
                pcp_combine_str_end+' '+APCP_str_end[1::]+' '+str(self.grib_path_temp)+'/'+ load_data_nc+ \
                ' -pcpdir '+str(self.grib_path_temp)+'/  -name "'+APCP_str_end+'"')
        elif 'HIRESNAM' in self.load_data and self.pre_acc % 3 == 1: #Error if desired interval is not 1 or divisible by 3
            raise ValueError('NAM pre_acc variable can only be 1 or divisible by 3')
        elif 'NEWSe60min' in self.load_data:
            if fcst_hr_load == 1: #If using fcst hour 1, no need to subtract data
                output = os.system(str(self.met_path)+'/pcp_combine -sum '+pcp_combine_str_beg+' '+APCP_str_beg[1::]+' '+pcp_combine_str_end+' '+ \
                    APCP_str_end[1::]+' '+str(self.grib_path_temp)+'/'+load_data_nc+' -name "'+APCP_str_end+'"')
            # elif fcst_hr_load > 1: #If not first hour, subtract current from previous hour
            #     output = os.system(str(self.met_path)+"/pcp_combine -subtract "+GRIB_PATH_TEMP+"/"+data_name_grib[fcst_hr_count]+" "+last_fcst_hr_str+" "+ \
            #         GRIB_PATH_TEMP+"/"+data_name_grib[fcst_hr_count-1]+" "+'{:02d}'.format(int(last_fcst_hr_str)-1)+" "+GRIB_PATH_TEMP+"/"+ \
            #         load_data_nc+' -name "'+APCP_str_end+'"')
        elif 'NEWSe5min' in self.load_data:
            output = os.system(str(self.met_path)+'/pcp_combine -sum  '+pcp_combine_str_beg+' '+APCP_str_beg[1::]+' '+pcp_combine_str_end+' '+ \
                APCP_str_end[1::]+' '+str(self.grib_path_temp)+'/'+load_data_nc+' -field \'name="'+APCP_str_beg+'"; level="Surface";\''+ \
                ' -name "'+APCP_str_end+'"')
        elif 'MRMS_' in self.load_data:
            call_str = [data_name_grib[i]+' \'name="MultiSensor_QPE_01H_Pass2" ; level="L0" ; \'' for i in range(int((fcst_hr_count)*(self.pre_acc/acc_int)),\
                int((fcst_hr_count+1)*(self.pre_acc/acc_int)))]
            output = os.system(str(self.met_path)+'/pcp_combine -add  '+' '.join(call_str)+' '+ \
                str(self.grib_path_temp)+'/'+load_data_nc+' -name "'+APCP_str_end+'" -v 3')
        elif 'MRMS15min_' in self.load_data:
            call_str = [data_name_grib[i]+' \'name="RadarOnly_QPE_15M" ; level="L0" ; \'' for i in range(int((fcst_hr_count)*(self.pre_acc/acc_int)),\
                int((fcst_hr_count+1)*(self.pre_acc/acc_int)))]
            output = os.system(str(self.met_path)+'/pcp_combine -add  '+' '.join(call_str)+' '+ \
                str(self.grib_path_temp)+'/'+load_data_nc+' -name "'+APCP_str_end+'"')
        elif 'HRRRe' in self.load_data:
            if self.pre_acc != 1:
                raise ValueError('HRRRe pre_acc variable can only be set to 1')
            #HRRRe accumulates precip, so must subtract data from previous hour to obtain 1 hour acc. precip.
            if fcst_hr == 1:
                output = os.system(str(self.met_path)+'/pcp_combine -sum '+pcp_combine_str_beg+' '+APCP_str_beg[1::]+' '+ \
                    pcp_combine_str_end+APCP_str_end[1::]+' '+str(self.grib_path_temp)+'/'+load_data_nc+' -pcpdir '+ \
                    str(self.grib_path_temp)+'/ -name "'+APCP_str_end+'"')
            else:
                output = os.system(str(self.met_path)+'/pcp_combine -subtract '+str(self.grib_path_temp)+'/'+data_name_grib[fcst_hr_count]+ \
                    ' '+'{:02d}'.format(fcst_hr_count+1)+'0000 '+str(self.grib_path_temp)+'/'+data_name_grib[fcst_hr_count-1]+' '+ \
                    '{:02d}'.format(fcst_hr_count)+'0000 '+str(self.grib_path_temp)+'/'+load_data_nc+ ' -name "'+APCP_str_end+ \
                    '"')
        else:
            output = os.system(str(self.met_path)+'/pcp_combine -sum  '+pcp_combine_str_beg+' '+APCP_str_beg[1::]+' '+pcp_combine_str_end+' '+ \
                APCP_str_end[1::]+' '+str(self.grib_path_temp)+'/'+load_data_nc+' -name "'+APCP_str_end+'"')
        
        time.sleep(2)

    def apply_snow_mask(self, model, data_name_nc, data_name_grib, data_name_grib_prev, \
        APCP_str_end, fcst_hr_count):

        """
        Applies a snow mask to the various model/analysis data.

        Parameters
        ----------
        model : int
            model counter (can include analysis)
        data_name_nc : string 
            netCDF string of data to be loaded
        data_name_grib : string
            grib name string of data to be loaded
        data_name_grib_prev : string
            grib name of previous data file to be loaded
        fcst_hr : int
            number of forecast hour
        APCP_str_end : string
            grib variable name to be loaded (end)
        fcst_hr_count : int
            counter through forecast hours

        Returns
        -------
            None
        """

        if self.snow_mask == True:

            #Determine masking string
            if 'NSSL' in self.load_data[model]:
                masking_str = 'WEASD'
                masking_val = 'le0'
            elif 'MRMS' in self.load_data[model]:
                masking_str = 'PrecipFlag'
                masking_val = 'ne3'
            else:
                masking_str = 'CSNOW'
                masking_val = 'le0'

            #Special conditions added because ST4 doesn't have masking option, so use model
            if 'ST4' not in self.load_data[model] and 'ST4' not in self.load_qpe: #No ST4 being loaded now or at all
                mask_file_in      = str(self.grib_path_temp)+'/'+data_name_grib[fcst_hr_count]
                mask_file_ou      = str(self.grib_path_temp)+'/'+data_name_grib[fcst_hr_count]+'.nc'
            elif 'ST4' not in self.load_data[model] and 'ST4' in self.load_qpe:   #No ST4 being loaded now, but will be loaded
                mask_file_in      = str(self.grib_path_temp)+ '/'+data_name_grib[fcst_hr_count]
                mask_file_ou      = str(self.grib_path_temp)+ '/'+data_name_grib[fcst_hr_count]+'.nc'
                mask_file_in_prev = str(self.grib_path_temp)+'_2/'+data_name_grib[fcst_hr_count]

                try:
                    os.mkdir(str(self.grib_path_temp)+'_2/')
                except FileExistsError:
                    pass

                os.system('cp '+mask_file_in+' '+mask_file_in_prev)
            elif 'ST4' in self.load_data[model]:                             #ST4 being loaded now
                mask_file_in      = str(self.grib_path_temp)+'_2/'+data_name_grib_prev[fcst_hr_count]
                mask_file_ou      = str(self.grib_path_temp)+ '/'+data_name_grib_prev[fcst_hr_count]+'.nc'

            #Regrid the data to common grid
            os.system(str(self.met_path)+'/regrid_data_plane '+mask_file_in+' '+str(self.grib_path)+'/temp/ensmean_sample_NODELETE '+ \
                mask_file_ou+' '+ '-field \'name="'+masking_str+'"; level="L0";\' -name "'+APCP_str_end+'"')

            #If considering WEASD, must subtract current hour from previous hour (if it exists)
            if masking_str == 'WEASD' and fcst_hr_count > 0:
                mask_file_ou_prev = str(self.grib_path_temp)+'/'+data_name_grib[fcst_hr_count-1]+'.nc'

                output = os.system(str(self.met_path)+'/pcp_combine -subtract '+mask_file_ou+' \'name="WEASD_L0"; level="L0";\' '+ \
                    mask_file_ou_prev+' \'name="WEASD_L0"; level="L0";\' '+mask_file_ou+'2')

                #Mask out all non-snow data
                os.system(str(self.met_path)+'/gen_vx_mask '+data_name_nc[fcst_hr_count]+' '+mask_file_ou+'2 '+data_name_nc[fcst_hr_count]+ \
                    '_2 -type data -input_field \'name="'+APCP_str_end+ \
                    '"; level="(*,*)";\' -mask_field \'name="'+APCP_str_end+'"; level="(*,*)";\' -thresh '+masking_val+' -value 0 -name '+APCP_str_end)
            else:
                output = os.system("cp "+mask_file_ou+' '+mask_file_ou+'2')
                #Mask out all non-snow data
                os.system(str(self.met_path)+'/gen_vx_mask '+data_name_nc[fcst_hr_count]+' '+mask_file_ou+'2 '+data_name_nc[fcst_hr_count]+ \
                    '_2 -type data -input_field \'name="'+APCP_str_end+ \
                    '"; level="(*,*)";\' -mask_field \'name="'+APCP_str_end+'"; level="(*,*)";\' -thresh '+masking_val+' -value 0 -name '+APCP_str_end)
            #Replace old file with properly masked file
            os.system("mv "+data_name_nc[fcst_hr_count]+"_2 "+data_name_nc[fcst_hr_count])


    def apply_reg_mask(self, model, data_name_nc, data_name_nc_prev, APCP_str_end, fcst_hr_count):
        """
        Applies a regional mask to the data.

        Parameters
        ----------
        model : int
            model counter (can include analysis)
        data_name_nc : string 
            name of analysis file
        data_name_nc_prev : string
            name of model file
        APCP_str_end : string
            grib variable name to be loaded (end)
        fcst_hr_count : int
            counter through forecast hours

        Returns
        -------
            None
        """
        if self.reg_mask_file != []:
            if (self.mtd_mode == 'Retro') and (self.load_qpe[0] in self.load_data[model]):
                if 'MRMS' in self.load_qpe[0] or 'ST4' in self.load_qpe[0]:
                    #Mask the model
                    os.system(str(self.met_path)+'/gen_vx_mask -type poly '+data_name_nc_prev[fcst_hr_count]+' '+self.reg_mask_file+' '\
                        +data_name_nc_prev[fcst_hr_count]+'2 -complement -input_field \'name="'+APCP_str_end+'" ; level="(*,*)" ;\''\
                        +' -value -9999 -name '+APCP_str_end)
                    os.system('mv '+data_name_nc_prev[fcst_hr_count]+'2 '+data_name_nc_prev[fcst_hr_count])

                    #Mask the analysis
                    os.system(str(self.met_path)+'/gen_vx_mask -type poly '+data_name_nc[fcst_hr_count]+' '+self.reg_mask_file+' '\
                        +data_name_nc[fcst_hr_count]+'2 -complement -input_field \'name="'+APCP_str_end+'" ; level="(*,*)" ;\''\
                        +' -value -9999 -name '+APCP_str_end)
                    os.system('mv '+data_name_nc[fcst_hr_count]+'2 '+data_name_nc[fcst_hr_count])

    def load_data_MTDV90(self, MTDfile_new, model):
        """
        load_data_MTDV90 loads MTD object data. This includes gridded simple and cluster binary
        data(simp_bin,clus_bin), which separately considers model/observation. Since
        the object attribute property data (e.g. simp_prop/pair_prop) includes both model
        and observation data (if obs exists), the same is true here. Hence, if both model
        and observation are requested, this function will load both to include both in
        object attribute files, but keeps model/obs separate in binary files.
        
        NOTE: The current 2d.txt MTD file includes simple-single and cluster-pairs, but they are
        not well labeled. The first lines start with simple-single forecast objects (e.g. OBJECT_ID=F_1),
        and show how they are merged into clusters (e.g. OBJECT_CAT=CF_1). Observations are then shown
        in the same format. Thereafter, forecast data is shown again for OBJECT_ID and OBJECT_CAT.
        At this point, the data is cluster-pairs for forecast data, with the cluster ID matching the
        observation cluster ID. The forecast cluster ID will match the earlier forecast cluster ID
        if that cluster did not involve multiple mergers, otherwise it is the centroid of the mergers.
        This code will gather the first round of forecast/obs into 'simp_prop' and the second into
        'pair_prop.'

        Parameters
        ----------
            MTDfile_new : string
                QPE string for the netcdf converted data (must point to model, not ST4. ST4 data is in model)
            model : int
                model counter (can include analysis)

        Returns
        -------
            lat : [float]
                grid of latitude from netcdf MODE output
            lon : [float]
                grid of longitude from netcdf MODE output
            grid_mod : [float]
                grid of raw forecast values
            grid_obs : [float]
                grid of raw observation values
            simp_bin : [float]
                grid of simple object ID information
            clus_bin : [float]
                grid of cluster ID information
            simp_prop : [float]
                simple centroid info of OBJECT_ID','OBJECT_CAT','TIME_INDEX','AREA','CENTROID_LAT',
                'CENTROID_LON','AXIS_ANG','10TH INTEN PERC','50TH INTEN PERC','90TH INTEN PERC'
            pair_prop : [float]
                paired centroid info of OBJECT_ID','OBJECT_CAT','TIME_INDEX','AREA','CENTROID_LAT',
                'CENTROID_LON','AXIS_ANG','10TH INTEN PERC','50TH INTEN PERC','90TH INTEN PERC'
            data_success : int
                file successfully loaded if equals zero
        """

        head_str_obj    = ['OBJECT_ID']
        head_len_obj    = [11]
        head_str_simp   = ['OBJECT_ID','OBJECT_CAT','TIME_INDEX','AREA','CENTROID_LAT','CENTROID_LON','AXIS_ANG']
        head_len_simp   = [11,            12,           12,         6,       14,           14,            8]

        simp_prop = np.ones((9,10000))*-999
        pair_prop = np.ones((9,10000))*-999

        #Check to see if file exists
        if os.path.isfile(str(self.grib_path_temp)+'/'+MTDfile_new+'_obj.nc'):

            #If file exists, try to open it up
            try:

                #####1) First read in the 2d text file to gather centroid intensity and location
                #Open file and read through it
                target = open(str(self.grib_path_temp)+'/'+MTDfile_new+'_2d.txt','r')
                data = target.readlines()
                target.close()

                #Loop through each line and gather the needed info
                f_count         = 0       #counter by line in file
                f_simp_count    = 0       #counter by each simple attribute ('OBJECT_ID','TIME_INDEX',etc)
                f_pair_count    = 0       #counter by each pair attribute ('OBJECT_ID','TIME_INDEX',etc)
                line_prev       = []      #load the previous line
                pair_load       = False   #reaching paired data line in data files

                for line in data:
                    
                    #Find location of data in each file using header information
                    if f_count == 0:
                        line1 = line
                        array_obj = [line1.find(head_str_obj[0]),line1.find(head_str_obj[0])+head_len_obj[0]]

                    if f_count > 0:
                        #Determine if paired objects exist (when objects go from obs back to fcst)
                        if 'F' in line[line1.find(head_str_simp[0])+5:line1.find(head_str_simp[0])+8] and \
                            'O' in line_prev[line1.find(head_str_simp[0])+5:line1.find(head_str_simp[0])+8]: #THIS INSTANCE ONLY HAPPENS ONCE!
                            pair_load = 'TRUE'

                        #Isolate simple cluster model/obs objects, gather stats comparing mod to obs in the form:
                        #[object ID, object cat, time index, area of object, centroid lat, centroid lon, axis angle]

                        for header in range(0,len(head_str_simp)):

                            #Gather data grabbing arrays; special conditions for getting OBJECT_ID/OBJECT_CAT
                            if header == 0 or header == 1:
                                array_simp = [line1.find(head_str_simp[header])+4,line1.find(head_str_simp[header])+head_len_simp[header]]
                            else:
                                array_simp = [line1.find(head_str_simp[header]),line1.find(head_str_simp[header])+head_len_simp[header]]

                            if pair_load: #Paired data saved to 'pair_prop'

                                if ('F' in line[array_obj[0]:array_obj[-1]]): # and 'ST4' not in MTDfile_new:

                                    #Special consideration for cluster data
                                    if head_str_simp[header] == 'OBJECT_ID': #object ID
                                        temp_d = line[array_simp[0]:array_simp[-1]]
                                        temp_d = temp_d.replace('CF','')
                                        pair_prop[header,f_pair_count] = float(temp_d)
                                    elif head_str_simp[header] == 'OBJECT_CAT': #Cluster ID
                                        if 'NA' in line[array_simp[0]:array_simp[-1]]:
                                            pair_prop[header,f_pair_count] = np.NaN
                                        else:
                                            temp_d = line[array_simp[0]+4:array_simp[-1]]
                                            temp_d = temp_d.replace('F','')
                                            pair_prop[header,f_pair_count] = float(temp_d)
                                    else:
                                        pair_prop[header,f_pair_count] = float(line[array_simp[0]:array_simp[-1]])

                                    if header == len(head_str_simp)-1:
                                        f_pair_count += 1

                                elif ('O' in line[array_obj[0]:array_obj[-1]]): # and 'ST4' in MTDfile_new:

                                    #Make observation ID's negative to differentiate from model
                                    if head_str_simp[header] == 'OBJECT_ID': #object ID
                                        temp_d = line[array_simp[0]:array_simp[-1]]
                                        temp_d = temp_d.replace('CO','')
                                        pair_prop[header,f_pair_count] = -float(temp_d)
                                    elif head_str_simp[header] == 'OBJECT_CAT': #Cluster ID
                                        if 'NA' in line[array_simp[0]:array_simp[-1]]:
                                            pair_prop[header,f_pair_count] = np.NaN
                                        else:
                                            temp_d = line[array_simp[0]+4:array_simp[-1]]
                                            temp_d = temp_d.replace('O','')
                                            pair_prop[header,f_pair_count] = -float(temp_d)
                                    else:
                                        pair_prop[header,f_pair_count] = float(line[array_simp[0]:array_simp[-1]])

                                    if header == len(head_str_simp)-1:
                                        f_pair_count += 1

                            else: #non-paired data saved to 'simp_prop'

                                if ('F' in line[array_obj[0]:array_obj[-1]]): # and 'ST4' not in MTDfile_new:

                                    #Special consideration for cluster data
                                    if head_str_simp[header] == 'OBJECT_ID': #object ID
                                        temp_d = line[array_simp[0]:array_simp[-1]]
                                        temp_d = temp_d.replace('F','')
                                        simp_prop[header,f_simp_count] = float(temp_d)
                                    elif head_str_simp[header] == 'OBJECT_CAT': #Cluster ID
                                        if 'NA' in line[array_simp[0]:array_simp[-1]]:
                                            simp_prop[header,f_simp_count] = np.NaN
                                        else:
                                            temp_d = line[array_simp[0]+4:array_simp[-1]]
                                            temp_d = temp_d.replace('F','')
                                            simp_prop[header,f_simp_count] = float(temp_d)
                                    else:
                                        simp_prop[header,f_simp_count] = float(line[array_simp[0]:array_simp[-1]])

                                    if header == len(head_str_simp)-1:
                                        f_simp_count += 1

                                elif ('O' in line[array_obj[0]:array_obj[-1]]): # and 'ST4' in MTDfile_new:

                                    #Make observation ID's negative to differentiate from model
                                    if head_str_simp[header] == 'OBJECT_ID': #object ID
                                        temp_d = line[array_simp[0]:array_simp[-1]]
                                        temp_d = temp_d.replace('O','')
                                        simp_prop[header,f_simp_count] = -float(temp_d)
                                    elif head_str_simp[header] == 'OBJECT_CAT': #Cluster ID
                                        if 'NA' in line[array_simp[0]:array_simp[-1]]:
                                            simp_prop[header,f_simp_count] = np.NaN
                                        else:
                                            temp_d = line[array_simp[0]+4:array_simp[-1]]
                                            temp_d = temp_d.replace('O','')
                                            simp_prop[header,f_simp_count] = -float(temp_d)
                                    else:
                                        simp_prop[header,f_simp_count] = float(line[array_simp[0]:array_simp[-1]])

                                    if header == len(head_str_simp)-1:
                                        f_simp_count += 1

                    line_prev = line
                    f_count += 1

                #Create obj. intensity since METv6.0 doesnt output it directly. First initialize additional row
                simp_prop = np.concatenate((simp_prop,np.full((1,simp_prop.shape[1]),np.NaN)),axis=0)
                pair_prop = np.concatenate((pair_prop,np.full((1,pair_prop.shape[1]),np.NaN)),axis=0)

                #Remove unpopulated dimensions
                simp_prop = simp_prop[:, np.nanmean(simp_prop[0:7,:] == -999, axis=0) == 0]
                pair_prop = pair_prop[:, np.nanmean(pair_prop[0:7,:] == -999, axis=0) == 0]

                #####2) Next read in the centroid shape data from the netcdf file
                f = Dataset(str(self.grib_path_temp)+'/'+MTDfile_new+'_obj.nc', "a", format="NETCDF4")
                lat=f.variables['lat'][:]
                lon=f.variables['lon'][:]

                #Try loading gridded obs to match data in 'simp_prop'
                try:
                    simp_bin_obs = f.variables['obs_object_id']
                    simp_bin_obs = simp_bin_obs[:]
                    simp_bin_obs[simp_bin_obs < 0] = 0

                    #Cluster object data
                    try: #May not always have clusters
                        clus_bin_obs = f.variables['obs_cluster_id']
                        clus_bin_obs = clus_bin_obs[:]
                        clus_bin_obs[clus_bin_obs < 0] = 0
                    except KeyError:
                        clus_bin_obs  = np.zeros((simp_bin_obs.shape))

                    #Rraw data
                    grid_obs = f.variables['obs_raw'][:]/25.4
                    grid_obs[grid_obs < 0] = np.nan
                except:
                     grid_obs = []

                #There is always gridded forecast object data to load
                simp_bin_mod = f.variables['fcst_object_id']
                simp_bin_mod = simp_bin_mod[:]
                simp_bin_mod[simp_bin_mod < 0] = 0

                #Cluster object data
                try: #May not always have clusters
                    clus_bin_mod = f.variables['fcst_cluster_id']
                    clus_bin_mod = clus_bin_mod[:]
                    clus_bin_mod[clus_bin_mod < 0] = 0
                except KeyError:
                    clus_bin_mod = np.zeros((simp_bin_mod.shape))

                #Raw data
                grid_mod = f.variables['fcst_raw'][:]/25.4
                grid_mod[grid_mod < 0] = np.nan

                #Loop through each line of 'simp_prop' and determine obj. 10,50,90th prctile
                #['OBJECT_ID','OBJECT_CAT','TIME_INDEX','AREA','CENTROID_LAT','CENTROID_LON','AXIS_ANG','INTENSITY_10','INTENSITY_50','INTENSITY_90']

                for line_c in range(0,simp_prop.shape[1]):

                    #Determine object ID
                    obj_id = int(simp_prop[0][line_c])
                    #Determine the forecast hour
                    try:
                        fcst_hr = int(simp_prop[2][line_c])
                    except ValueError:
                        fcst_hr = -999 #data missing

                    if fcst_hr != -999: #data exists                                                                                                                                                                                                                                                                                                                                                                                                                                          
                        if obj_id < 0: #Load obs data
                            simp_prop[7,line_c] = np.round(np.nanpercentile(grid_obs[fcst_hr,simp_bin_obs[fcst_hr,:,:] == np.abs(obj_id)], 10),3)
                            simp_prop[8,line_c] = np.round(np.nanpercentile(grid_obs[fcst_hr,simp_bin_obs[fcst_hr,:,:] == np.abs(obj_id)], 50),3)
                            simp_prop[9,line_c] = np.round(np.percentile(grid_obs[fcst_hr,simp_bin_obs[fcst_hr,:,:] == np.abs(obj_id)], 90),3)
                        else:          #Load model data
                            simp_prop[7,line_c] = np.round(np.nanpercentile(grid_mod[fcst_hr,simp_bin_mod[fcst_hr,:,:] == np.abs(obj_id)], 10),3)
                            simp_prop[8,line_c] = np.round(np.nanpercentile(grid_mod[fcst_hr,simp_bin_mod[fcst_hr,:,:] == np.abs(obj_id)], 50),3)
                            simp_prop[9,line_c] = np.round(np.nanpercentile(grid_mod[fcst_hr,simp_bin_mod[fcst_hr,:,:] == np.abs(obj_id)], 90),3)
                    else: #data missing
                        if obj_id < 0: #Load obs data
                            simp_prop[7,line_c] = np.NaN
                            simp_prop[8,line_c] = np.NaN
                            simp_prop[9,line_c] = np.NaN
                        else:          #Load model data
                            simp_prop[7,line_c] = np.NaN
                            simp_prop[8,line_c] = np.NaN
                            simp_prop[9,line_c] = np.NaN

                #Loop through each line of 'pair_prop' and determine obj. 10,50,90th prctile
                #['OBJECT_ID','TIME_INDEX','AREA','CENTROID_LAT','CENTROID_LON','AXIS_ANG','INTENSITY_10','INTENSITY_50','INTENSITY_90']
                for line_c in range(0,pair_prop.shape[1]):

                    #Determine object ID
                    obj_id = int(pair_prop[0][line_c])
                    #Determine the forecast hour
                    try:
                        fcst_hr = int(pair_prop[2][line_c])
                    except ValueError:
                        fcst_hr = -999 #data missing

                    if fcst_hr != 999: #data exists                                                                                                                                                                                                                                                                                                                                                                                                                                           
                        if obj_id < 0: #Load obs data
                            try:
                                pair_prop[7,line_c] = np.round(np.nanpercentile(grid_obs[fcst_hr,clus_bin_obs[fcst_hr,:,:] == np.abs(obj_id)], 10),3)
                                pair_prop[8,line_c] = np.round(np.nanpercentile(grid_obs[fcst_hr,clus_bin_obs[fcst_hr,:,:] == np.abs(obj_id)], 50),3)
                                pair_prop[9,line_c] = np.round(np.nanpercentile(grid_obs[fcst_hr,clus_bin_obs[fcst_hr,:,:] == np.abs(obj_id)], 90),3)
                            except ValueError:
                                pair_prop[7,line_c] = np.NaN
                                pair_prop[8,line_c] = np.NaN
                                pair_prop[9,line_c] = np.NaN

                        else:          #Load model data
                            try:
                                pair_prop[7,line_c] = np.round(np.nanpercentile(grid_mod[fcst_hr,clus_bin_mod[fcst_hr,:,:] == np.abs(obj_id)], 10),3)
                                pair_prop[8,line_c] = np.round(np.nanpercentile(grid_mod[fcst_hr,clus_bin_mod[fcst_hr,:,:] == np.abs(obj_id)], 50),3)
                                pair_prop[9,line_c] = np.round(np.nanpercentile(grid_mod[fcst_hr,clus_bin_mod[fcst_hr,:,:] == np.abs(obj_id)], 90),3)
                            except ValueError:
                                pair_prop[7,line_c] = np.NaN
                                pair_prop[8,line_c] = np.NaN
                                pair_prop[9,line_c] = np.NaN

                    else: #data missing
                        if obj_id < 0: #Load obs data
                            pair_prop[7,line_c] = np.NaN
                            pair_prop[8,line_c] = np.NaN
                            pair_prop[9,line_c] = np.NaN
                        else:          #Load model data
                            pair_prop[7,line_c] = np.NaN
                            pair_prop[8,line_c] = np.NaN
                            pair_prop[9,line_c] = np.NaN

                #Output only mod or obs in 'bin' matrices, convert to boolean
                if self.ismodel[model] == 0:
                   simp_bin = simp_bin_obs
                   clus_bin = clus_bin_obs
                else:
                   simp_bin = simp_bin_mod
                   clus_bin = clus_bin_mod

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
                lat      = lat[y_beg:y_end,x_beg:x_end]
                lon      = lon[y_beg:y_end,x_beg:x_end]
                try:
                    grid_obs = np.transpose(grid_obs[:,y_beg:y_end,x_beg:x_end],(1, 2, 0))
                except TypeError:
                    pass
                grid_mod = np.transpose(grid_mod[:,y_beg:y_end,x_beg:x_end],(1, 2, 0))
                simp_bin = np.transpose(simp_bin[:,y_beg:y_end,x_beg:x_end],(1, 2, 0))
                clus_bin = np.transpose(clus_bin[:,y_beg:y_end,x_beg:x_end],(1, 2, 0))

                #Remove cluster information not in the lat/lon grid specifications.
                #First find IDs for simple clusters that are not in the domain. Since
                #cluster clusters do not have lat/lon information, they are removed
                #if any simple clusters are removed.
                data_subset = (simp_prop[4,:] >= self.latlon_dims[1]) & (simp_prop[4,:] < self.latlon_dims[3]) & \
                    (simp_prop[5,:] >= self.latlon_dims[0]) & (simp_prop[5,:] < self.latlon_dims[2])
                simp_prop=simp_prop[:,data_subset]

                data_subset = (pair_prop[4,:] >= self.latlon_dims[1]) & (pair_prop[4,:] < self.latlon_dims[3]) & \
                    (pair_prop[5,:] >= self.latlon_dims[0]) & (pair_prop[5,:] < self.latlon_dims[2])
                pair_prop=pair_prop[:,data_subset]

                data_success = np.zeros((clus_bin.shape[2])) #Success
                f.close()
            except (RuntimeError, TypeError, NameError, ValueError, KeyError): #File exists but has no clusters

                print("*************************** ERROR ******************************")
                lat = f.variables['lat'][:]
                lon = f.variables['lon'][:]

                try:
                    grid_obs = f.variables['obs_raw'][:]/25.4
                except KeyError:
                    grid_obs = []

                grid_mod = f.variables['fcst_raw'][:]/25.4

                simp_bin  = np.zeros((grid_mod.shape[0], grid_mod.shape[1], grid_mod.shape[2]),dtype=np.bool)
                clus_bin  = np.zeros((grid_mod.shape[0], grid_mod.shape[1], grid_mod.shape[2]),dtype=np.bool)

                simp_prop = np.nan
                pair_prop = np.nan

                #Model should exist and does
                data_success = np.zeros((clus_bin.shape[2])) #Success
                f.close()
            #END try statement

        else: #If statement, file does not exist

            lat          = np.nan
            lon          = np.nan
            grid_mod     = np.nan
            grid_obs     = np.nan

            simp_bin     = np.nan
            clus_bin     = np.nan

            simp_prop    = np.nan
            pair_prop    = np.nan

            #Model should exist, but doesnt
            data_success = 1 #Failure
        #END if statement to check for file existance

        return(lat, lon, grid_mod, grid_obs, simp_bin, clus_bin, simp_prop, pair_prop, data_success)


    def port_data_FILES(self, curr_date,start_hrs,hour_success,MTDfile_new,simp_prop,pair_prop,data_exist,mem):
        """
        This function creates NPZ track files from one retro model/analysis run.

        Parameters
        ----------
        curr_date : datetime.datetime object
            Start date of data to be loaded.
        start_hrs : float
            number start forecast hour
        hour_success : [int]
            binomial array of successful forecast hours loaded
        MTDfile_new : [str]
            string array of MTD files
        simp_prop : [?]
            simple centroid info
        pair_prop : [?]
            pair centroid info
        data_exist : [?]
            binomial 2-D spatial array of successful data loaded
        mem : [str]
            string arry of ensemble member info
        Returns
        -------
            None
        """
        
        #Check for HRRRV4 TLE only
        HRRR_check  = ['HRRR' in i for i in self.load_data]*1

        #Move the original track text file from the temp directory to the track directory for mtd_biaslookup_HRRRto48h.py
        if (self.mtd_mode == 'Operational' and np.nanmean(HRRR_check) == 1 and self.snow_mask == False) or (self.mtd_mode == 'Both'):
            for model in range(len(self.load_data)):
                os.system('mv '+str(self.grib_path_temp)+'/'+MTDfile_new[model]+'* '+str(self.track_path))

        if self.mtd_mode == 'Retro':

            #Save the simple and paired model/obs track files specifying simp/pair, start/end time, hour acc. interval, and threshold
            if (np.sum(hour_success) > 0) and (self.mtd_mode == 'Retro') and (self.load_qpe[0] in self.load_data[1]):
                np.savez(str(self.track_path)+self.grib_path_des+mem[0]+'_'+'{:04d}'.format(curr_date.year)+'{:02d}'.format(curr_date.month)+ \
                    '{:02d}'.format(curr_date.day)+'{:02d}'.format(curr_date.hour)+'_simp_prop'+'_s'+str(int(start_hrs))+\
                    '_e'+str(int(start_hrs+self.end_fcst_hrs))+'_h'+'{0:.2f}'.format(self.pre_acc)+'_t'+str(self.thresh),simp_prop_k = simp_prop[0],data_exist = data_exist)
                np.savez(str(self.track_path)+self.grib_path_des+mem[0]+'_'+'{:04d}'.format(curr_date.year)+'{:02d}'.format(curr_date.month)+ \
                    '{:02d}'.format(curr_date.day)+'{:02d}'.format(curr_date.hour)+'_pair_prop'+'_s'+str(int(start_hrs))+\
                    '_e'+str(int(start_hrs+self.end_fcst_hrs))+'_h'+'{0:.2f}'.format(self.pre_acc)+'_t'+str(self.thresh),pair_prop_k = pair_prop[0],data_exist = data_exist)

                #Gunzip the files
                output = os.system('gzip '+str(self.track_path)+self.grib_path_des+mem[0]+'_'+'{:04d}'.format(curr_date.year)+'{:02d}'.format(curr_date.month)+ \
                    '{:02d}'.format(curr_date.day)+'{:02d}'.format(curr_date.hour)+'_simp_prop'+'_s'+str(int(start_hrs))+\
                    '_e'+str(int(start_hrs+self.end_fcst_hrs))+'_h'+'{0:.2f}'.format(self.pre_acc)+'_t'+str(self.thresh)+'.npz')
                output = os.system('gzip '+str(self.track_path)+self.grib_path_des+mem[0]+'_'+'{:04d}'.format(curr_date.year)+'{:02d}'.format(curr_date.month)+ \
                    '{:02d}'.format(curr_date.day)+'{:02d}'.format(curr_date.hour)+'_pair_prop'+'_s'+str(int(start_hrs))+\
                    '_e'+str(int(start_hrs+self.end_fcst_hrs))+'_h'+'{0:.2f}'.format(self.pre_acc)+'_t'+str(self.thresh)+'.npz')

    def port_data_FIGS(self, curdate):
        """
        This function ports figures to the proper servers/location at WPC

        Parameters
        ----------
        curr_date : datetime.datetime object
            datetime element for current date

        Returns
        -------
            None
        """
        if self.mtd_mode == 'Operational':
            if self.snow_mask == False:
                if 'HRRRv415min' in ''.join(self.load_data):

                    os.system('scp '+str(self.fig_path)+'/'+str(self.grib_path_des)+'*'+self.domain_sub[1:]+'*.* hpc@wpcrzdm:/home/people/hpc/www/htdocs/HRRRSubHr/images/'+ \
                        self.domain_sub[1:]+'/')
                    os.system('aws s3 cp '+str(self.fig_path)+'/ s3://s3-east-www.wpc.woc.noaa.gov/public/HRRRSubHr/images/'+self.domain_sub[1:]+'/'+ \
                        ' --recursive --exclude "*" --include "*'+str(self.grib_path_des)+'*'+self.domain_sub[1:]+'*.*"')

                    #Clear files older than two days from wpcrzdm to save memory
                    curdate_old  = curdate - datetime.timedelta(days=2)
                    yrmonday_old = '{:04d}'.format(curdate_old.year)+'{:02d}'.format(curdate_old.month)+'{:02d}'.format(curdate_old.day)
                    os.system("ssh hpc@wpcrzdm 'rm /home/people/hpc/www/htdocs/HRRRSubHr/images/"+self.domain_sub[1:]+"/*"+str(self.grib_path_des)+"*"+ \
                        yrmonday_old+"*.*'")
                    os.system('aws s3 rm s3://s3-east-www.wpc.woc.noaa.gov/public/HRRRSubHr/images/'+self.domain_sub[1:]+'/ --recursive --exclude "*" --include "*'+\
                        yrmonday_old+'*.*"')

                else:

                    os.system('scp '+str(self.fig_path)+'/'+str(self.grib_path_des)+'*'+self.domain_sub[1:]+'*.* hpc@wpcrzdm:/home/people/hpc/www/htdocs/verification/mtd/images/')
                    os.system('aws s3 cp '+str(self.fig_path)+'/ s3://s3-east-www.wpc.woc.noaa.gov/public/verification/mtd/images/ --recursive --exclude "*" --include "*'+ \
                        str(self.grib_path_des)+'*'+self.domain_sub[1:]+'*.*"')

                    #Clear files older than two days from wpcrzdm to save memory
                    curdate_old  = curdate - datetime.timedelta(days=2)
                    yrmonday_old = '{:04d}'.format(curdate_old.year)+'{:02d}'.format(curdate_old.month)+'{:02d}'.format(curdate_old.day)
                    os.system("ssh hpc@wpcrzdm 'rm /home/people/hpc/www/htdocs/verification/mtd/images/*"+str(self.grib_path_des)+"*"+yrmonday_old+"*.*'")
                    os.system('aws s3 rm s3://s3-east-www.wpc.woc.noaa.gov/public/verification/mtd/images/ --recursive --exclude "*" --include "*'+\
                        yrmonday_old+'*.*"')

            elif snow_mask == True:

                os.system('scp '+str(self.fig_path)+'/'+str(self.grib_path_des)+'*'+domain_sub[1:]+'*.* hpc@wpcrzdm:/home/people/hpc/www/htdocs/snowbands/images/')
                os.system('aws s3 cp '+str(self.fig_path)+'/ s3://s3-east-www.wpc.woc.noaa.gov/public/snowbands/images/ --recursive --exclude "*" --include "*'+ \
                    str(self.grib_path_des)+'*'+domain_sub[1:]+'*.*"')

                #Clear files older than 5 days from wpcrzdm to save memory
                curdate_old  = curdate - datetime.timedelta(days=5)
                yrmonday_old = '{:04d}'.format(curdate_old.year)+'{:02d}'.format(curdate_old.month)+'{:02d}'.format(curdate_old.day)
                os.system("ssh hpc@wpcrzdm 'rm /home/people/hpc/www/htdocs/snowbands/images/*"+str(self.grib_path_des)+"*"+yrmonday_old+"*.*'")
                os.system('aws s3 rm s3://s3-east-www.wpc.woc.noaa.gov/public/snowbands/images/ --recursive --exclude "*" --include "*'+\
                    yrmonday_old+'*.*"')

    def sweep(self,curr_date,MTDfile_new): 
        """
        This function deleted all unneeded data & figs after mtd code has run

        Parameters
        ----------
        curr_date : datetime.datetime object
            datetime element for current date
        MTDfile_new : [str]
            list of strings of MTD *.nc and *.txt files

        Returns
        -------
            None
        """

        #Delete the source files to save hard drive space
        output=os.system('rm -rf '+str(self.grib_path_temp)+'*')
        print('rm -rf '+str(self.grib_path_temp)+'*')
        output=os.system('rm -rf '+str(self.grib_path_temp)+'_2')
        print('rm -rf '+str(self.grib_path_temp)+'_2')

        #Fix an issue by deleting a folder created in METLoadEnsemble.setupData if no plotting is requested
        if self.plot_allhours == False and self.mtd_mode == 'Retro':
            os.system("rm -rf "+str(self.fig_path)+"/"+'{:04d}'.format(datetime_curdate.year)+'{:02d}'.format(datetime_curdate.month)+ \
                '{:02d}'.format(datetime_curdate.day)+'{:02d}'.format(datetime_curdate.hour))

        #Delete MTD output in specific situations
        if str(self.grib_path_des) == 'MTD_QPF_HRRRTLE_EXT_OPER': #Delete MTD output after bias lookup code runs
            for model in range(len(MTDfile_new)):
                print('rm -rf '+str(self.track_path)+MTDfile_new[model][0:21]+'*')
                os.system('rm -rf '+str(self.track_path)+MTDfile_new[model][0:21]+'*')

    def run_mtd(self):
        """
        MASTER MTD FUNCTION TO BE RUN IN OPERATIONAL OR RETROSPECTIVE MODE. HAS MULTIPLE MODEL/ENSEMBLE 
        OPTIONS. CAN ONLY SPECIFY ONE THRESHOLD AT A TIME. WHEN RUN IN RETROSPECTIVE MODE, CAN ONLY
        USE ONE MODEL/MEMBER AT A TIME. MJE. 201708-20190606,20210609.

        Some specifics, 1) This code considers accumulation intervals with a resolution of one minute. 
                        2) This code considers model initializations with a resolution of one hour.

        Parameters
        ----------
            None

        Returns
        -------
            None
        """

        #Record the current date
        datetime_now = datetime.datetime.now()
        start_hrs = 0.0

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

                #Given the variables, create proper config files
                mtd_conf_filename = 'MTDConfig_USE_SAMP'+'_f'+str(int(self.end_fcst_hrs))+'_p'+APCP_str_end+'_t'+str(self.thresh)
                mtd_conf_fullpath = pathlib.Path(self.config_path, mtd_conf_filename)

                if 'NEWSe' in ''.join(self.load_model) or '15min' in ''.join(self.load_model):
                    config_name = gen_mtdconfig_15m(mtd_conf_fullpath,self.thresh,self.conv_radius,self.min_volume,self.ti_thresh,APCP_str_end)
                else:
                    config_name = gen_mtdconfig(mtd_conf_fullpath,self.thresh,self.conv_radius,self.min_volume,self.ti_thresh,APCP_str_end)

                #Use list comprehension to cat on '.nc' to model list
                load_data_nc = [x + '.nc' for x in self.load_data]

                #Isolate member number if there are multiple members
                if 'mem' in load_data_nc[model]:
                    mem = np.append(mem,'_m'+load_data_nc[model][load_data_nc[model].find('mem')+3:load_data_nc[model].find('mem')+5])
                else:
                    mem = np.append(mem,'')
                    
                #Determine lag from string specification
                mod_lag = int(self.load_data[model][self.load_data[model].find('lag')+3:self.load_data[model].find('lag')+5])

                #NEWSe 60min files only store accumulated precip. Must load previous hour in instances of lag
                if 'NEWSe60min' in self.load_data[model] and mod_lag > 0:
                    hrs_all = np.arange(self.hrs[0]-1,self.hrs[-1]+self.pre_acc,self.pre_acc)
                    data_success = np.concatenate((data_success,np.ones([1, len(self.load_data)])),axis = 0)
                else: 
                    hrs_all = np.arange(self.hrs[0],self.hrs[-1]+self.pre_acc,self.pre_acc)

                while 1: # In operational mode, wait for a while to make sure all data comes in

                    # When snow_mask = True, there is no ST4 analysis mask, so ues the model CSNOW (will utitlize MRMS later)
                    try:
                        data_name_grib_prev = data_name_grib
                    except NameError:
                        data_name_grib_prev = []
                    
                    fcst_hr_count     = 0
                    data_name_grib    = []
                    data_name_nc      = []
                    data_name_nc_prev = []

                    # Loop through the forecast hours
                    for fcst_hr in hrs_all: 
        
                        # Create proper string of last hour loaded to read in MTD file
                        last_fcst_hr_str = '{:02d}'.format(int(fcst_hr+self.offset_fcsthr[model]))
        
                        # Sum through accumulated precipitation interval
                        sum_hr_count = 0
                        data_name_temp = []
                        data_name_temp_part = []
        
                        for sum_hr in np.arange(self.pre_acc-self.acc_int[model],-self.acc_int[model],-self.acc_int[model]):
                        
                            #range(int(self.pre_acc) - int(acc_int[model]),-1,int(-acc_int[model])):
                            
                            #Determine the forecast hour and min to load given current forecast hour, summing location and offset
                            fcst_hr_load = float(fcst_hr) + self.offset_fcsthr[model] - float(round(sum_hr,2))
                            fcst_hr_str  = '{:02d}'.format(int(fcst_hr_load))
                            fcst_min_str = '{:02d}'.format(int(round((fcst_hr_load-int(fcst_hr_load))*60)))
        
                            #Determine the end date for the summation of precipitation accumulation
                            curdate_ahead = curr_date+datetime.timedelta(hours=fcst_hr_load - self.offset_fcsthr[model])
        
                            #Use function to load proper data filename string
                            data_name_temp = np.append(data_name_temp,load_data_str(self.grib_path, self.load_data[model], self.init_yrmondayhr[model], self.acc_int[model], fcst_hr_str, fcst_min_str))
        
                            #Find location of last slash to isolate file name from absolute path
                            for i in range(0,len(data_name_temp[sum_hr_count])):
                                if '/' in data_name_temp[sum_hr_count][i]:
                                    str_search = i + 1
                            data_name_temp_part = np.append(data_name_temp_part, data_name_temp[sum_hr_count][str_search::])
        
                            #Copy the needed files to a temp directory and note successes
                            output=os.system("cp "+data_name_temp[sum_hr_count]+".gz "+ str(self.grib_path_temp))
                            sum_hr_count += 1

                        #END loop through accumulated precipitation
                        
                        #Create the string for the time increment ahead
                        yrmonday_ahead = str(curdate_ahead.year)+'{:02d}'.format(int(curdate_ahead.month))+'{:02d}'.format(int(curdate_ahead.day))
                        hrmin_ahead = '{:02d}'.format(int(curdate_ahead.hour))+'{:02d}'.format(int(curdate_ahead.minute))
        
                        #Gunzip the file 
                        output = os.system("gunzip "+str(self.grib_path_temp)+"/*.gz")
                        
                        #Create archive of last hour grib file name within the summed increment
                        data_name_grib = np.append(data_name_grib, data_name_temp_part)

                        ########## USE MET PCP_COMBINE TO CONVERT DATA TO NETCDF AND SUM PRECIPITATION WHEN APPLICABLE ############
                        self.pcp_combine(model, load_data_nc[model], data_name_grib, yrmonday_ahead, \
                            hrmin_ahead, fcst_hr, APCP_str_beg, APCP_str_end, fcst_hr_count)

                        #For NEWSe/WoF the domain changes so determine domain here
                        if self.mtd_mode == 'Operational':
                            if 'NEWSe' in self.load_model[model] and not 'latlon_dims_keep' in locals():
                                try:
                                    f = Dataset(str(self.grib_path_temp)+"/"+load_data_nc[model], "a", format="NETCDF4")
                                    self.latlon_sub = [[np.nanmin(np.array(f.variables['lon'][:]))-0.5,np.nanmin(np.array(f.variables['lat'][:]))-0.1,\
                                        np.nanmax(np.array(f.variables['lon'][:])),np.nanmax(np.array(f.variables['lat'][:]))+0.5]]
                                    self.domain_sub = ['ALL']
                                    f.close()
                                except:
                                    pass
                        
                        #Regrid to ST4 grid using regrid_data_plane
                        data_success[fcst_hr_count,model] = os.system(str(self.met_path)+"/regrid_data_plane "+str(self.grib_path_temp)+"/"+ \
                            load_data_nc[model]+" "+str(self.grib_path)+"/temp/ensmean_sample_NODELETE "+str(self.grib_path_temp)+"/"+load_data_nc[model]+"2 "+ \
                            "-field 'name=\""+APCP_str_end+"\"; level=\""+APCP_str_end+ \
                            "\";' -name "+APCP_str_end)

                        #Remove original netcdf files
                        output=os.system("rm -rf "+str(self.grib_path_temp)+"/"+load_data_nc[model])
        
                        #Rename netcdf file
                        data_name_nc = np.append(data_name_nc,str(self.grib_path_temp)+"/"+load_data_nc[model][0:-3]+"_f"+fcst_hr_str+fcst_min_str+".nc")
                        output=os.system("mv "+str(self.grib_path_temp)+"/"+load_data_nc[model]+"2 "+data_name_nc[fcst_hr_count])
                            
                        #Construct an array of filenames in the previous model portion of the loop when in retro mode
                        if (self.mtd_mode == 'Retro') and (self.load_model[0] not in self.load_data[model]):
                            data_name_nc_prev = np.append(data_name_nc_prev,str(self.grib_path_temp)+"/"+load_data[0]+"_f"+fcst_hr_str+fcst_min_str+".nc")
                        else:
                            data_name_nc_prev = np.append(data_name_nc_prev,'null')
                        
                        #If snow_mask is True 1) interpolate to common domain (regrid_data plane), 2) mask out non-snow data mask
                        self.apply_snow_mask(model, data_name_nc, data_name_grib, data_name_grib_prev, APCP_str_end, fcst_hr_count)

                        #Apply a regional mask if wanted
                        self.apply_reg_mask(model, data_name_nc, data_name_nc_prev, APCP_str_end, fcst_hr_count)
     
                        fcst_hr_count += 1
                    #END through forecast hour

                    #Remove original grib files
                    for files in data_name_grib:
                        output = os.system('rm -rf '+str(self.grib_path_temp)+'/'+files+'*')
                      
                    #Determine which hours are successfully loaded for each model
                    if (self.mtd_mode == 'Retro'):
                        hour_success = (data_success[:,[i for i in range(0,len(self.load_data)) if self.load_data[i] == self.load_qpe[0]][0]] + \
                            data_success[:,[i for i in range(0,len(self.load_data)) if self.load_data[i] == self.load_model[0]][0]]) == 0
                    elif (self.mtd_mode == 'Operational'): #No OBS, compare model to itself 
                        hour_success = data_success[:,model] == 0
                    
                    #If not all of the data is in during operational mode, then pause for 2 minutes and try again. Repeat for one hour
                    if np.nanmean(data_success[:,model]) != 0 and self.mtd_mode == 'Operational':
                        if (datetime.datetime.now() - datetime_now).seconds > self.ops_check:
                            print('Missing Model Data; Plotting What We Have')
                            break
                        else:
                            print('There is missing Model Data; Pausing')
                            time.sleep(120)
                    elif np.nanmean(data_success[:,model]) == 0 and self.mtd_mode == 'Operational':
                        break
                    else:
                        break 

                #END while check for all model data  

                #Create variable name for default MTD file and renamed MTD file
                if ((self.mtd_mode == 'Retro' and self.load_qpe[0] in self.load_data[model]) or (self.mtd_mode == 'Operational')):
                    #Find first tracked forecast hour for mtd output label
                    hours_MTDlabel    = (((np.argmax(hour_success == 1)+1)*self.pre_acc)+int(self.load_data[model][-2:])+int(self.init_yrmondayhr[model][-2:]))
                    curdate_MTDlabel  = curr_date + datetime.timedelta(hours = ((np.argmax(hour_success == 1)+1)*self.pre_acc))
                    yrmonday_MTDlabel = str(curdate_MTDlabel.year)+'{:02d}'.format(int(curdate_MTDlabel.month))+'{:02d}'.format(int(curdate_MTDlabel.day))

                    if hours_MTDlabel >= 24:
                        hours_MTDlabel = hours_MTDlabel - 24
                    mins_MTDlabel     = '{:02d}'.format(int(((hours_MTDlabel - int(hours_MTDlabel))*60)))
                    hours_MTDlabel    = '{:02d}'.format(int(hours_MTDlabel))

                    MTDfile_old[model] = 'mtd_'+yrmonday_MTDlabel+'_'+hours_MTDlabel+mins_MTDlabel+'00V'

                MTDfile_new[model] = 'mtd_'+self.init_yrmondayhr[model][:-2]+'_h'+self.init_yrmondayhr[model][-2:]+'_f'+last_fcst_hr_str+'_'+ \
                    load_data_nc[model][0:-3]+'_p'+'{0:.2f}'.format(self.pre_acc)+'_t'+str(self.thresh)

                #Remove old MTD output files if they exist
                if os.path.isfile(str(self.grib_path_temp)+'/'+MTDfile_new[model]): 
                    os.remove(str(self.grib_path_temp)+'/'+MTDfile_new[model])    
     
                #In retro mode, if there is no data in the model, then quit this attempt
                if self.mtd_mode == 'Retro' and self.load_data[model] != self.load_qpe[0] and np.nanmean(data_success[:,model]) == 1:
                    print('skipped')
                    break
                
                #Run MTD: 1) if QPE exists compare model to obs, otherwise 2) run mtd in single mode   
                output = []
                if self.mtd_mode == 'Retro' and self.load_qpe[0] in self.load_data[model]:
                    mtd_success = os.system(str(self.met_path)+'/mtd -fcst '+' '.join(data_name_nc_prev[hour_success])+ \
                        ' -obs '+' '.join(data_name_nc[hour_success])+' -config '+str(config_name)+' -outdir '+str(self.grib_path_temp))
                elif self.mtd_mode == 'Operational' and self.load_qpe[0] not in self.load_data[model]: #No QPE, compare model to itself
                    mtd_success = os.system(str(self.met_path)+'/mtd -single '+' '.join(data_name_nc[hour_success])+' -config '+ \
                        str(config_name)+' -outdir '+str(self.grib_path_temp))
                
                #Matrix to gather all netcdf file strings
                data_name_nc_all = np.append(data_name_nc_all,data_name_nc)

                #Rename cluster file and 2d text file (QPE is handled later)
                if ((self.mtd_mode == 'Retro' and self.load_qpe[0] in self.load_data[model]) or (self.mtd_mode == 'Operational')):
                   
                    output = os.system('mv '+str(self.grib_path_temp)+'/'+MTDfile_old[model]+'_obj.nc '+ str(self.grib_path_temp)+'/'+MTDfile_new[model]+ \
                        '_obj.nc ')
                    output = os.system('mv '+str(self.grib_path_temp)+'/'+MTDfile_old[model]+'_2d.txt '+ str(self.grib_path_temp)+'/'+MTDfile_new[model]+ \
                        '_2d.txt ')
                    output = os.system('rm -rf '+str(self.grib_path_temp)+'/'+MTDfile_old[model]+'_3d_ss.txt')
                    output = os.system('rm -rf '+str(self.grib_path_temp)+'/'+MTDfile_old[model]+'_3d_sc.txt')
                    output = os.system('rm -rf '+str(self.grib_path_temp)+'/'+MTDfile_old[model]+'_3d_ps.txt')
                    output = os.system('rm -rf '+str(self.grib_path_temp)+'/'+MTDfile_old[model]+'_3d_pc.txt')

            # END through model

            #Remove the netcdf files
            for files in data_name_nc_all:
                output=os.system("rm -rf "+files)
     
            # ################################################################################
            # #########4 LOAD MTD DATA IN PROPER FORMAT TO BE SAVED###########################
            # ################################################################################

            #Find model index location to load MTD data
            mod_loc = [i for i in range(0,len(MTDfile_new)) if self.load_model[0] in MTDfile_new[i]][0]
            
            #Function to read in MTD data
            for model in range(len(self.load_data)): #Through the 1 model and observation
                
                #Retro mode has one model/one obs only, but the obs are in the model data, so load model but collect obs for retro
                if self.mtd_mode == 'Retro':    
                    (lat_t, lon_t, fcst_p, obs_p, simp_bin_p, clus_bin_p, simp_prop_k, pair_prop_k, data_success_p) = \
                        self.load_data_MTDV90(MTDfile_new[1], model)
                else:
                    (lat_t, lon_t, fcst_p, obs_p, simp_bin_p, clus_bin_p, simp_prop_k, pair_prop_k, data_success_p) = \
                        self.load_data_MTDV90(MTDfile_new[model], model)

                #Determine length of hours, place into matrices properly time-matched
                if self.mtd_mode == 'Retro':
                    hour_success = (data_success[:,[i for i in range(0,len(self.load_data)) if self.load_data[i] == self.load_qpe[0]][0]] + \
                        data_success[:,[i for i in range(0,len(self.load_data)) if self.load_data[i] == self.load_model[0]][0]]) == 0
                elif self.mtd_mode == 'Operational': #No OBS, compare model to itself 
                    hour_success = data_success[:,model] == 0

                hours_true = np.where(hour_success == 1)[0]
                
                if not np.isnan(np.nanmean(simp_prop_k)):                
                    #Add data to total matrices
                    simp_prop[model]       = simp_prop_k
                    pair_prop[model]       = pair_prop_k
                else:
                    simp_prop[model]       = np.full((10,1),np.NaN)
                    pair_prop[model]       = np.full((10,1),np.NaN)

                if np.mean(np.isnan(simp_bin_p)) != 1:
                    simp_bin[:,:,hours_true,model] = simp_bin_p
                if np.mean(np.isnan(clus_bin_p)) != 1:
                    clus_bin[:,:,hours_true,model] = clus_bin_p
              
                del simp_bin_p
                      
                #Create binomial grid where = 1 if model and obs exist
                if np.mean(np.isnan(obs_p)) == 1 or  np.mean(np.isnan(fcst_p)) == 1 or \
                    np.isnan(np.mean(np.isnan(obs_p))) == 1 or  np.isnan(np.mean(np.isnan(fcst_p))) == 1:
                    data_exist = np.zeros((self.lat.shape))
                else:
                    data_exist = (np.isnan(obs_p[:,:,0].data)*1 + np.isnan(fcst_p[:,:,0].data)*1) == 0
                
                #If requested, plot hourly paired mod/obs objects to manually check results by-eye
                if self.plot_allhours == True and model > 0:
                    #If the domain is subset, determine proper coordinates for plotting
                    latmin = np.nanmin(lat_t[fcst_p[:,:,0].data>0])
                    latmax = np.nanmax(lat_t[fcst_p[:,:,0].data>0])
                    lonmin = np.nanmin(lon_t[fcst_p[:,:,0].data>0])
                    lonmax = np.nanmax(lon_t[fcst_p[:,:,0].data>0])

                    mtd_plot_retro(str(self.grib_path_des), str(self.fig_path),[lonmin,latmin,lonmax,latmax],self.pre_acc,hrs_all,self.thresh,curr_date, \
                         data_success,load_data_nc,self.lat,self.lon,fcst_p,obs_p,clus_bin,pair_prop)
                
                del clus_bin_p
                del fcst_p
                del obs_p

            #Create npz files for retro runs with model/analysis
            self.port_data_FILES(curr_date,start_hrs,hour_success,MTDfile_new,simp_prop,pair_prop,data_exist,mem)

            if self.mtd_mode == 'Operational':#If in operational mode, create plots

                #Save data for FF site
                # METLoadEnsemble.port_data_FF(TRACK_PATH,GRIB_PATH_DES,datetime_curdate,start_hrs,end_fcst_hrs,hrs_all,pre_acc,thres,sigma,\
                #     load_model,simp_bin,simp_prop,lat_t,lon_t,snow_mask)

                for subsets in range(0,len(self.domain_sub)): #Loop through domain subset specifications to plot specific regions
                    if self.snow_mask == False:

                        #Plot the smoothed ensemble probabilities with object centroids
                        mtd_plot_all_fcst(str(self.grib_path_des)+self.domain_sub[subsets], str(self.fig_path), self.latlon_sub[subsets], self.pre_acc, \
                            self.hrs, self.thresh, curr_date, data_success, load_data_nc, self.lat, self.lon, simp_bin, simp_prop, self.sigma)
            
                        #Plot the objects for each TLE
                        mtd_plot_tle_fcst(str(self.grib_path_des)+self.domain_sub[subsets], str(self.fig_path), self.latlon_sub[subsets], self.pre_acc, \
                            self.hrs, self.thresh, curr_date, data_success, load_data_nc, self.lat, self.lon, simp_bin, simp_prop, self.sigma)

                    elif self.snow_mask == True:        
                        #Plot the smoothed ensemble probabilities with object centroids
                        mtd_plot_all_snow_fcst(str(self.grib_path_des)+self.domain_sub[subsets], str(self.fig_path), self.latlon_sub[subsets], self.pre_acc, \
                            self.hrs, self.thresh, curr_date, data_success, load_data_nc, self.lat, self.lon, simp_bin, simp_prop)
                    
                    if (self.transfer_to_prod):
                        #Copy over relevant images to proper websites 
                        self.port_data_FIGS(curr_date)
                    

            del simp_prop 
            del pair_prop 
            del simp_bin

            #Delete any files that are no longer needed
            self.sweep(curr_date, MTDfile_new)
