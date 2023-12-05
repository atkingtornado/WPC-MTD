import os
import dataclasses
from typing import ClassVar
import numpy as np
import datetime
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter

from .mtd import WPCMTD
from .utils import read_CONUS_mask

@dataclasses.dataclass
class WPCMTDBiasLookup(WPCMTD):
    """
    This class is responsible for configuring all the necessary components 
    required for providing displacement information for real-time objects by 
    linking real-time objects to retrospective modeled objects and their observations.
    """

    grib_path_ret    : str   # An appended string label for the retro HRRRv4 data to be loaded
    grid_res_nat     : int   # Grid res. (deg) of the native grid. If =[], everything is aggregated spatially
    grid_res_agg     : int   # Grid res. (deg) of the aggregated grid. If =[], everything is aggregated spatially
    end_fcst_hrs_ret : float # Last forecast hour to be considered in retro loading
    update_freq      : int   # Update frequency to regenerate bias look up tables (to save time)

    lon_nat:   ClassVar[float] = None
    lat_nat:   ClassVar[float] = None
    CONUS_mask: ClassVar[int]   = None

    # Perform post-initalization checks and property updates based on initial values
    def __post_init__(self):

        #If either the aggregated or native grid resolution is blank, they are both blank and everything is aggregated throughout the domain
        if self.grid_res_nat == [] or self.grid_res_agg == []:
            self.grid_res_nat = []
            self.grid_res_agg = []

        #If the aggregation search radius is less than the native resolution grid, set them as equal
        if self.grid_res_agg < self.grid_res_nat:
            grid_res_agg = grid_res_nat

    def create_grid(self):
        """
        CREATE GRID TO MAP RETRO BIASES TO OPERATIONAL OBJECTS

        Parameters
        ----------
            None

        Returns
        -------
            None
        """

        #If grid domain is specified create grid/CONUSmask
        if self.grid_res_nat != []:
            [self.lon_nat,self.lat_nat] = np.meshgrid(np.linspace(self.latlon_dims[0],self.latlon_dims[2],int(np.round((self.latlon_dims[2]-self.latlon_dims[0])/self.grid_res_nat,2)+1)), \
                np.linspace(self.latlon_dims[3],self.latlon_dims[1],int(np.round((self.latlon_dims[3]-self.latlon_dims[1])/self.grid_res_nat,2)+1)))
            (self.CONUS_mask,self.lat,self.lon) = read_CONUS_mask(self.grib_path,self.latlon_dims,self.grid_res_nat)
            self.CONUS_mask[np.isnan(self.CONUS_mask)] = 0
            #Interpolate mask to new grid since they are 1 degree off from each other and smooth to capture the slightly offshore regions
            self.CONUS_mask = interpolate.griddata(np.stack((self.lon.flatten(),self.lat.flatten()),axis=1),self.CONUS_mask.flatten(),(self.lon-self.grid_res_nat/2,self.lat+self.grid_res_nat/2),method='nearest')
            self.CONUS_mask = gaussian_filter(self.CONUS_mask,1) >= 0.1
        else: #Create a grid where everything is spatially aggregated to one point
            self.lon_nat = np.array([[self.latlon_dims[0],self.latlon_dims[2]],[self.latlon_dims[0],self.latlon_dims[2]]])
            self.lat_nat = np.array([[self.latlon_dims[3],self.latlon_dims[3]],[self.latlon_dims[1],self.latlon_dims[1]]])
            self.CONUS_mask = np.ones((self.lat_nat.shape))

    #####################INPUT FILES FOR load_pair_prop###############################
    #TRACK_PATH        = path to track data
    #GRIB_PATH_DES     = unique string ID for MET run
    #load_model        = a string list of model data to be loaded
    #list_date         = list of dates (datetime element)
    #start_hrs         = start hour for MTD run (usually zero)
    #end_fcst_hrs      = Last forecast hour to be considered
    #pre_acc           = precipitation accumulation interval
    #thres             = thrshold for precipitation data
    #lat_nat           = latitude native grid
    #lon_nat           = longitude native grid
    #grid_res_agg      = grid resolution to aggregate statistics (e.g. search radius)
    ####################OUTPUT FILES FOR load_pair_prop################################
    #grid_pair         = grid showing model/obs differences of paired object attributes
    #grid_init         = grid showing model/obs differences of initiation object attributes
    #grid_disp         = grid showing model/obs differences of disipation object attributes
    #pair_prop_count   = total sample size of data
    ##data_exist_sum   = total sample size of data by grid point
    ###################################################################################

    def load_pair_prop(self):
        """
        THIS FUNCTION LOADS THE PAIRED OBJECTS CREATED RETROSPECTIVELY USING 
        'mtd_retroruns_save.py.' ALL OF THE DATES ARE LOADED INSIDE OF THIS FUNCTION
        AND THE STATISTICS ARE AGGREGATED SPATIALLY. THERE IS A FILTERING OPTION BY 
        FORECAST HOUR, DIURNAL HOUR, AND SPATIALLY. THIS FUNCTION ALSO CALCULATES 
        DIFFERENCES IN PAIRED OBJECT INITIATION AND DISSIPATION. THIS NEW VERSION 
        ALLOWS FOR SPATIAL AGGREGATION OF DATA ON A DIFFERENT DOMAIN RESOLUTION THAN 
        THE NATIVE DOMAIN.

        Parameters
        ----------
            None

        Returns
        -------
            None
        """

        #Determine the total number of possible forecast hours
        fcst_hrs_all = np.arange(self.start_hrs,self.end_fcst_hrs_ret,self.pre_acc)

        #Calculate native grid resolution
        grid_res_nat = np.unique(np.round(np.diff(self.lon_nat),1))
        
        #Calculate search radius out for aggregate statistics in grid points
        if self.grid_res_agg == []:
            search_radius = 0
        else:
            search_radius = np.floor(self.grid_res_agg / grid_res_nat ) - 1
        
        #Initialize the paired attributes matrices and other temporary variables
        grid_pair       = [[[] for i in range(0,self.lat_nat.shape[1])] for j in range(0,self.lat_nat.shape[0])]         
        grid_init       = [[[] for i in range(0,self.lat_nat.shape[1])] for j in range(0,self.lat_nat.shape[0])]   
        grid_disp       = [[[] for i in range(0,self.lat_nat.shape[1])] for j in range(0,self.lat_nat.shape[0])] 
        xloc_u = []
        yloc_u = []
        pair_prop_count = 0
        data_sum_count  = 0
       
        #Determine if there is one or several datetime elements
        try:
            temps = self.list_date.shape
        except AttributeError:
            self.list_date = [self.list_date]
     
        #Loop through and aggregate data
        for datetime_curdate in self.list_date: #through the dates
            for model in [self.load_model[0]]: #through the members

                #Load paired model/obs track files here
                try:
        
                    #Isolate member number if there are multiple members
                    if 'mem' in model:
                        mem = model[model.find('mem')+3:model.find('mem')+5]
                        filename_load = str(self.track_path)+'/'+str(self.grib_path_ret)+'_m'+mem+'_'+'{:04d}'.format(datetime_curdate.year)+ \
                            '{:02d}'.format(datetime_curdate.month)+'{:02d}'.format(datetime_curdate.day)+'{:02d}'.format(datetime_curdate.hour)+\
                            '_pair_prop'+'_s'+str(int(self.start_hrs))+'_e'+str(int(self.start_hrs+self.end_fcst_hrs_ret))+'_h'+'{0:.2f}'.format(self.pre_acc)+'_t'+str(self.thresh)+'.npz'
                    else:
                        filename_load = str(self.track_path)+'/'+str(self.grib_path_ret)+'_'+'{:04d}'.format(datetime_curdate.year)+ \
                            '{:02d}'.format(datetime_curdate.month)+'{:02d}'.format(datetime_curdate.day)+'{:02d}'.format(datetime_curdate.hour)+\
                            '_pair_prop'+'_s'+str(int(self.start_hrs))+'_e'+str(int(self.start_hrs+self.end_fcst_hrs_ret))+'_h'+'{0:.2f}'.format(self.pre_acc)+'_t'+str(self.thresh)+'.npz'
                        
                    #Unzip the file, load the data, and rezip the original file
                    output = os.system('gunzip '+filename_load+'.gz')
                    data   = np.load(filename_load)
                    output = os.system('gzip '+filename_load)
                        
                    #Initialize grid to sum total number of missing points
                    if data_sum_count == 0:
                        data_exist_sum = np.zeros((data['data_exist'].shape))
                        data_sum_count += 1
                        
                    pair_prop_k     = data['pair_prop_k']
                    data_exist_sum  = data_exist_sum + data['data_exist']
                        
                except IOError:
                    print(filename_load+' Not Found')
                    pair_prop_k = np.nan

                #UNCOMMENT to print out information on object information        
                #for ind in np.unique(pair_prop_k[0]):
                #    print 'Index is '+str(ind)
                #    print pair_prop_k[2][pair_prop_k[0]==ind]
                #    print pair_prop_k[4][pair_prop_k[0]==ind]
                #    print pair_prop_k[5][pair_prop_k[0]==ind]
                    
                ##FIRST LOOP THROUGH pair_prop_k, GATHERING 1) INSTANCES WHERE MOD/OBS MATCHED/MERGED OBJECTS EXIST 
                ##AT THE SAME TIME AND CALCULATE DIFFERENCES AND 2) INSTANCES WHERE MATCHED/MERGED MOD/OBS INITIATE 
                ##OR DISSIPATE AND CALCULATE DIFFERENCES PERTAINING TO THAT.
                    
                if np.mean(np.isnan(pair_prop_k)) < 1 and np.mean(pair_prop_k) == 0: #Model exists, but with no data
                    pair_prop_count += 1
                        
                elif np.mean(np.isnan(pair_prop_k)) < 1 and np.mean(pair_prop_k) != 0: #Model exists with data

                    for obs_num in np.unique(pair_prop_k[0][pair_prop_k[0]<0]): #Each unique "storm" track
                            
                        #Isolate model/obs and find all common time indices to both
                        obs_use  = np.where(pair_prop_k[0] ==  obs_num)[0]
                        mod_use  = np.where(pair_prop_k[0] == -obs_num)[0]
                        time_use = np.union1d(pair_prop_k[2][obs_use],pair_prop_k[2][mod_use])

                        #Append a spurious +1 to the end of 'time_use' so the end time is always properly captured
                        time_use = np.hstack((time_use,time_use[-1]+1))

                        ##Subset times based on user specification
                        #time_use = np.intersect1d(time_use,subset_hours)
                
                        #Binomial variables to determine 1) at start, mod or obs first and 2) at end, mod or obs last
                        first_obs = 0
                        final_obs = 0 
                        first_mod = 0
                        final_mod = 0
                        tloc_first_obs = 0
                        tloc_first_mod = 0
                        tloc_final_obs = len(fcst_hrs_all)-1
                        tloc_final_mod = len(fcst_hrs_all)-1
                               
                        #Loop through all data within each specific paired object
                        for time_val in time_use: #Each element within the "storm" tracks
                                
                            #Find matching obs/model index location
                            obs_ind = np.where((pair_prop_k[1][:] ==  obs_num) & (pair_prop_k[2][:] == time_val))[0]
                            mod_ind = np.where((pair_prop_k[1][:] == -obs_num) & (pair_prop_k[2][:] == time_val))[0]
                                
                            #Check that values is within domain boundaries (using observation data)
                            if not (pair_prop_k[4][obs_ind] < np.nanmin(self.lat_nat) or pair_prop_k[4][obs_ind] > np.nanmax(self.lat_nat) or \
                                pair_prop_k[5][obs_ind] < np.nanmin(self.lon_nat) or pair_prop_k[5][obs_ind] > np.nanmax(self.lon_nat)):

                                #1) FIND ALL INSTANCES OF MOD/OBS INITIATION/DISSIPATION. THIS ALLOWS FOR THE
                                #COMPARISON OF MOD AND OBS AT DIFFERENT TIMES. 
                    
                                #Find first instance of obs
                                if len(obs_ind) > 0 and first_obs == 0:
                                          
                                    #Index location of data to be stored in archived matrix
                                    yloc_first_obs  = int(np.argmax(np.diff([self.lat_nat[:,1]>pair_prop_k[4][obs_ind][0]]))+1)
                                    xloc_first_obs  = int(np.argmax(np.diff([self.lon_nat[1,:]<pair_prop_k[5][obs_ind][0]]))+1)
                                    tloc_first_obs  = int(time_val)
                                        
                                    #Value of data to be stored in archived matrix
                                    lat_first_obs    = pair_prop_k[4][obs_ind][0]
                                    lon_first_obs    = pair_prop_k[5][obs_ind][0]
                                    ang_first_obs    = pair_prop_k[6][obs_ind][0]
                                    int_first_obs_10 = pair_prop_k[7][obs_ind][0]
                                    int_first_obs_50 = pair_prop_k[8][obs_ind][0]
                                    int_first_obs_90 = pair_prop_k[9][obs_ind][0]
                                    area_first_obs   = pair_prop_k[3][obs_ind][0]
                                    fhour_first_obs  = (pair_prop_k[2][obs_ind][0] + 1) * self.pre_acc
                                    #dhour_first_obs  = (((pair_prop_k[2][obs_ind][0] + 1) * self.pre_acc) + datetime_curdate.hour) - \
                                    #    24 * ((((pair_prop_k[2][obs_ind][0] + 1) * self.pre_acc) + datetime_curdate.hour) >= 24) * 1
                                    dhour_first_obs  = (((pair_prop_k[2][obs_ind][0] + 1) * self.pre_acc) + datetime_curdate.hour) - \
                                        (((((pair_prop_k[2][obs_ind][0] + 1) * self.pre_acc) + datetime_curdate.hour) // 24) * 24) #diurnal hour increment

                                        
                                    first_obs = 1
                                    
                                #Find last instance of obs (note we grab the data from the previous index)
                                elif len(obs_ind) == 0 and first_obs == 1 and final_obs == 0: 

                                    #Index location of data to be stored in archived matrix
                                    yloc_final_obs = int(np.argmax(np.diff([self.lat_nat[:,1]>pair_prop_k[4][obs_ind_prev][0]]))+1)
                                    xloc_final_obs = int(np.argmax(np.diff([self.lon_nat[1,:]<pair_prop_k[5][obs_ind_prev][0]]))+1)
                                    tloc_final_obs = int(time_val - 1)
                                        
                                    #Value of data to be stored in archived matrix
                                    lat_final_obs    = pair_prop_k[4][obs_ind_prev][0]
                                    lon_final_obs    = pair_prop_k[5][obs_ind_prev][0]
                                    ang_final_obs    = pair_prop_k[6][obs_ind_prev][0]
                                    int_final_obs_10 = pair_prop_k[7][obs_ind_prev][0]
                                    int_final_obs_50 = pair_prop_k[8][obs_ind_prev][0]
                                    int_final_obs_90 = pair_prop_k[9][obs_ind_prev][0]
                                    area_final_obs   = pair_prop_k[3][obs_ind_prev][0]
                                    fhour_final_obs  = (pair_prop_k[2][obs_ind_prev][0] + 1) * self.pre_acc
                                    #dhour_final_obs  = (((pair_prop_k[2][obs_ind_prev][0] + 1) * self.pre_acc) + datetime_curdate.hour) - \
                                    #    24 * ((((pair_prop_k[2][obs_ind_prev][0] + 1) * self.pre_acc) + datetime_curdate.hour) >= 24) * 1
                                    dhour_final_obs  = (((pair_prop_k[2][obs_ind_prev][0] + 1) * self.pre_acc) + datetime_curdate.hour) - \
                                        (((((pair_prop_k[2][obs_ind_prev][0] + 1) * self.pre_acc) + datetime_curdate.hour) // 24) * 24) #diurnal hour increment                                    
             
                                    final_obs = 1
                                        
                                #Find first instance of mod
                                if len(mod_ind) > 0 and first_mod == 0:
                                        
                                    #Index location of data to be stored in archived matrix
                                    yloc_first_mod = int(np.argmax(np.diff([self.lat_nat[:,1]>pair_prop_k[4][mod_ind][0]]))+1)
                                    xloc_first_mod = int(np.argmax(np.diff([self.lon_nat[1,:]<pair_prop_k[5][mod_ind][0]]))+1)
                                    tloc_first_mod = int(time_val)

                                    #Value of data to be stored in archived matrix
                                    lat_first_mod    = pair_prop_k[4][mod_ind][0]
                                    lon_first_mod    = pair_prop_k[5][mod_ind][0]
                                    ang_first_mod    = pair_prop_k[6][mod_ind][0]
                                    int_first_mod_10 = pair_prop_k[7][mod_ind][0]
                                    int_first_mod_50 = pair_prop_k[8][mod_ind][0]
                                    int_first_mod_90 = pair_prop_k[9][mod_ind][0]
                                    area_first_mod   = pair_prop_k[3][mod_ind][0]
                                    fhour_first_mod  = (pair_prop_k[2][mod_ind][0] + 1) * self.pre_acc
                                    #dhour_first_mod  = (((pair_prop_k[2][mod_ind][0] + 1) * self.pre_acc) + datetime_curdate.hour) - \
                                    #    24 * ((((pair_prop_k[2][mod_ind][0] + 1) * self.pre_acc) + datetime_curdate.hour) >= 24) * 1
                                    dhour_first_mod  = (((pair_prop_k[2][mod_ind][0] + 1) * self.pre_acc) + datetime_curdate.hour) - \
                                        (((((pair_prop_k[2][mod_ind][0] + 1) * self.pre_acc) + datetime_curdate.hour) // 24) * 24) #diurnal hour increment
                                    first_mod = 1
                                        
                                #Find last instance of mod (note we grab the data from the previous index)
                                elif len(mod_ind) == 0 and first_mod == 1 and final_mod == 0: 
                
                                    #Index location of data to be stored in archived matrix
                                    yloc_final_mod = int(np.argmax(np.diff([self.lat_nat[:,1]>pair_prop_k[4][mod_ind_prev][0]]))+1)
                                    xloc_final_mod = int(np.argmax(np.diff([self.lon_nat[1,:]<pair_prop_k[5][mod_ind_prev][0]]))+1)
                                    tloc_final_mod = int(time_val - 1)
                                        
                                    #Value of data to be stored in archived matrix
                                    lat_final_mod    = pair_prop_k[4][mod_ind_prev][0]
                                    lon_final_mod    = pair_prop_k[5][mod_ind_prev][0]
                                    ang_final_mod    = pair_prop_k[6][mod_ind_prev][0]
                                    int_final_mod_10 = pair_prop_k[7][mod_ind_prev][0]
                                    int_final_mod_50 = pair_prop_k[8][mod_ind_prev][0]
                                    int_final_mod_90 = pair_prop_k[9][mod_ind_prev][0]
                                    area_final_mod   = pair_prop_k[3][mod_ind_prev][0]
                                    fhour_final_mod  = (pair_prop_k[2][mod_ind_prev][0] + 1) * self.pre_acc
                                    #dhour_final_mod  = (((pair_prop_k[2][mod_ind_prev][0] + 1) * self.pre_acc) + datetime_curdate.hour) - \
                                    #    24 * ((((pair_prop_k[2][mod_ind_prev][0] + 1) * self.pre_acc) + datetime_curdate.hour) >= 24) * 1
                                    dhour_final_mod  = (((pair_prop_k[2][mod_ind_prev][0] + 1) * self.pre_acc) + datetime_curdate.hour) - \
                                        (((((pair_prop_k[2][mod_ind_prev][0] + 1) * self.pre_acc) + datetime_curdate.hour) // 24) * 24) #diurnal hour increment                                    

                                    final_mod = 1
                                        
                                #2) FIND AND RECORD ALL INSTANCES OF MOD AND OBS AT A FIXED TIME. 
                                #NOTE BOTH MOD AND OBS MUST EXIST FOR THE TIME PERIOD ANALYZED.
                    
                                #Collect Statistics when mod/obs both exist
                                if len(mod_ind) > 0 and len(obs_ind) > 0: 
     
                                    #Find y,x,t location on the archived paired attributes grid
                                    #Note: anything from -95.9999 to -94 is put into -94 longitude; anything from 37.999 to 36 is put into 36 latitude
                                    yloc = int(np.argmax(np.diff([self.lat_nat[:,1]>pair_prop_k[4][obs_ind][0]]))+1)
                                    xloc = int(np.argmax(np.diff([self.lon_nat[1,:]<pair_prop_k[5][obs_ind][0]]))+1)
                                    #tloc = int(time_val)
                                                                
                                    #Latitude model displacement (model south is negative)
                                    if pair_prop_k[4][mod_ind][0] > pair_prop_k[4][obs_ind][0]:
                                        ysign = 1
                                    else:
                                        ysign = -1
                                        
                                    #longitude model displacement (model west is negative)
                                    if pair_prop_k[5][mod_ind][0] > pair_prop_k[5][obs_ind][0]:
                                        xsign = 1
                                    else:
                                        xsign = -1
                                            
                                    #Calculate x and y vector distance separately
                                    #ydiff  = pair_prop_k[4][mod_ind][0] - pair_prop_k[4][obs_ind][0]
                                    #xdiff  = pair_prop_k[5][mod_ind][0] - pair_prop_k[5][obs_ind][0]
                                    ydiff = ysign * (haversine2.distance((pair_prop_k[4][mod_ind][0], pair_prop_k[5][mod_ind][0]), \
                                        (pair_prop_k[4][obs_ind][0], pair_prop_k[5][mod_ind][0])))
                                    xdiff = xsign * (haversine2.distance((pair_prop_k[4][mod_ind][0], pair_prop_k[5][mod_ind][0]), \
                                        (pair_prop_k[4][mod_ind][0], pair_prop_k[5][obs_ind][0])))
                    
                                    #Calculate intensity, area, and angle differences (model - obs)
                                    intdiff_10 = pair_prop_k[7][mod_ind][0] - pair_prop_k[7][obs_ind][0]
                                    intdiff_50 = pair_prop_k[8][mod_ind][0] - pair_prop_k[8][obs_ind][0]
                                    intdiff_90 = pair_prop_k[9][mod_ind][0] - pair_prop_k[9][obs_ind][0]
                                    areadiff   = pair_prop_k[3][mod_ind][0] - pair_prop_k[3][obs_ind][0]
                                    angdiff    = pair_prop_k[6][mod_ind][0] - pair_prop_k[6][obs_ind][0]
                                    fhour      = (pair_prop_k[2][obs_ind][0] + 1) * self.pre_acc
                                    #dhour      = (((pair_prop_k[2][obs_ind][0] + 1) * self.pre_acc) + datetime_curdate.hour) - \
                                    #    24 * ((((pair_prop_k[2][obs_ind][0] + 1) * self.pre_acc) + datetime_curdate.hour) >= 24) * 1 #diurnal hour increment
                                    dhour      = (((pair_prop_k[2][obs_ind][0] + 1) * self.pre_acc) + datetime_curdate.hour) - \
                                        (((((pair_prop_k[2][obs_ind][0] + 1) * self.pre_acc) + datetime_curdate.hour) // 24) * 24) #diurnal hour increment

                                        
                                    #Define search radius on aggregate grid, considering grid domains
                                    xsearch = np.arange(xloc-search_radius,xloc+search_radius+1,1)
                                    xsearch = xsearch[(xsearch >= 0) & (xsearch < self.lon_nat.shape[1])]
                                    xsearch = xsearch.astype(int)
                                    ysearch = np.arange(yloc-search_radius,yloc+search_radius+1,1)
                                    ysearch = ysearch[(ysearch >= 0) & (ysearch < self.lat_nat.shape[0])]
                                    ysearch = ysearch.astype(int)
                                        
                                    #Aggregate statistics to native grid using aggregate grid search radius            
                                    for xloc_l in xsearch: #through xloc points
                                        for yloc_l in ysearch: #through yloc points    
            
                                            #Append data to paired attribute list [ydiff,xdiff,int_diff,area_diff,ang_diff]
                                            if len(grid_pair[yloc_l][xloc_l]) == 0:
                                                grid_pair[yloc_l][xloc_l] = [[ydiff,xdiff,intdiff_10,intdiff_50,intdiff_90,areadiff,angdiff,fhour,dhour]]
                                                xloc_u = np.append(xloc_u,xloc_l)
                                                yloc_u = np.append(yloc_u,yloc_l)
                                            else:
                                                grid_pair[yloc_l][xloc_l] = np.vstack((grid_pair[yloc_l][xloc_l],[ydiff,xdiff,intdiff_10,intdiff_50,intdiff_90,areadiff,angdiff,fhour,dhour]))
                                                xloc_u = np.append(xloc_u,xloc_l)
                                                yloc_u = np.append(yloc_u,yloc_l)
                                    
                                #print('Paired obs mapped to '+str([yloc,xloc,tloc])+ 'with values of '+str([ydiff,xdiff,intdiff,areadiff,angdiff]))
                                #END model/obs check
                                    
                                #Recall previous indices for model and observation
                                obs_ind_prev = obs_ind
                                mod_ind_prev = mod_ind
                                    
                            #END domain check    
                        #END through time_val
                            
                        #3) RECORD ALL INSTANCES OF MOD/OBS INITIATION AND CALCULATE DIFFERENCES.
                        #DISREGARD ALL INSTANCES WHERE OBJ EXISTED AT TIME ZERO.
                
                        if tloc_first_obs != 0 and tloc_first_mod != 0:
                
                            #Latitude model displacement (model south is negative)
                            if lat_first_mod > lat_first_obs:
                                ysign = 1
                            else:
                                ysign = -1
                                
                            #longitude model displacement (model west is negative)
                            if lon_first_mod > lon_first_obs:
                                xsign = 1
                            else:
                                xsign = -1
                                    
                            #Calculate y and x distance with x and y, respectively, constant between mod/obs object !!!!!!!!!!!1START HERE!!!!!!!!!!!!
                            ydiff = ysign * (haversine2.distance((lat_first_mod, lon_first_mod),(lat_first_obs, lon_first_mod)))
                            xdiff = xsign * (haversine2.distance((lat_first_mod, lon_first_mod),(lat_first_mod, lon_first_obs)))
                
                            #Calculate intensity, area, and angle differences (model - obs)
                            intdiff_10 = int_first_mod_10  - int_first_obs_10
                            intdiff_50 = int_first_mod_50  - int_first_obs_50
                            intdiff_90 = int_first_mod_90  - int_first_obs_90
                            areadiff   = area_first_mod    - area_first_obs
                            angdiff    = ang_first_mod     - ang_first_obs
                            timediff   = tloc_first_mod    - tloc_first_obs
                                
                            #Define search radius on aggregate grid, considering grid domains
                            xsearch = np.arange(xloc_first_obs-search_radius,xloc_first_obs+search_radius+1,1)
                            xsearch = xsearch[(xsearch >= 0) & (xsearch < self.lon_nat.shape[1])]
                            xsearch = xsearch.astype(int)
                            ysearch = np.arange(yloc_first_obs-search_radius,yloc_first_obs+search_radius+1,1)
                            ysearch = ysearch[(ysearch >= 0) & (ysearch < self.lat_nat.shape[0])]
                            ysearch = ysearch.astype(int)
      
                            #Aggregate statistics to native grid using aggregate grid search radius            
                            for xloc_l in xsearch: #through xloc points
                                for yloc_l in ysearch: #through yloc points    
        
                                    #Append data to init attribute list mapped to obs location [ydiff,xdiff,intdiff_10,intdiff_50,intdiff_90,areadiff,angdiff,timediff,fhour_first_obs,dhour_first_obs]
                                    if len(grid_init[yloc_l][xloc_l]) == 0:
                                        grid_init[yloc_l][xloc_l] = [[ydiff,xdiff,intdiff_10,intdiff_50,intdiff_90,areadiff,angdiff,timediff,fhour_first_obs,dhour_first_obs]]
                                    else:
                                        grid_init[yloc_l][xloc_l] = np.vstack((grid_init[yloc_l][xloc_l], \
                                            [ydiff,xdiff,intdiff_10,intdiff_50,intdiff_90,areadiff,angdiff,timediff,fhour_first_obs,dhour_first_obs]))
                                    #print grid_init[yloc_l][xloc_l]        
                                    #print('Model Init mapped to '+str([yloc_l,xloc_l])+ 'with values of '+str([ydiff,xdiff,intdiff,areadiff,angdiff,timediff]))
                                            
                        #4) RECORD ALL INSTANCES OF MOD/OBS DISSIPATION AND CALCULATE DIFFERENCES.
                        #DISREGARD ALL INSTANCES WHERE OBJ EXISTED AT FINAL TIME
                            
                        #Final time of obj must be before last time period considered, remove all others
                        if tloc_final_obs != len(fcst_hrs_all)-1 and tloc_final_mod != len(fcst_hrs_all)-1:
            
                            #Latitude model displacement (model south is negative)
                            if lat_final_mod > lat_final_obs:
                                ysign = 1
                            else:
                                ysign = -1
                                
                            #longitude model displacement (model west is negative)
                            if lon_final_mod > lon_final_obs:
                                xsign = 1
                            else:
                                xsign = -1
                                    
                            #Calculate y and x distance with x and y, respectively, constant between mod/obs object
                            ydiff = ysign * (haversine2.distance((lat_final_mod, lon_final_mod),(lat_final_obs, lon_final_mod)))
                            xdiff = xsign * (haversine2.distance((lat_final_mod, lon_final_mod),(lat_final_mod, lon_final_obs)))
                
                            #Calculate intensity, area, and angle differences (model - obs)
                            intdiff_10 = int_final_mod_10 - int_final_obs_10
                            intdiff_50 = int_final_mod_50 - int_final_obs_50
                            intdiff_90 = int_final_mod_90 - int_final_obs_90
                            areadiff   = area_final_mod   - area_final_obs
                            angdiff    = ang_final_mod    - ang_final_obs
                            timediff   = tloc_final_mod   - tloc_final_obs
                                
                            #Define search radius on aggregate grid, considering grid domains
                            xsearch = np.arange(xloc_final_obs-search_radius,xloc_final_obs+search_radius+1,1)
                            xsearch = xsearch[(xsearch >= 0) & (xsearch < self.lon_nat.shape[1])]
                            xsearch = xsearch.astype(int)
                            ysearch = np.arange(yloc_final_obs-search_radius,yloc_final_obs+search_radius+1,1)
                            ysearch = ysearch[(ysearch >= 0) & (ysearch < self.lat_nat.shape[0])]
                            ysearch = ysearch.astype(int)
      
                            #Aggregate statistics to native grid using aggregate grid search radius            
                            for xloc_l in xsearch: #through xloc points
                                for yloc_l in ysearch: #through yloc points    
       
                                    #Append data to disp attribute list mapped to obs location [ydiff,xdiff,intdiff_10,intdiff_50,intdiff_90,areadiff,angdiff,timediff,fhour_final_obs,dhour_final_obs]
                                    if len(grid_disp[yloc_l][xloc_l]) == 0:
                                        grid_disp[yloc_l][xloc_l] = [[ydiff,xdiff,intdiff_10,intdiff_50,intdiff_90,areadiff,angdiff,timediff,fhour_final_obs,dhour_final_obs]]
                                    else:
                                        grid_disp[yloc_l][xloc_l] = np.vstack((grid_disp[yloc_l][xloc_l], \
                                            [ydiff,xdiff,intdiff_10,intdiff_50,intdiff_90,areadiff,angdiff,timediff,fhour_final_obs,dhour_final_obs]))
                                        
                            #print('Model Disp. mapped to '+str([yloc_final_obs,xloc_final_obs,tloc_final_obs])+ 'with values of '+str([ydiff,xdiff,intdiff,areadiff,angdiff,timediff]))
                    
                    #END through track in one run 
                    pair_prop_count += 1   
                #END loop through model exist check        
            #END loop through the models    
        #END loop through the dates
        print(grid_pair, grid_init, grid_disp, pair_prop_count, data_exist_sum)
        return(grid_pair, grid_init, grid_disp, pair_prop_count, data_exist_sum)

    def run_biaslookup(self):
        """
        THIS CODE LOADS REAL TIME MODEL DATA AND RETRO MODEL DATA AND LINKS THEM. SPECIFICALLY THIS CODE:
        1) LOADS THE CURRENT DAY MODEL/ENSEMBLE AND IDENTIFIES ANY REAL-TIME OBJECTS
        2) LOADS A TRAINING PERIOD (OR A PREVIOUSLY GENERATED TRAINING PERIOD) AND GATHERS ALL DIFFERENCE 
        STATISTICS ANDmPROBABILY OF OBSERVATION (POY) EXISTING GIVEN THE MODEL FOR ONE MODEL
        3) APPENDS THE DIFFERENCE STATISTICS AND POY TO THE OPERATIONAL TRACKS IN A NEW VARIABLE CALLED 'pair_diff'
        4) PLOTS THE OPER. TRACKS AND RETRO BIASES (FIGURE DEEPENDS ON WHETHER THE REAL-TIME TRACKING WAS ON A MODEL
        OR TLE). SIMILAR RETRO OBJECTS ARE BASED ON SEASON, LOCATION, AND FORECAST HOUR.
        
        NOTE: THIS CODE LOADS THE RETRO DIFFERENCE STATISTICS (MODEL-ANALY) MAPPED TO THE ANALY GRID.
        FOR CONSISTENCY, THE MAGNITUDE OF THE DIFFERENCE STATISTICS ARE FLIPPED, AND IT IS ASSUMED THAT
        THE DIFFERENCE BETWEEN MAPPING TO THE MODEL GRID AND ANALY GRID IS NEGLIGABLE.
        

        Parameters
        ----------
            None

        Returns
        -------
            None
        """

        #Record the current date
        datetime_now = datetime.datetime.now()
        

        #If grid domain is specified create grid/CONUSmask
        self.create_grid()

        #Append 'nc' to load_model
        load_model_nc = [i+'.nc' for i in self.load_model]

        self.setup_data(self.list_date[-1])

        #Load retro stats when specified, otherwise load previously existing files
        biaslookup_file = str(self.track_path)+'/'+ str(self.grib_path_des)+'_biaslookup_s'+\
            str(int(self.start_hrs))+'_e'+str(int(self.start_hrs+self.end_fcst_hrs_ret))+'_h'+\
            '{0:.2f}'.format(self.pre_acc)+'_t'+str(self.thresh)+'.npz'

        age_hrs = 1e10
        #Determine age of bias look up table file, if age can't be determined, run new bias lookup tables
        # try:
        #     age     = os.stat(biaslookup_file)
        #     age_hrs = (time.time()-age.st_mtime)/3600
        # except OSError: #Rerun bias look
        #     age_hrs = 1e10

        print(biaslookup_file, age_hrs)
        #Rerun bias look up tables when it's time, otherwise load preexisting file
        if age_hrs >= self.update_freq:
            #Load the retro aggregated stats
            self.load_pair_prop()

            # #Load the retro POY existing
            # self.load_pair_POYgPMY()

            # #Save the file
            # np.savez(biaslookup_file,grid_pair = grid_pair, grid_POYgPMY = grid_POYgPMY)

        else:
            #Load the preexisting file
            data          = np.load(biaslookup_file,allow_pickle='True')
            grid_pair     = data['grid_pair']
            grid_POYgPMY  = data['grid_POYgPMY']

