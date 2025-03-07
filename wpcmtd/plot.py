import sys
import subprocess
import string
import gzip
import os
import numpy as np
import datetime
import time
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter
import datetime
import copy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cf
import cartopy.crs as ccrs
import geojsoncontour
import geojson
import xarray as xr
mpl.use('Agg') #So plots can be saved in cron



#Simple definition to load all of the basic map properites. Inputs can be a 
#grid of latitude/longitude coordinates (2-D) or lon/lat minimum and maximum.
#MJE. 20210205-0309.
#Update: Add alpha = 1 feature so map remains visible with gridded data. MJE. 20211207.
#
#####################INPUT FILES FOR plot_map################################
# lat   = latitude data (2-D grid or [min,max])
# lon   = longitude data (2-D grid or [min,max])
# ax    = axis properties
#####################OUTPUT FILES FOR plot_map################################
##################################################################################

def plot_map(lat,lon,ax):

    lat = np.array(lat)
    lon = np.array(lon)

    #fig = plt.figure(figsize=(12,6)) #Create Figure
    #ax  = plt.axes(projection=ccrs.Mercator(),alpha=1) #Create geoaxes to the projection

    #Draw coastlines, state and country boundaries, edge of map.
    ax.coastlines(edgecolor='grey',alpha=1,linewidth=0.5)
    ax.add_feature(cf.STATES,edgecolor='grey',alpha=1,linewidth=0.3)
    ax.add_feature(cf.BORDERS,edgecolor='grey',alpha=1,linewidth=0.3)
    #Determine resolution of gridlines
    if len(lon.shape) == 2:
        gridsize = np.ceil((np.ceil(np.nanmax(lon))-np.floor(np.nanmin(lon))))
    else:
        gridsize = np.ceil((np.ceil(lon[1])-np.floor(lon[0])))

    if gridsize >= 250:
        gridline_res = 40
    elif gridsize < 250 and gridsize >= 200:
        gridline_res = 30
    elif gridsize < 200 and gridsize >= 150:
        gridline_res = 20
    elif gridsize < 150 and gridsize >= 100:
        gridline_res = 10
    elif gridsize < 100:
        gridline_res = 5
    if len(lon.shape) == 2:
        xticks = np.arange(np.nanmin(lon)-np.nanmin(lon)%gridline_res,np.nanmax(lon)+np.nanmax(lon)%gridline_res+gridline_res,gridline_res)
        yticks = np.arange(np.nanmin(lat)-np.nanmin(lat)%gridline_res,np.nanmax(lat)+np.nanmax(lat)%gridline_res+gridline_res,gridline_res)
    else:
        xticks = np.arange(lon[0]-lon[0]%gridline_res,lon[1]+lon[1]%gridline_res+gridline_res,gridline_res)
        yticks = np.arange(lat[0]-lat[0]%gridline_res,lat[1]+lat[1]%gridline_res+gridline_res,gridline_res)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.3,color='black', linestyle='--',xlocs=xticks,ylocs=yticks)
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    if len(lon.shape) == 2:
        ax.set_extent([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], crs=ccrs.PlateCarree())
    else:
        ax.set_extent([lon[0], lon[1], lat[0], lat[1]], crs=ccrs.PlateCarree())

    return()

########################################################################################
#mtd_plot_retro_disp_vect is designed to create ensemble object based plots
#utilitzing bias look up tables. This plot creates displacement vectors 
#whos tip of the arrow head corresponds to the median displacement 
#from the centroid. This plot also plots the centroid intensity and 
#intensity adjustment. MJE. 20210409-20210803.
#
#UPDATE: Adjust code to copy figures to new WOC. MJE. 20210817.
#
#####################INPUT FILES FOR MTDPlotRetroDispVect###############################
#GRIB_PATH_DES   = unique identifier assigned to figure name
#FIG_PATH_s      = path where figures are to be saved
#latlon_dims     = latitude/longitude dimensions for plotting [WLON,SLAT,ELON,NLAT]
#pre_acc         = accumulation interval
#hrs             = array of forecast hours
#raw             = threshold for MET
#curdate         = datetime attribute
#data_success    = record of which data were succesfully loaded
#load_data_nc    = list of model/obs data to be loaded
#lat             = gridded latitude data
#lon             = gridded longitude data
#simp_bin        = gridded simple data
#simp_prop       = simple cluster centroid info of lat,lon,max. Takes the form of [model][attributes]
#pair_diff       = retro bias differences (not averaged) for all events. Takes the form of [model][track]
#sigma           = sigma value for Gaussian smoother (radius of smoothing in grid points)
####################OUTPUT FILES FOR MTDPlotRetroDispVect###############################
#Note: "Simple" means unmerged while "cluster" refers to merged
#"Single" refers to unmatched objects while "pair" refers to matched objects.
#
###################################################################################
def mtd_plot_retro_disp_vect(GRIB_PATH_DES,FIG_PATH_s,latlon_dims,pre_acc,hrs,thres,curdate,data_success,load_data_nc,lat,lon,simp_bin,simp_prop,pair_diff,sigma):

    #latlon_dims   = latlon_sub[subsets]
    #GRIB_PATH_DES = GRIB_PATH_DES+domain_sub[subsets]

    #Set proper time index depending on ensemble
    if 'NEWSe5min' in ''.join(load_data_nc) or 'HRRRv415min' in ''.join(load_data_nc):
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = (simp_prop[model][2] / 4.0) + 0.25
    else:
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = simp_prop[model][2] + 1

    #Find and isolate ST4 data
    isolate_st4=[]
    for i in range(0,len(load_data_nc),1):
        if 'ST4' in load_data_nc[i]:
            isolate_st4 = np.append(isolate_st4,False)
        else:
            isolate_st4 = np.append(isolate_st4,True)
    isolate_st4       = isolate_st4 == 1

    #Currently this function only requires a logical matrix
    simp_bin = simp_bin > 0

    #Add NaNs to missing data in binary matrix. Convert to float
    simp_bin=simp_bin.astype(float)
    NaN_loc = (data_success == 1).astype(float)
    NaN_loc[NaN_loc == 1] = np.NaN

    for j in range(simp_bin.shape[0]):
        for i in range(simp_bin.shape[1]):
            simp_bin[j,i,:,:] = simp_bin[j,i,:,:] + NaN_loc

    #Compute the proper forecast hour strings
    curdate_currn = curdate + datetime.timedelta(hours=0)

    #Establish a smoothing kernel to smooth ensemble probaiblities
    kern=np.hanning(50)
    kern /= kern.sum()

    #Create colorlist/legend names for 90th percentile of object rainfall (describe)
    colorlist_mag  = ["#ffffff", "#02fd02", "#01c501", "#228b22", "#9acd32", "#fdf802", "#f8d826", "#e5bc00", "#fd9500", "#fd0000", "#bc0000"]
    values_mag     = np.round(np.linspace(0,1*10*thres,len(colorlist_mag)),2)
    my_cmap1       = mpl.colors.ListedColormap(colorlist_mag)
    leg_nam_byint1 = ['90th Object Percentile >='+str(x)+"\"" for x in values_mag]

    #Create colorlist/legend names for difference of 90th percentile of object rainfall (dark green, light green, white, yellow, orange, red)
    colorlist_dif  = ["#bc0000", "#fd0000", "#fd9500", "#ede913", "#fdf802", "#ffffff", "#c5ff33", "#7def36", "#01c501", "#228b22", "#144c07"]
    values_dif     = np.round(np.linspace(-0.5,0.5*10*thres,len(colorlist_dif)),2)
    my_cmap2       = mpl.colors.ListedColormap(colorlist_dif)
    leg_nam_byint2 = ['='+str(x)+"\"" for x in values_dif]

    #Create colorlist, white to blue, representing ensemble probabilities
    values_ens     = np.linspace(0,1,11)
    interval       = np.linspace(0.3, 1)
    interval       = np.linspace(0.3, 0.8)
    colors         = plt.cm.Blues(interval)
    my_cmap3       = LinearSegmentedColormap.from_list('name', colors)
    my_cmap3.set_under('white')

    #Given a default model name list, build a list based on user specified models
    leg_nam_bymod_default = ['ST4'  ,'HIRESNAM','HIRESW','HRRR','NSSL-OP','NSSL-ARW','NSSL-NMB','HRAM','NEWSe']
    leg_sym_bymod_default = ['s','d','*','^','<','>','v','o','8']

    if len(load_data_nc) <= 8: #If list is small enough, specific markers for individual models
        leg_nam_bymod  = load_data_nc
        leg_sym_bymod  = leg_sym_bymod_default[0:len(load_data_nc)]
    else:                      #Create markers by model type
        leg_nam_bymod = []
        leg_sym_bymod = ['d']

    #Determine the smoothing radius width
    #NOTE: It is better to create a slightly larger binning radius than 40 km and then smooth slightly
    #since the raw probabilities are very noisy. In this case a sigma of 1.44 (80 km Gaussian filter)
    #is chosen using a binning radius of 1 degree lat/lon.
    smoth_filter = 80
    grid_delta   = 1.
    levels_cntr  = np.array([0.01,0.02,0.05,0.10,0.15,0.20,0.30,0.50,0.75])

    ###############################################################################################################
    ############SECOND TEST PLOT THAT SHOWS MEDIAN DISPLACEMENT OF HISTORICAL TRACKS FOR ENTIRE ENSEMBLE###########
    ###############################################################################################################

    fig_name = []
    hr_count = 0
    for hr_inc in hrs:

        #Determine time stamp for hour
        curdate_curr = curdate_currn + datetime.timedelta(hours=hr_inc)

        #Determine hour matrix that isolates models existing for that hour
        isolate_hr = np.array(isolate_st4 == 1) & np.array(data_success[hr_count,:] == 0)

        #Determine hourly object probabilities
        ens_mean = np.nanmean(simp_bin[:,:,hr_count,isolate_hr == 1], axis = 2)
        ens_mean[ens_mean > 0.95] = 0.95

        #Load default map for plotting

        fig = plt.figure(figsize=(12,6)) #Create Figure
        ax  = plt.axes(projection=ccrs.Mercator(),alpha=1) #Create geoaxes to the projection
        props = dict(boxstyle='round', facecolor='wheat', alpha=1)

        #Plot contour filled probably of being in an ensemble object
        kwargs = {"transform_first": True}
        cs = plt.contourf(lon, lat, gaussian_filter(ens_mean*100, sigma), levels=values_ens*100, extend='min', cmap=my_cmap3, antialiased=False, \
            vmin=np.nanmin(values_ens)*100, vmax=np.nanmax(values_ens)*100, transform=ccrs.PlateCarree(),**kwargs)
        cb = plt.colorbar(cs,ax=ax)

        for model in range(len(load_data_nc)): #through each model

            #Specify marker properly
            if len(load_data_nc) == len(leg_nam_bymod):
                marker = leg_sym_bymod[model]
            elif len(leg_sym_bymod) == 1:
                marker = leg_sym_bymod[0]
            else:
                for bymod in range(0,len(leg_nam_bymod)):
                    if leg_nam_bymod[bymod] in load_data_nc[model]:
                        marker=leg_sym_bymod[bymod]

            #Find modeled/observed objects in metadata
            if np.mean(np.isnan(simp_prop[model])) != 1:
                #Isolate/identify specific hour object attributes
                hr_loc = simp_prop[model][2] == hr_inc

                #If data exists for that hour, isolate and plot
                if len(hr_loc) > 0:
                    elements = np.unique(simp_prop[model][0][hr_loc])
                    elements = np.unique(simp_prop[model][0])

                    #Separate model and observations and set plotting properties
                    if isolate_st4[model] == 0:
                        elements   = elements[elements < 0]
                        linecolor  = 'b'
                        linewidth  = 3
                        markersize = 60
                        zorder     = 3
                    else:
                        elements = elements[elements > 0]
                        linecolor  = 'k'
                        linewidth  = 2
                        markersize = 30
                        zorder     = 2

                    for tot_obj in elements: #Through all unique tracks

                        #Continue to next iteration if track does not exist at a given hour
                        if np.nansum(np.asarray(simp_prop[model][0] == tot_obj) & np.asarray(simp_prop[model][2] == hr_inc)) == 0:
                            continue
                        #Continue to next iteration if track does not have more than 5 increments total
                        if np.nansum(simp_prop[model][0] == tot_obj) <= 5:
                            continue

                        #Tracks for instantaneous time and total time
                        ind_snap = np.asarray(simp_prop[model][0] == tot_obj) & np.asarray(simp_prop[model][2] == hr_inc)
                        ind_all  = np.asarray(simp_prop[model][0] == tot_obj)

                        #Extract/mean the retro biases specific to the operational tracks
                        latp       = np.array([simp_prop[model][4][ind_snap][0]])
                        lonp       = np.array([simp_prop[model][5][ind_snap][0]])
                        intp       = np.array([simp_prop[model][9][ind_snap][0]])
                        retro_temp = pair_diff[model][np.where(ind_snap)[0][0]]

                        if retro_temp != [] and retro_temp.shape[0] > 10: #if there is retro data and with sufficient sample size

                                #Convert retro_temp from km to lat/lon temp and extract needed data
                                retro_temp[:,0] = retro_temp[:,0]/110.574
                                retro_temp[:,1] = retro_temp[:,1]/(111.320*math.cos(latp*(math.pi/180)))

                                ydiff = np.array([np.nanmedian(retro_temp[:,0])])
                                xdiff = np.array([np.nanmedian(retro_temp[:,1])])
                                idiff = np.array([np.nanmedian(retro_temp[:,4])])
                                POYgM = np.array([np.nanmedian(retro_temp[:,9])])
                               
                        else:
                            ydiff = np.array([0.])
                            xdiff = np.array([0.])
                            idiff = np.array([0.])
                            POYgM = np.array([0.])

                        #Plot the instantanous track location
                        plt.scatter(lonp, latp, c = intp - np.nanmean(np.diff(values_mag)), marker=marker, vmin=np.nanmin(values_mag), \
                            vmax=np.nanmax(values_mag), s=markersize, linewidth=0.25, cmap=my_cmap1, zorder=zorder+1, alpha=1, transform=ccrs.PlateCarree())

                        #Plot the entire track centroid
                        plt.plot(simp_prop[model][5][ind_all], simp_prop[model][4][ind_all], color=linecolor, linewidth=1, zorder=zorder, \
                            transform=ccrs.PlateCarree())

                        #Plot the median retro displacement bias (intp+idiff because idiff is the negative of model-obs)
                        plt.arrow(lonp[0], latp[0], xdiff[0], ydiff[0],facecolor=colorlist_mag[np.argmax(values_mag>=intp[0]+idiff[0])-1], 
                            width=0, head_width=0.6, head_length=0.2, linewidth=0.4, length_includes_head=True, zorder=zorder+1, transform=ccrs.PlateCarree())
                    #END through each track
                #END if data exists for a particular hour
            #END if track exists
        #END through the models

        #Create text at top/bottom portion of map
        ax.text(0.5, 0.03, str(int(np.sum(isolate_hr == 1)))+' Members Found',
            transform=ax.transAxes, fontsize=12,verticalalignment='bottom', horizontalalignment='center', bbox=props)
        ax.text(0.5, 0.97, "Valid Time: "+curdate_curr.strftime('%Y%m%d%H%M')+" - "+'{0:.2f}'.format(pre_acc)+' Hour Accumulation ',
            transform=ax.transAxes, fontsize=12,verticalalignment='top', horizontalalignment='center', bbox=props)

        #Pass line instances to a legend to create centroid maximum legend by color
        line_label=[]
        line_str=[];
        for x in range(len(colorlist_mag)):
            line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color=colorlist_mag[x], linestyle='None', marker='o', markersize=7)))
            line_str   = np.hstack((line_str, leg_nam_byint1[x]))
        first_legend=plt.legend(line_label, line_str, fontsize=6, numpoints=1, loc=3, framealpha=1)
        ax2 = plt.gca().add_artist(first_legend)

        #Pass line instances to a legend denoting member-type by markersize
        line_label=[]
        line_str=[];
        if len(leg_nam_bymod) > 0: #Only plot name legend if > 1 unique name exists
            for x in range(len(leg_sym_bymod)):
                line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color='k', linestyle='None', marker=leg_sym_bymod[x], markersize=7)))
                line_str   = np.hstack((line_str, leg_nam_bymod[x]))
            plt.legend(line_label, line_str, fontsize=6, numpoints=1, loc=4, framealpha=1)

        plot_map([latlon_dims[1],latlon_dims[3]],[latlon_dims[0],latlon_dims[2]],ax)
        plt.show()
        plt.savefig(FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot3_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+ \
            '_t'+str(thres)+'_f'+'{:02d}'.format(int(hr_count)+1)+'.png', bbox_inches='tight',dpi=200)
        plt.close()

        # os.system('scp '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot3_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+ \
        #     '_t'+str(thres)+'_f'+'{:02d}'.format(int(hr_count)+1)+'.png hpc@wpcrzdm:/home/people/hpc/www/htdocs/verification/mtd_exp/images')
        # os.system('aws s3 cp '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot3_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+ \
        #     '_t'+str(thres)+'_f'+'{:02d}'.format(int(hr_count)+1)+'.png s3://s3-east-www.wpc.woc.noaa.gov/public/verification/mtd_exp/images/'+ \
        #     GRIB_PATH_DES+'_retroplot3_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'_f'+\
        #     '{:02d}'.format(int(hr_count)+1)+'.png')

        #Update figure names to create animated figure
        fig_name=np.append(fig_name,FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot3_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+ \
            '_t'+str(thres)+'_f'+'{:02d}'.format(int(hr_count)+1)+'.png')

        hr_count += 1

    #END through the hours

    #Create an animation from the figures previously saved and create a non date specific file (website thumbnail)
    os.system('convert -delay 100 '+' '.join(fig_name)+' -loop 0 '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot3_'+ \
        curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'.gif')
    os.system('cp '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot3_'+curdate_currn.strftime('%Y%m%d%H')+\
        '_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'.gif '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot3_'+\
        'p'+'{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'.gif')
    # os.system('scp '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot3_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'.gif '+\
    #     'hpc@wpcrzdm:/home/people/hpc/www/htdocs/verification/mtd_exp/images')
    # os.system('aws s3 cp '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot3_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'.gif'+\
    #     ' s3://s3-east-www.wpc.woc.noaa.gov/public/verification/mtd_exp/images/'+GRIB_PATH_DES+'_retroplot3_'+\
    #     'p'+'{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'.gif')

    #Convert data from fraction of an hour back to increments
    if 'NEWSe5min' in ''.join(load_data_nc) or 'HRRRv415min' in ''.join(load_data_nc):
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = (simp_prop[model][2] - 0.25) * 4
    else:
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = simp_prop[model][2] - 1

    # #Clear files older than 3 days from wpcrzdm to save memory
    # curdate_old  = curdate - datetime.timedelta(days=3)
    # yrmonday_old = '{:04d}'.format(curdate_old.year)+'{:02d}'.format(curdate_old.month)+'{:02d}'.format(curdate_old.day)
    # os.system("ssh hpc@wpcrzdm 'rm /home/people/hpc/www/htdocs/verification/mtd_exp/images/*"+GRIB_PATH_DES+"*"+yrmonday_old+"*png'")
    # print("ssh hpc@wpcrzdm 'rm /home/people/hpc/www/htdocs/verification/mtd_exp/images/*"+GRIB_PATH_DES+"*"+yrmonday_old+"*png'")
    # os.system('aws s3 rm s3://s3-east-www.wpc.woc.noaa.gov/public/verification/mtd_exp/images/ --recursive --exclude "*" --include "*'+\
    #     yrmonday_old+'*.*"')

    return None

########################################################################################
#mtd_plot_retro_prob is designed to create ensemble object based plots
#utilitzing bias look up tables. This plot creates a probability field
#of where the observed object falls within 40 km of a point relative to the
#operational model object. MJE. 20210409-20210803.
#
#UPDATE: Adjust code to copy figures to new WOC. MJE. 20210817.
#UPDATE: Add utility that only plots tracks with retro stats (e.g. 
#tracks that at one point are near CONUS. MJE. 20210929.
#UPDATE: Fixed minor bug with mapping retro biases to oper. grid. MJE. 20211203.
#UPDATE: Fixed isssue allowing plots when there are temporal gaps in object attributes. MJE. 20220511.
#UPDATE: Code now archives HPOD probabilities. MJE. 20220520.
#UPDATE: Added new retroplot of CONUS probabilities. MJE. 20220525.
#UPDATE: Fix specific error when the instantaneous track index (ind_snap) 
#creates a dynamic domain, but does not exist due to gap in track, use previous ind_snap. MJE. 20220718.
#
#####################INPUT FILES FOR MTDPlotRetroProb###############################
#GRIB_PATH_DES   = unique identifier assigned to figure name
#FIG_PATH_s      = path where figures are to be saved
#latlon_dims     = latitude/longitude dimensions for plotting [WLON,SLAT,ELON,NLAT]
#pre_acc         = accumulation interval
#hrs             = array of forecast hours
#raw             = threshold for MET
#curdate         = datetime attribute
#data_success    = record of which data were succesfully loaded
#load_data_nc    = list of model/obs data to be loaded
#lat             = gridded latitude data
#lon             = gridded longitude data
#simp_bin        = gridded simple data
#simp_prop       = simple cluster centroid info of lat,lon,max. Takes the form of [model][attributes]
#pair_diff       = retro bias differences (not averaged) for all events. Takes the form of [model][track]
####################OUTPUT FILES FOR MTDPlotRetroProb###############################
#Note: "Simple" means unmerged while "cluster" refers to merged
#"Single" refers to unmatched objects while "pair" refers to matched objects.
#
###################################################################################
def mtd_plot_retro_prob(GRIB_PATH_DES,FIG_PATH_s,latlon_dims,pre_acc,hrs,thres,curdate,data_success,load_data_nc,lat,lon,simp_bin,simp_prop,pair_diff):

    #latlon_dims   = latlon_sub[subsets]
    #GRIB_PATH_DES = GRIB_PATH_DES+domain_sub[subsets]

    #######################Variables that can be adapted as inputs to this function, if needed########################
    #Determine the smoothing radius width
    #NOTE: It is better to create a slightly larger binning radius than 40 km and then smooth slightly
    #since the raw probabilities are very noisy. In this case a sigma of 1.44 (80 km Gaussian filter)
    #is chosen using a binning radius of 1 degree lat/lon.
    smoth_filter = 80
    grid_delta   = 1.
    levels_cntr  = np.array([0.01,0.02,0.05,0.10,0.15,0.20,0.30,0.50,0.75])
    min_elements = 8
    #################################################################################################################

    #Set proper time index depending on ensemble
    if 'NEWSe5min' in ''.join(load_data_nc) or 'HRRRv415min' in ''.join(load_data_nc):
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = (simp_prop[model][2] / 4.0) + 0.25
    else:
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = simp_prop[model][2] + 1

    #Find and isolate ST4 data
    isolate_st4=[]
    for i in range(0,len(load_data_nc),1):
        if 'ST4' in load_data_nc[i]:
            isolate_st4 = np.append(isolate_st4,False)
        else:
            isolate_st4 = np.append(isolate_st4,True)
    isolate_st4       = isolate_st4 == 1

    #Currently this function only requires a logical matrix
    simp_bin = simp_bin > 0

    #Add NaNs to missing data in binary matrix. Convert to float
    simp_bin=simp_bin.astype(float)
    NaN_loc = (data_success == 1).astype(float)
    NaN_loc[NaN_loc == 1] = np.NaN

    for j in range(simp_bin.shape[0]):
        for i in range(simp_bin.shape[1]):
            simp_bin[j,i,:,:] = simp_bin[j,i,:,:] + NaN_loc

    #Compute the proper forecast hour strings
    curdate_currn = curdate + datetime.timedelta(hours=0)

    #Establish a smoothing kernel to smooth ensemble probaiblities
    kern=np.hanning(50)
    kern /= kern.sum()

    #Create colorlist/legend names for 90th percentile of object rainfall (describe)
    colorlist_mag  = ["#ffffff", "#02fd02", "#01c501", "#228b22", "#9acd32", "#fdf802", "#f8d826", "#e5bc00", "#fd9500", "#fd0000", "#bc0000"]
    values_mag     = np.round(np.linspace(0,1*10*thres,len(colorlist_mag)),2)
    my_cmap1       = mpl.colors.ListedColormap(colorlist_mag)
    leg_nam_byint1 = ['90th Object Percentile >='+str(x)+"\"" for x in values_mag]

    #Create colorlist/legend names for difference of 90th percentile of object rainfall (dark green, light green, white, yellow, orange, red)
    colorlist_dif  = ["#bc0000", "#fd0000", "#fd9500", "#ede913", "#fdf802", "#ffffff", "#c5ff33", "#7def36", "#01c501", "#228b22", "#144c07"]
    values_dif     = np.round(np.linspace(-0.5,0.5*10*thres,len(colorlist_dif)),2)
    my_cmap2       = mpl.colors.ListedColormap(colorlist_dif)
    leg_nam_byint2 = ['='+str(x)+"\"" for x in values_dif]

    #Create colorlist, white to blue, representing ensemble probabilities
    values_ens     = np.linspace(0,1,11)
    interval       = np.linspace(0.3, 1)
    interval       = np.linspace(0.3, 0.8)
    colors         = plt.cm.Blues(interval)
    my_cmap3       = LinearSegmentedColormap.from_list('name', colors)
    my_cmap3.set_under('white')

    #Given a default model name list, build a list based on user specified models
    leg_nam_bymod_default = ['ST4'  ,'HIRESNAM','HIRESW','HRRR','NSSL-OP','NSSL-ARW','NSSL-NMB','HRAM','NEWSe']
    leg_sym_bymod_default = ['s','d','*','^','<','>','v','o','8']

    if len(load_data_nc) <= 8: #If list is small enough, specific markers for individual models
        leg_nam_bymod  = load_data_nc
        leg_sym_bymod  = leg_sym_bymod_default[0:len(load_data_nc)]
    else:                      #Create markers by model type
        leg_nam_bymod = []
        leg_sym_bymod = ['d']

    #############################################################################################
    ############FIRST TEST PLOT THAT SHOWS CONTOUR CLOUD OF HISTORICAL TRACKS BY MODEL###########
    #############################################################################################
   
    #Init a grid of lat, lon, and probs[lat,lon,hrs,model] that contains the total object maximum probabilities
    [lon_prob,lat_prob] = np.meshgrid(np.linspace(latlon_dims[0],latlon_dims[2], \
        int(np.round((latlon_dims[2]-latlon_dims[0])/1,2)+1)),np.linspace(latlon_dims[3], \
        latlon_dims[1],int(np.round((latlon_dims[3]-latlon_dims[1])/1,2)+1)))
    grid_prob   = np.zeros((lon_prob.shape[0],lon_prob.shape[1],len(hrs),len(load_data_nc)))

    for model in range(len(load_data_nc)): #through each model

        #Specify marker properly
        if len(load_data_nc) == len(leg_nam_bymod):
            marker = leg_sym_bymod[model]
        elif len(leg_sym_bymod) == 1:
            marker = leg_sym_bymod[0]
        else:
            for bymod in range(0,len(leg_nam_bymod)):
                if leg_nam_bymod[bymod] in load_data_nc[model]:
                    marker=leg_sym_bymod[bymod]

        #Find modeled/observed objects in metadata
        if np.mean(np.isnan(simp_prop[model])) != 1:

            #If data exists for that hour, isolate and plot
            if len(simp_prop[model][0]) > 0:
                elements = np.unique(simp_prop[model][0])

                #Separate model and observations and set plotting properties
                if isolate_st4[model] == 0:
                    elements   = elements[elements < 0]
                    linecolor  = 'b'
                    linewidth  = 3
                    markersize = 60
                    zorder     = 3
                else:
                    elements = elements[elements > 0]
                    linecolor  = 'k'
                    linewidth  = 2
                    markersize = 30
                    zorder     = 2

                #Initialize temporaray probability grid to save [lat,lon,elements,hrs]
                grid_prob_full = np.zeros((lon_prob.shape[0],lon_prob.shape[1],len(elements),len(hrs)))
            
                obj_count = 0
                for tot_obj in elements: #Through all unique tracks

                    hr_count = 0
                    for hr_inc in range(len(hrs)):

                        #Continue to next iteration if track does not have more than 'min_elements' increments total
                        if np.nansum(simp_prop[model][0] == tot_obj) <= min_elements:
                            continue

                        #Continue to next iteration if obj is outside the start/end hours of obj lifespan, while accounting for gaps
                        obj_lifespan_beg = np.nanmin(simp_prop[model][2][simp_prop[model][0] == tot_obj])
                        obj_lifespan_end = np.nanmax(simp_prop[model][2][simp_prop[model][0] == tot_obj])
                        if hrs[hr_inc] < obj_lifespan_beg or hrs[hr_inc] > obj_lifespan_end:
                            continue

                        #Determine time stamp for hour
                        curdate_curr = curdate_currn + datetime.timedelta(hours=hrs[hr_inc])

                        #Determine hourly object probabilities
                        ens_mean = (simp_bin[:,:,hr_inc,model] > 0) * 1.

                        #Tracks for instantaneous time and total time
                        ind_snap = np.asarray(simp_prop[model][0] == tot_obj) & np.asarray(simp_prop[model][2] == hrs[hr_inc])
                        ind_all  = np.asarray(simp_prop[model][0] == tot_obj)

                        #Continue to next iteration if there is no object probabilities at any point in the track
                        if np.isnan(np.nanmean(pair_diff[model][np.where(ind_all)[0][0]])):
                            continue

                        #Define plotted domain to feature following if object is large, otherwise static domain
                        if np.nanmax(simp_prop[model][5][ind_all])-np.nanmin(simp_prop[model][5][ind_all]) >= 25 or \
                            np.nanmax(simp_prop[model][4][ind_all])-np.nanmin(simp_prop[model][4][ind_all]) >= 25:
                            try:     #Apply dynamic domain
                                latlon_dims_s  = np.array([np.floor(np.nanmin(simp_prop[model][5][ind_snap]))-12, \
                                    np.floor(np.nanmin(simp_prop[model][4][ind_snap]))-7.5,np.ceil(np.nanmax(simp_prop[model][5][ind_snap]))+12,\
                                    np.ceil(np.nanmax(simp_prop[model][4][ind_snap]))+7.5])
                            except: #If gap in tracks then use centroid just before gap
                                latlon_dims_s  = np.array([np.floor(np.nanmin(simp_prop[model][5][ind_snap_prev]))-12, \
                                    np.floor(np.nanmin(simp_prop[model][4][ind_snap_prev]))-7.5,np.ceil(np.nanmax(simp_prop[model][5][ind_snap_prev]))+12,\
                                    np.ceil(np.nanmax(simp_prop[model][4][ind_snap_prev]))+7.5])
                        else:
                            latlon_dims_s  = np.array([np.floor(np.nanmin(simp_prop[model][5][ind_all]))-10, \
                                np.floor(np.nanmin(simp_prop[model][4][ind_all]))-5,np.ceil(np.nanmax(simp_prop[model][5][ind_all]))+10,\
                                np.ceil(np.nanmax(simp_prop[model][4][ind_all]))+5])

                        #Create a grid to calculate the retro probabilities on a 1 degree lat/lon domain
                        [lon_nat,lat_nat] = np.meshgrid(np.linspace(latlon_dims_s[0],latlon_dims_s[2], \
                            int(np.round((latlon_dims_s[2]-latlon_dims_s[0])/1,2)+1)),np.linspace(latlon_dims_s[3], \
                            latlon_dims_s[1],int(np.round((latlon_dims_s[3]-latlon_dims_s[1])/1,2)+1)))

                        #Create a grid to calculate the retro probabilities at approximately 80 km (KEEP BUT COMMENT OUT)
                        #[lon_nat,lat_nat] = np.meshgrid(np.linspace(latlon_dims_s[0],latlon_dims_s[2], \
                        #    int(np.round((latlon_dims_s[2]-latlon_dims_s[0])/(9/12),2)+1)),np.linspace(latlon_dims_s[3], \
                        #    latlon_dims_s[1],int(np.round((latlon_dims_s[3]-latlon_dims_s[1])/(9/12),2)+1)))

                        #Extract/mean the retro biases specific to the operational tracks
                        try:
                            latp        = np.array([simp_prop[model][4][ind_snap][0]])
                            lonp        = np.array([simp_prop[model][5][ind_snap][0]])
                            intp        = np.array([simp_prop[model][9][ind_snap][0]])
                            retro_temp  = pair_diff[model][np.where(ind_snap)[0][0]]
                        except IndexError:
                            latp        = []
                            lonp        = []
                            intp        = []
                            retro_temp  = []
                        retro_count = np.zeros(lat_nat.shape)

                        if retro_temp != [] and retro_temp.shape[0] > 4: #if there is retro data and with sufficient sample size

                                #Convert retro_temp from km to lat/lon and add to original point to get displaced point
                                latpts = latp + retro_temp[:,0]/110.574
                                lonpts = lonp + retro_temp[:,1]/(111.320*math.cos(latp*(math.pi/180)))

                                #Create gridded frequency count of retro tracks normalized by total count
                                for j in range(lat_nat.shape[0]-1):     #through latitude
                                    for i in range(lat_nat.shape[1]-1): #through longitude
                                        retro_count[j,i] = np.nansum((latpts >= lat_nat[j+1,i]) & (latpts < lat_nat[j,i]) \
                                            & (lonpts >= lon_nat[j,i]) & (lonpts < lon_nat[j,i+1]))

                                retro_count = retro_count / len(latpts)
                                idiff = np.array([np.nanmedian(retro_temp[:,4])])
                                POYgM = np.array([np.nanmedian(retro_temp[:,9])])
                        else:
                            idiff = np.array([])
                            POYgM = np.array([])
 
                        #Load default map for plotting

                        fig = plt.figure(figsize=(12,6)) #Create Figure
                        ax  = plt.axes(projection=ccrs.Mercator(),alpha=1) #Create geoaxes to the projection
                        props = dict(boxstyle='round', facecolor='wheat', alpha=1)

                        #Plot contour filled probably of being in an ensemble object
                        kwargs = {"transform_first": True}
                        cs = plt.contourf(lon, lat, ens_mean*100, levels=values_ens*100, extend='min', cmap=my_cmap3, antialiased=False, \
                            vmin=np.nanmin(values_ens)*100, vmax=np.nanmax(values_ens)*100, transform=ccrs.PlateCarree(),**kwargs)

                        #Plot the instantanous track location and track length 
                        plt.scatter(lonp, latp, c = intp, marker=marker, vmin=np.nanmin(values_mag), vmax=np.nanmax(values_mag), s=markersize, \
                            linewidth=0.25, cmap=my_cmap1, zorder=zorder+1, alpha=0.8, transform=ccrs.PlateCarree())

                        #Plot POYgM text next to track
                        if len(POYgM > 0):
                            plt.text(lonp-2,latp+2,'P = '+str(int(POYgM[0])),fontsize=10,bbox=dict(fc='white', ec='black'),ha='center',\
                                va='center',transform=ccrs.PlateCarree())

                        #Plot the entire track centroid
                        plt.plot(simp_prop[model][5][ind_all], simp_prop[model][4][ind_all], color=linecolor, linewidth=linewidth, zorder=zorder, \
                            transform=ccrs.PlateCarree())

                        #Plot the median retro displacement bias and a scatter of all the retro biases
                        grid_prob_part = gaussian_filter(retro_count,((smoth_filter * 2) / (111 * grid_delta))/2)
                        kwargs = {"transform_first": True}
                        ctr = plt.contour(lon_nat+0.5,lat_nat-0.5,grid_prob_part*100,levels=levels_cntr*100, \
                            colors='k',linewidths=1,transform=ccrs.PlateCarree(),**kwargs)
                        plt.clabel(ctr,ctr.levels,inline=True,inline_spacing=-5,fmt ='% 1.2f',fontsize=8)
                        #plt.clabel(ctr,ctr.levels,inline=False,fmt ='% 1.2f',fontsize=8)
                        #Create text at top/bottom portion of map
                        ax.text(0.5, 0.03, load_data_nc[model][0:-13],transform=ax.transAxes,fontsize=12,verticalalignment='bottom', \
                            horizontalalignment='center', bbox=props)
                        ax.text(0.5, 0.97, "Valid Time: "+curdate_curr.strftime('%Y-%m-%d at %H:%M UTC')+" - "+'{0:.0f}'.format(pre_acc)+' Hour Acc. ',
                            transform=ax.transAxes, fontsize=12,verticalalignment='top', horizontalalignment='center', bbox=props)

                        #Pass line instances to a legend to create centroid maximum legend by color
                        line_label=[]
                        line_str=[];
                        for x in range(len(colorlist_mag)):
                            line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color=colorlist_mag[x], linestyle='None', marker='o', markersize=7)))
                            line_str   = np.hstack((line_str, leg_nam_byint1[x]))
                        first_legend=plt.legend(line_label, line_str, fontsize=6, numpoints=1, loc=3, framealpha=1)
                        ax2 = plt.gca().add_artist(first_legend)

                        plot_map([latlon_dims_s[1],latlon_dims_s[3]],[latlon_dims_s[0],latlon_dims_s[2]],ax)
                        plt.show()
                        plt.savefig(FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot2_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+ \
                            '{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'_'+load_data_nc[model][0:-3]+'_obj'+str(obj_count)+'_f'+\
                            '{:02d}'.format(hr_inc+1)+'.png',bbox_inches='tight',dpi=200)
                        plt.close()

                        # os.system('scp '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot2_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+ \
                        #     '{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'_'+load_data_nc[model][0:-3]+'_obj'+str(obj_count)+'_f'+\
                        #     '{:02d}'.format(hr_inc+1)+'.png hpc@wpcrzdm:/home/people/hpc/www/htdocs/verification/mtd_exp/images')
                        # os.system('aws s3 cp '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot2_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+ \
                        #     '{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'_'+load_data_nc[model][0:-3]+'_obj'+str(obj_count)+'_f'+\
                        #     '{:02d}'.format(hr_inc+1)+'.png s3://s3-east-www.wpc.woc.noaa.gov/public/verification/mtd_exp/images/' \
                        #     +GRIB_PATH_DES+'_retroplot2_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+'_t'+ \
                        #     str(thres)+'_'+load_data_nc[model][0:-3]+'_obj'+str(obj_count)+'_f'+'{:02d}'.format(hr_inc+1)+'.png') 

                        grid_prob_full[:,:,obj_count,hr_inc] = interpolate.griddata(np.stack((lon_nat.flatten(),lat_nat.flatten()),axis=1),\
                            grid_prob_part.flatten(),(lon_prob,lat_prob),method='linear')

                        #Save previous snap index if all values are false
                        if np.nansum(ind_snap) > 0:
                            ind_snap_prev = ind_snap
                        hr_count += 1

                    #END through the hours
                    if hr_count > 0:
                        obj_count += 1
                    
                #END through each tracks

                #Assume probabilities are independent and sum them to create a total probability field for each hour
                grid_prob[:,:,:,model] = np.nansum(grid_prob_full,axis = 2)

                #Simple temporal smoothing
                for hr_inc in range(len(hrs)):
                    if hr_inc == 0:
                        grid_prob[:,:,hr_inc,model] = grid_prob[:,:,hr_inc-0,model]*0.6+\
                            grid_prob[:,:,hr_inc+1,model]*0.3+grid_prob[:,:,hr_inc+2,model]*0.1
                    elif hr_inc == 1:
                        grid_prob[:,:,hr_inc,model] = grid_prob[:,:,hr_inc-1,model]*0.25+\
                            grid_prob[:,:,hr_inc-0,model]*0.4+grid_prob[:,:,hr_inc+1,model]*0.25+\
                            grid_prob[:,:,hr_inc+2,model]*0.1
                    elif hr_inc == len(hrs)-2:
                        grid_prob[:,:,hr_inc,model] = grid_prob[:,:,hr_inc-2,model]*0.1+\
                            grid_prob[:,:,hr_inc-1,model]*0.25+grid_prob[:,:,hr_inc-0,model]*0.4+\
                            grid_prob[:,:,hr_inc+1,model]*0.25
                    elif hr_inc == len(hrs)-1:
                        grid_prob[:,:,hr_inc,model] = grid_prob[:,:,hr_inc-2,model]*0.1+\
                            grid_prob[:,:,hr_inc-1,model]*0.3+grid_prob[:,:,hr_inc-0,model]*0.6
                    else:
                        grid_prob[:,:,hr_inc,model] = grid_prob[:,:,hr_inc-2,model]*0.1+\
                            grid_prob[:,:,hr_inc-1,model]*0.2+grid_prob[:,:,hr_inc-0,model]*0.4+\
                            grid_prob[:,:,hr_inc+1,model]*0.2+grid_prob[:,:,hr_inc+2,model]*0.1

                #Create second colorlist, trimming the beginning lighter colors
                interval = np.linspace(0.3, 1)
                colors = plt.cm.Blues(interval)
                my_cmap2 = LinearSegmentedColormap.from_list('name', colors)
                my_cmap2.set_under('white')
                values = np.linspace(0.01,0.15,num=15)

                #Create CONUS wide probabilities of an object within 40 km of a point summed across all objects
                for hr_inc in range(len(hrs)):

                    #Determine time stamp for hour
                    curdate_curr = curdate_currn + datetime.timedelta(hours=hrs[hr_inc])

                    fig = plt.figure(figsize=(12,6)) #Create Figure
                    ax  = plt.axes(projection=ccrs.Mercator(),alpha=1) #Create geoaxes to the projection
                    props = dict(boxstyle='round', facecolor='wheat', alpha=1)

                    #Plot contour filled probably of being in an ensemble object
                    kwargs = {"transform_first": True}

                    cs = plt.contourf(lon_prob, lat_prob, grid_prob[:,:,hr_inc,model]*100,   levels=values*100, extend='both', cmap=my_cmap2, antialiased=False, vmin=np.spacing(0.0),\
                        vmax=np.nanmax(values)*100, transform=ccrs.PlateCarree(),**kwargs)
                    cb = plt.colorbar(cs,ax=ax)

                    ax.text(0.5, 0.03, load_data_nc[model][0:-13],transform=ax.transAxes,fontsize=12,verticalalignment='bottom', \
                        horizontalalignment='center', bbox=props)
                    ax.text(0.5, 0.97, "Valid Time: "+curdate_curr.strftime('%Y-%m-%d at %H:%M UTC')+" - "+'{0:.0f}'.format(pre_acc)+' Hour Acc. ',
                        transform=ax.transAxes, fontsize=12,verticalalignment='top', horizontalalignment='center', bbox=props)

                    plot_map([latlon_dims[1],latlon_dims[3]],[latlon_dims[0],latlon_dims[2]],ax)
                    plt.show()
                    plt.savefig(FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot1_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+ \
                        '{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'_'+load_data_nc[model][0:-3]+'_f'+\
                        '{:02d}'.format(hr_inc+1)+'.png',bbox_inches='tight',dpi=200)
                    plt.close()

                    # os.system('scp '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot1_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+ \
                    #     '{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'_'+load_data_nc[model][0:-3]+'_f'+\
                    #     '{:02d}'.format(hr_inc+1)+'.png hpc@wpcrzdm:/home/people/hpc/www/htdocs/verification/mtd_exp/images')
                    # os.system('aws s3 cp '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot1_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+ \
                    #     '{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'_'+load_data_nc[model][0:-3]+'_f'+\
                    #     '{:02d}'.format(hr_inc+1)+'.png s3://s3-east-www.wpc.woc.noaa.gov/public/verification/mtd_exp/images/' \
                    #     +GRIB_PATH_DES+'_retroplot1_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+'_t'+ \
                    #     str(thres)+'_'+load_data_nc[model][0:-3]+'_f'+'{:02d}'.format(hr_inc+1)+'.png')

            #END if data exists for a particular hour
        #END if track exists
    #END through the models

    # #Save the HPOD grid probabilities
    # np.savez('/usr1/wpc_cpgffh/gribs/HPOD/HPOD_'+GRIB_PATH_DES+'_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+\
    #     '{0:.2f}'.format(pre_acc)+'_t'+str(thres),probs = grid_prob, lat = lat_prob, lon = lon_prob)

    ##########COMMENT OUT PLOTTING CODE TO TEST SAVING THE DATA########################
    #fig = plt.figure(figsize=(12,6)) #Create Figure
    #ax  = plt.axes(projection=ccrs.Mercator(),alpha=1) #Create geoaxes to the projection
    #props = dict(boxstyle='round', facecolor='wheat', alpha=1)

    ##Plot the median retro displacement bias and a scatter of all the retro biases
    #kwargs = {"transform_first": True}
    #ctr = plt.contour(lon_prob+0.5,lat_prob-0.5,grid_prob[:,:,16,0],levels=levels_cntr, \
    #    colors='k',linewidths=1,transform=ccrs.PlateCarree(),**kwargs)
    #plt.clabel(ctr,ctr.levels,inline=True,inline_spacing=-5,fmt ='% 1.2f',fontsize=8)

    #cartopy_maps.plot_map([latlon_dims[1],latlon_dims[3]],[latlon_dims[0],latlon_dims[2]],ax)
    #plt.show()
    #plt.savefig(FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot1_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+ \
    #    '{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'_'+load_data_nc[model][0:-3]+'_obj'+str(obj_count)+'_f'+\
    #    '{:02d}'.format(hr_inc+1)+'.png',bbox_inches='tight',dpi=200)
    #plt.close()

    #print('scp '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot1_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+ \
    #    '{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'_'+load_data_nc[model][0:-3]+'_obj'+str(obj_count)+'_f'+\
    #    '{:02d}'.format(hr_inc+1)+'.png hpc@wpcrzdm:/home/people/hpc/www/htdocs/verification/mtd_exp/images')
    #os.system('scp '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_retroplot1_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+ \
    #    '{0:.2f}'.format(pre_acc)+'_t'+str(thres)+'_'+load_data_nc[model][0:-3]+'_obj'+str(obj_count)+'_f'+\
    #    '{:02d}'.format(hr_inc+1)+'.png hpc@wpcrzdm:/home/people/hpc/www/htdocs/verification/mtd_exp/images')
    #########END TEST PLOTTING CODE########################################################

    # #Clear files older than 3 days from wpcrzdm to save memory
    # curdate_old  = curdate - datetime.timedelta(days=3)
    # yrmonday_old = '{:04d}'.format(curdate_old.year)+'{:02d}'.format(curdate_old.month)+'{:02d}'.format(curdate_old.day)
    # os.system("ssh hpc@wpcrzdm 'rm /home/people/hpc/www/htdocs/verification/mtd_exp/images/*"+GRIB_PATH_DES+"*"+yrmonday_old+"*png'")
    # print("ssh hpc@wpcrzdm 'rm /home/people/hpc/www/htdocs/verification/mtd_exp/images/*"+GRIB_PATH_DES+"*"+yrmonday_old+"*png'")
    # os.system('aws s3 rm s3://s3-east-www.wpc.woc.noaa.gov/public/verification/mtd_exp/images/ --recursive --exclude "*" --include "*'+\
    #     yrmonday_old+'*.*"')


###################################################################################
#THIS FUNCTION IS DESGINED TO LOAD AND PLOT ALL PAIRED MODEL/OBS FUNCTION ATTRIBUTES 
#AND PRECIPITATION BY MODEL INITIALIZATION FOR JUST A SINGLE SPECIFIED RUN. IT IS 
#DESIGNED TO COMPARE INTUITIVE "BY EYE" RESULTS TO THE TRACKER SO THE TRACKER CAN 
#BE TUNED AND ADJUSTED AS NEEDED. THIS FUNCTION LOADS THE RETROSPECTIVE DATA FORMAT ONLY.
#
#Created by MJE. 20171205.
#####################INPUT FILES FOR MTDPlotRetro####################################
#GRIB_PATH_DES   = unique identifier assigned to figure name
#FIG_PATH_s      = path where figures are to be saved
#latlon_dims     = latitude/longitude dimensions for plotting [WLON,SLAT,ELON,NLAT]
#pre_acc         = accumulation interval
#hrs             = array of forecast hours
#raw_thres       = raw threshold for MODE
#curdate         = datetime attribute
#data_success    = record of which data were succesfully loaded
#load_data_nc    = list of model/obs data to be loaded
#lat             = gridded latitude data
#lon             = gridded longitude data
#fcst            = raw forecast field
#obs             = raw observation field
#clus_bin        = gridded cluster data
#pair_prop       = paired cluster centroid info of lat,lon,max. Takes the form of [model][attributes]
####################OUTPUT FILES FOR MTDPlotRetro####################################
#Note: "Simple" means unmerged while "cluster" refers to merged
#"Single" refers to unmatched objects while "pair" refers to matched objects.
#
###################################################################################
def mtd_plot_retro(GRIB_PATH_DES,FIG_PATH_s,latlon_dims,pre_acc,hrs,raw_thres,curdate,data_success,load_data_nc,lat,lon,fcst,obs,clus_bin,pair_prop):

    #fcst       = fcst_p
    #obs        = obs_p
    #hrs        = hrs_all
    #raw_thres  = thres
    #FIG_PATH_s = FIG_PATH
    #latlon_dims = [lonmin,latmin,lonmax,latmax]
    
    #Find and isolate ST4 data
    isolate_analy=[]
    for i in range(0,len(load_data_nc),1):
        if 'ST4' in load_data_nc[i] or 'MRMS' in load_data_nc[i]:
            isolate_analy = np.append(isolate_analy,False)
        else:
            isolate_analy = np.append(isolate_analy,True)
    isolate_analy       = isolate_analy == 1

    #Find one model and one analy and analy is blank in 'pair_prop'
    if np.nanmean(isolate_analy) == 0.5 and np.mean(np.isnan(pair_prop[0])) == 1:
        pair_prop[0] = pair_prop[1] + 0 #Both model/analy included in all pair_prop
        
    #Set proper time index depending on ensemble
    for model in range(0,len(load_data_nc)):
        pair_prop[model][2] = (pair_prop[model][2] + 1) * pre_acc
        
    #Currently this function only requires a logical matrix
    clus_bin = clus_bin > 0
    
    #Add NaNs to missing data in binary matrix. Convert to float
    clus_bin=clus_bin.astype(float)
    NaN_loc = (np.mean(data_success,axis=1)>0)*1.0
    NaN_loc[NaN_loc == 1] = np.NaN
    
    for j in range(clus_bin.shape[0]):
        for i in range(clus_bin.shape[1]):
            for m in range(clus_bin.shape[3]):
                clus_bin[j,i,:,m] = clus_bin[j,i,:,m] + NaN_loc

    #Compute the proper forecast hour strings
    curdate_currn = curdate + datetime.timedelta(hours=0)
    
    #Establish a smoothing kernel to smooth ensemble probaiblities
    kern=np.hanning(50)
    kern /= kern.sum() 
    
    #Create first colorlist
    colorlist = ["#ffffff", "#02fd02", "#01c501", "#228b22", "#9acd32", "#fdf802", "#f8d826", "#e5bc00", "#fd9500", "#fd0000", "#bc0000"]
    my_cmap1 = mpl.colors.ListedColormap(colorlist)

    #Given a default model name list, build a list based on user specified models
    leg_nam_bymod_default = ['ST4'  'MRMS','HIRESNAM','HIRESW','HRRR','NSSL-OP','NSSL-ARW','NSSL-NMB','HRAM','NEWSe']
    leg_sym_bymod_default = ['s','d','*','^','<','>','v','o']
    
    if len(load_data_nc) <= 8: #If list is small enough, specific markers for individual models
        leg_nam_bymod  = load_data_nc
        leg_sym_bymod  = leg_sym_bymod_default[0:len(load_data_nc)]
    else:                      #Create markers by model type
        leg_nam_bymod = []
        leg_sym_bymod = ['d']
    
    leg_nam_byint  = ['>='+str(x)+"\"" for x in np.round(np.linspace(0,1*10*raw_thres,len(colorlist)),2)]

    #Create second colorlist, replace first element with white
    my_cmap2 = copy.copy( mpl.cm.get_cmap('cool'))
    my_cmap2.set_under('white')
     
    #############################################################################################
    #ANIMATION OF OBJ OUTLINE, INTENSITY, TYPE FOR MODEL AND OBSERVATION#########################
    #############################################################################################
    fig_name=[];
    
    hr_count = 0
    for hr_inc in hrs:
        
        if np.isnan(NaN_loc[hr_count]) != 1:

            #Determine time stamp for hour
            curdate_curr = curdate_currn + datetime.timedelta(hours=hr_inc)
        
            #Create range of values to plot
            fig = plt.figure(figsize=(12,6)) #Create Figure
            ax  = plt.axes(projection=ccrs.Mercator(),alpha=1) #Create geoaxes to the projection
            props = dict(boxstyle='round', facecolor='wheat', alpha=1)
            values = np.linspace(0,1,11)
            ax.text(0.5, 0.97, "Valid Time: "+curdate_curr.strftime('%Y%m%d%H%M')+" - "+'{0:.2f}'.format(pre_acc)+' Hour Accumulation ',
                transform=ax.transAxes, fontsize=12,verticalalignment='top', horizontalalignment='center', bbox=props)
            #Plot the model object outline in grey
            if np.sum(clus_bin[:,:,hr_count,isolate_analy==1]) > 0:
                try:
                    kwargs = {"transform_first": True}
                    cs = plt.contour(lon, lat, np.squeeze(np.round(clus_bin[:,:,hr_count,isolate_analy==1],2)),   colors='grey', linewidths=1, \
                        zorder=1, transform=ccrs.PlateCarree(),**kwargs)
                except ValueError:
                    pass
            #Plot the observation object outline in black
            if np.sum(clus_bin[:,:,hr_count,isolate_analy==0]) > 0:
                try:
                    kwargs = {"transform_first": True}
                    cs = plt.contour(lon, lat, np.squeeze(np.round(clus_bin[:,:,hr_count,isolate_analy==0],2)),   colors='black', linewidths=1, zorder=1, \
                        transform=ccrs.PlateCarree(),**kwargs)
                except ValueError:
                    pass
                    
        #Plot object attributes
        for model in range(len(load_data_nc)):
      
            #Specify marker properly 
            if len(load_data_nc) == len(leg_nam_bymod):
                marker = leg_sym_bymod[model]    
            else:
                for bymod in range(0,len(leg_nam_bymod)):
                    if leg_nam_bymod[bymod] in load_data_nc[model]:
                        marker=leg_sym_bymod[bymod]
                
            #Find modeled/observed objects in metadata
            if np.mean(np.isnan(pair_prop[model])) != 1:
                
                #Isolate/identify specific hour object attributes
                hr_loc = pair_prop[model][2] == hr_inc
                
                #If data exists for that hour, isolate and plot
                if len(hr_loc) > 0:
                    elements = np.unique(pair_prop[model][0][hr_loc])
                    
                    #Separate model and observations and set plotting properties
                    if isolate_analy[model] == 0:
                        elements   = elements[elements < 0]
                        linecolor  = 'b'
                        linewidth  = 1.5
                        markersize = 50
                        zorder     = 3
                    else:
                        elements = elements[elements > 0]
                        linecolor  = 'k'
                        linewidth  = 0.5
                        markersize = 25
                        zorder     = 2

                    for tot_obj in elements: #Through each track
                        #Tracks for instantaneous time and total time
                        ind_snap = np.asarray(pair_prop[model][0] == tot_obj) & np.asarray(pair_prop[model][2] == hr_inc)
                        ind_all  = np.asarray(pair_prop[model][0] == tot_obj)
                        
                        if len(pair_prop[model][5][ind_snap]) > 2:
                            plt.plot(pair_prop[model][5][ind_snap], pair_prop[model][4][ind_snap], c = linecolor, linewidth=linewidth, zorder=zorder, \
                                transform=ccrs.PlateCarree())
                            plt.scatter(pair_prop[model][5][ind_snap], pair_prop[model][4][ind_snap], c = pair_prop[model][9][ind_snap], \
                                marker=marker, vmin=0, vmax=np.nanmax(values), s=markersize, linewidth=0.25, cmap=my_cmap1, zorder=zorder, alpha=1, \
                                transform=ccrs.PlateCarree())  
                        else:
                            plt.plot(pair_prop[model][5][ind_snap][0], pair_prop[model][4][ind_snap][0], c = linecolor, linewidth=linewidth, zorder=zorder, \
                                transform=ccrs.PlateCarree())
                            plt.scatter(pair_prop[model][5][ind_snap][0], pair_prop[model][4][ind_snap][0], c = pair_prop[model][9][ind_snap][0], \
                                marker=marker, vmin=0, vmax=np.nanmax(values), s=markersize, linewidth=0.25, cmap=my_cmap1, zorder=zorder, alpha=0.8, \
                                transform=ccrs.PlateCarree())
                                
                        if len(pair_prop[model][5][ind_all]) > 2:
                            plt.plot(pair_prop[model][5][ind_all], pair_prop[model][4][ind_all], color=linecolor, linewidth=linewidth-0.25, zorder=zorder, \
                                transform=ccrs.PlateCarree()) 
                        else:
                            plt.plot(pair_prop[model][5][ind_all][0], pair_prop[model][4][ind_all][0], color=linecolor, linewidth=linewidth-0.25, zorder=zorder, \
                                transform=ccrs.PlateCarree())
                            
                                            
                    #COMMENTS: pre_acc should not be used to build values range.  instead use thres and only thres so 0.1 spans from 0.1 to 1                                        
                                                                            
                    #if len(pair_prop[fcst_loc][5][ind_snap]) > 2:
                    #    plt.plot(pair_prop[fcst_loc][5][ind_snap], pair_prop[fcst_loc][4][ind_snap], c = linecolor, linewidth=linewidth, zorder=zorder,  )
                    #    plt.scatter(pair_prop[fcst_loc][5][ind_snap], pair_prop[fcst_loc][4][ind_snap], c = pair_prop[fcst_loc][9][ind_snap], \
                    #        marker=marker, vmin=0, vmax=np.nanmax(values), s=markersize, linewidth=0.25, cmap=my_cmap1, zorder=zorder, alpha=1,  )  
                    #else:
                    #    plt.plot(pair_prop[fcst_loc][5][ind_snap][0], pair_prop[fcst_loc][4][ind_snap][0], c = linecolor, linewidth=linewidth, zorder=zorder,  )
                    #    plt.scatter(pair_prop[fcst_loc][5][ind_snap][0], pair_prop[fcst_loc][4][ind_snap][0], c = pair_prop[fcst_loc][9][ind_snap][0], \
                    #        marker=marker, vmin=0, vmax=np.nanmax(values), s=markersize, linewidth=0.25, cmap=my_cmap1, zorder=zorder, alpha=0.8,  )
                    #        
                    #if len(pair_prop[fcst_loc][5][ind_all]) > 2:
                    #    plt.plot(pair_prop[fcst_loc][5][ind_all], pair_prop[fcst_loc][4][ind_all], color=linecolor, linewidth=linewidth-0.25, zorder=zorder,  ) 
                    #else:
                    #1    plt.plot(pair_prop[fcst_loc][5][ind_all][0], pair_prop[fcst_loc][4][ind_all][0], color=linecolor, linewidth=linewidth-0.25, zorder=zorder,  )
                        
                                                        
                    #END check for hourly data existing
                #END model VS obs check
            #END through each track
        #END through the models
    
        #Pass line instances to a legend to create centroid maximum legend by color
        line_label=[]
        line_str=[];
        for x in range(len(colorlist)):
            line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color=colorlist[x], linestyle='None', marker='o', markersize=7)))
            line_str   = np.hstack((line_str, leg_nam_byint[x]))
        first_legend=plt.legend(line_label, line_str, fontsize=7, numpoints=1, loc=3, framealpha=1)
        ax2 = plt.gca().add_artist(first_legend)
        
        #Pass line instances to a legend denoting member-type by markersize
        line_label=[]
        line_str=[];
        for x in range(len(leg_sym_bymod)):
            line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color='k', linestyle='None', marker=leg_sym_bymod[x], markersize=7)))
            line_str   = np.hstack((line_str, leg_nam_bymod[x]))
        plt.legend(line_label, line_str, fontsize=7, numpoints=1, loc=4, framealpha=1)
        
        #Create the title
        ax.set_title("Ensemble Object Probabilities - "+"Precip. >= "+str(raw_thres)+"\" \n "+'{0:.2f}'.format(pre_acc)+ \
            " Hour Acc. Precip. - Initialized "+curdate.strftime('%Y%m%d%H'),fontsize=13)
            
        #Show and save the plot
        #plt.hold(True)
        plot_map([latlon_dims[1],latlon_dims[3]],[latlon_dims[0],latlon_dims[2]],ax)
        plt.show()

        plt.savefig(FIG_PATH_s+'/'+GRIB_PATH_DES+'_TOTENS_objshape_byhour_'+curdate_currn.strftime('%Y%m%d%H%M')+ \
            '_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'_f'+'{:02d}'.format(int(hr_count)+1)+'.png', bbox_inches='tight',dpi=150)
        plt.close()
        
        #Update figure names to create animated figure
        fig_name=np.append(fig_name,FIG_PATH_s+'/'+GRIB_PATH_DES+'_TOTENS_objshape_byhour_'+ \
            curdate_currn.strftime('%Y%m%d%H%M')+'_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'_f'+'{:02d}'.format(int(hr_count)+1)+'.png')
            
        hr_count += 1

    os.system('convert -delay 100 '+' '.join(fig_name)+' -loop 0 '+FIG_PATH_s+'/'+ \
        GRIB_PATH_DES+'_TOTENS_objshape_byhour_'+curdate_currn.strftime('%Y%m%d%H%M')+'_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'.gif')
 
    #############################################################################################
    ####ANIMATION OF RAW FORECAST PRECIPITATION WITH OBJECT ATTRIBUTE OVERLAY####################
    #############################################################################################

    fig_name=[];
    
    hr_count = 0
    for hr_inc in hrs:
        
        if np.isnan(NaN_loc[hr_count]) != 1:

            #Determine time stamp for hour
            curdate_curr = curdate_currn + datetime.timedelta(hours=hr_inc)
        
            #Create range of values to plot
            values = np.linspace(0,0.3,11)

            fig = plt.figure(figsize=(12,6)) #Create Figure
            ax  = plt.axes(projection=ccrs.Mercator(),alpha=1) #Create geoaxes to the projection
            props = dict(boxstyle='round', facecolor='wheat', alpha=1)
            ax.text(0.5, 0.97, "Valid Time: "+curdate_curr.strftime('%Y%m%d%H%M')+" - "+'{0:.2f}'.format(pre_acc)+' Hour Accumulation ',
                transform=ax.transAxes, fontsize=12,verticalalignment='top', horizontalalignment='center', bbox=props)
            #Plot the model object outline in grey
            if np.nansum(fcst[:,:,hr_count]) > 0:
                kwargs = {"transform_first": True}
                cs = plt.contourf(lon, lat, np.round(fcst[:,:,hr_count],2),   levels=values, extend='min', cmap=my_cmap2, antialiased=False, vmin=raw_thres,\
                    vmax=np.nanmax(values), transform=ccrs.PlateCarree(),**kwargs)
                cb = plt.colorbar(cs,ax=ax)
        #Plot forecast object attributes
        fcst_loc = np.where(isolate_analy==1)[0][0]
        
        #Specify marker properly 
        if len(load_data_nc) == len(leg_nam_bymod):
            marker = leg_sym_bymod[fcst_loc]    
        else:
            for bymod in range(0,len(leg_nam_bymod)):
                if leg_nam_bymod[bymod] in load_data_nc[fcst_loc]:
                    marker=leg_sym_bymod[bymod]
            
        #Find modeled/observed objects in metadata
        if np.mean(np.isnan(pair_prop[fcst_loc])) != 1:
            
            #Isolate/identify specific hour object attributes
            hr_loc = pair_prop[fcst_loc][2] == hr_inc
            
            #If data exists for that hour, isolate and plot
            if len(hr_loc) > 0:
                elements = np.unique(pair_prop[fcst_loc][0][hr_loc])
                
                #Separate model and observations and set plotting properties

                elements   = elements[elements > 0]
                linecolor  = 'b'
                linewidth  = 1.5
                markersize = 50
                zorder     = 3
                
                for tot_obj in elements: #Through each track
                    #Tracks for instantaneous time and total time
                    ind_snap = np.asarray(pair_prop[fcst_loc][0] == tot_obj) & np.asarray(pair_prop[fcst_loc][2] == hr_inc)
                    ind_all  = np.asarray(pair_prop[fcst_loc][0] == tot_obj)
                    
                    if len(pair_prop[fcst_loc][5][ind_snap]) > 2:
                        plt.plot(pair_prop[fcst_loc][5][ind_snap], pair_prop[fcst_loc][4][ind_snap], c = linecolor, linewidth=linewidth, zorder=zorder, \
                            transform=ccrs.PlateCarree())
                        plt.scatter(pair_prop[fcst_loc][5][ind_snap], pair_prop[fcst_loc][4][ind_snap], c = pair_prop[fcst_loc][9][ind_snap], \
                            marker=marker, vmin=0, vmax=np.nanmax(values), s=markersize, linewidth=0.25, cmap=my_cmap1, zorder=zorder, alpha=1, \
                            transform=ccrs.PlateCarree())  
                    else:
                        plt.plot(pair_prop[fcst_loc][5][ind_snap][0], pair_prop[fcst_loc][4][ind_snap][0], c = linecolor, linewidth=linewidth, zorder=zorder, \
                            transform=ccrs.PlateCarree())
                        plt.scatter(pair_prop[fcst_loc][5][ind_snap][0], pair_prop[fcst_loc][4][ind_snap][0], c = pair_prop[fcst_loc][9][ind_snap][0], \
                            marker=marker, vmin=0, vmax=np.nanmax(values), s=markersize, linewidth=0.25, cmap=my_cmap1, zorder=zorder, alpha=0.8, \
                            transform=ccrs.PlateCarree())
                            
                    if len(pair_prop[fcst_loc][5][ind_all]) > 2:
                        plt.plot(pair_prop[fcst_loc][5][ind_all], pair_prop[fcst_loc][4][ind_all], color=linecolor, linewidth=linewidth-0.25, zorder=zorder, \
                            transform=ccrs.PlateCarree()) 
                    else:
                        plt.plot(pair_prop[fcst_loc][5][ind_all][0], pair_prop[fcst_loc][4][ind_all][0], color=linecolor, linewidth=linewidth-0.25, zorder=zorder, \
                            transform=ccrs.PlateCarree())
                                        
                #END check for hourly data existing
            #END model VS obs check
        #END through each track
    
        #Pass line instances to a legend to create centroid maximum legend by color
        line_label=[]
        line_str=[];
        for x in range(len(colorlist)):
            line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color=colorlist[x], linestyle='None', marker='o', markersize=7)))
            line_str   = np.hstack((line_str, leg_nam_byint[x]))
        first_legend=plt.legend(line_label, line_str, fontsize=7, numpoints=1, loc=3, framealpha=1)
        ax2 = plt.gca().add_artist(first_legend)
        
        #Pass line instances to a legend denoting member-type by markersize
        line_label=[]
        line_str=[];
        for x in range(len(leg_sym_bymod)):
            line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color='k', linestyle='None', marker=leg_sym_bymod[x], markersize=7)))
            line_str   = np.hstack((line_str, leg_nam_bymod[x]))
        plt.legend(line_label, line_str, fontsize=7, numpoints=1, loc=4, framealpha=1)
        
        #Create the title
        ax.set_title("Model Object and Raw Precipitation - "+"Precip. >= "+str(raw_thres)+"\" \n "+'{0:.2f}'.format(pre_acc)+ \
            " Hour Acc. Precip. - Initialized "+curdate.strftime('%Y%m%d%H%M'),fontsize=13)
            
        #Show and save the plot
        #plt.hold(True)
        plot_map([latlon_dims[1],latlon_dims[3]],[latlon_dims[0],latlon_dims[2]],ax)
        plt.show()
        plt.savefig(FIG_PATH_s+'/'+GRIB_PATH_DES+'_fcst_precipAobjs_byhour_'+curdate_currn.strftime('%Y%m%d%H%M')+ \
            '_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'_f'+'{:02d}'.format(int(hr_count)+1)+'.png', bbox_inches='tight',dpi=150)
        plt.close()
        
        #Update figure names to create animated figure
        fig_name=np.append(fig_name,FIG_PATH_s+'/'+GRIB_PATH_DES+'_fcst_precipAobjs_byhour_'+\
            curdate_currn.strftime('%Y%m%d%H%M')+'_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'_f'+'{:02d}'.format(int(hr_count)+1)+'.png')

        hr_count += 1
    #Create an animation from the figures previously saved
    os.system('convert -delay 50 '+' '.join(fig_name)+' -loop 0 '+FIG_PATH_s+'/'+ \
        GRIB_PATH_DES+'_fcst_precipAobjs_byhour_'+curdate_currn.strftime('%Y%m%d%H%M')+'_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'.gif')

    #############################################################################################
    ####ANIMATION OF RAW OBSERVATION PRECIPITATION WITH OBJECT ATTRIBUTE OVERLAY#################
    #############################################################################################

    fig_name=[];
    hr_count = 0
    for hr_inc in hrs:
        
        if np.isnan(NaN_loc[hr_count]) != 1:

            #Determine time stamp for hour
            curdate_curr = curdate_currn + datetime.timedelta(hours=hr_inc)
        
            #Create range of values to plot
            values = np.linspace(0,0.3,11)

            fig = plt.figure(figsize=(12,6)) #Create Figure
            ax  = plt.axes(projection=ccrs.Mercator(),alpha=1) #Create geoaxes to the projection
            props = dict(boxstyle='round', facecolor='wheat', alpha=1)
            ax.text(0.5, 0.97, "Valid Time: "+curdate_curr.strftime('%Y%m%d%H%M')+" - "+'{0:.2f}'.format(pre_acc)+' Hour Accumulation ',
                transform=ax.transAxes, fontsize=12,verticalalignment='top', horizontalalignment='center', bbox=props)
            #Plot the obs object outline in grey
            if np.nansum(obs[:,:,hr_count]) > 0:
                kwargs = {"transform_first": True}
                cs = plt.contourf(lon, lat, np.round(obs[:,:,hr_count],2),   levels=values, extend='min', cmap=my_cmap2, antialiased=False, vmin=raw_thres,\
                    vmax=np.nanmax(values), transform=ccrs.PlateCarree(),**kwargs)
                cb = plt.colorbar(cs,ax=ax)

        #Plot observation object attributes
        obs_loc = np.where(isolate_analy==0)[0][0]
        
        #Specify marker properly 
        if len(load_data_nc) == len(leg_nam_bymod):
            marker = leg_sym_bymod[obs_loc]    
        else:
            for bymod in range(0,len(leg_nam_bymod)):
                if leg_nam_bymod[bymod] in load_data_nc[obs_loc]:
                    marker=leg_sym_bymod[bymod]
            
        #Find modeled/observed objects in metadata
        if np.mean(np.isnan(pair_prop[obs_loc])) != 1:
            
            #Isolate/identify specific hour object attributes
            hr_loc = pair_prop[obs_loc][2] == hr_inc
            
            #If data exists for that hour, isolate and plot
            if len(hr_loc) > 0:
                elements = np.unique(pair_prop[obs_loc][0][hr_loc])
                
                #Separate model and observations and set plotting properties

                elements   = elements[elements < 0]
                linecolor  = 'b'
                linewidth  = 1.5
                markersize = 50
                zorder     = 3

                for tot_obj in elements: #Through each track
                    #Tracks for instantaneous time and total time
                    ind_snap = np.asarray(pair_prop[obs_loc][0] == tot_obj) & np.asarray(pair_prop[obs_loc][2] == hr_inc)
                    ind_all  = np.asarray(pair_prop[obs_loc][0] == tot_obj)
                    
                    if len(pair_prop[obs_loc][5][ind_snap]) > 2:
                        plt.plot(pair_prop[obs_loc][5][ind_snap], pair_prop[obs_loc][4][ind_snap], c = linecolor, linewidth=linewidth, zorder=zorder, \
                            transform=ccrs.PlateCarree())
                        plt.scatter(pair_prop[obs_loc][5][ind_snap], pair_prop[obs_loc][4][ind_snap], c = pair_prop[obs_loc][9][ind_snap], \
                            marker=marker, vmin=0, vmax=np.nanmax(values), s=markersize, linewidth=0.25, cmap=my_cmap1, zorder=zorder, alpha=1, \
                            transform=ccrs.PlateCarree())  
                    else:
                        plt.plot(pair_prop[obs_loc][5][ind_snap][0], pair_prop[obs_loc][4][ind_snap][0], c = linecolor, linewidth=linewidth, zorder=zorder, \
                            transform=ccrs.PlateCarree())
                        plt.scatter(pair_prop[obs_loc][5][ind_snap][0], pair_prop[obs_loc][4][ind_snap][0], c = pair_prop[obs_loc][9][ind_snap][0], \
                            marker=marker, vmin=0, vmax=np.nanmax(values), s=markersize, linewidth=0.25, cmap=my_cmap1, zorder=zorder, alpha=0.8, \
                            transform=ccrs.PlateCarree())
                            
                    if len(pair_prop[obs_loc][5][ind_all]) > 2:
                        plt.plot(pair_prop[obs_loc][5][ind_all], pair_prop[obs_loc][4][ind_all], color=linecolor, linewidth=linewidth-0.25, zorder=zorder, \
                            transform=ccrs.PlateCarree()) 
                    else:
                        plt.plot(pair_prop[obs_loc][5][ind_all][0], pair_prop[obs_loc][4][ind_all][0], color=linecolor, linewidth=linewidth-0.25, zorder=zorder, \
                            transform=ccrs.PlateCarree())
                                        
                #END check for hourly data existing
            #END model VS obs check
        #END through each track
    
        #Pass line instances to a legend to create centroid maximum legend by color
        line_label=[]
        line_str=[];
        for x in range(len(colorlist)):
            line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color=colorlist[x], linestyle='None', marker='o', markersize=7)))
            line_str   = np.hstack((line_str, leg_nam_byint[x]))
        first_legend=plt.legend(line_label, line_str, fontsize=7, numpoints=1, loc=3, framealpha=1)
        ax2 = plt.gca().add_artist(first_legend)
        
        #Pass line instances to a legend denoting member-type by markersize
        line_label=[]
        line_str=[];
        for x in range(len(leg_sym_bymod)):
            line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color='k', linestyle='None', marker=leg_sym_bymod[x], markersize=7)))
            line_str   = np.hstack((line_str, leg_nam_bymod[x]))
        plt.legend(line_label, line_str, fontsize=7, numpoints=1, loc=4, framealpha=1)
        
        #Create the title
        ax.set_title("Observed Object and Raw Precipitation - "+"Precip. >= "+str(raw_thres)+"\" \n "+'{0:.2f}'.format(pre_acc)+ \
            " Hour Acc. Precip. - Initialized "+curdate.strftime('%Y%m%d%H%M'),fontsize=13)
            
        #Show and save the plot
        #plt.hold(True)
        plot_map([latlon_dims[1],latlon_dims[3]],[latlon_dims[0],latlon_dims[2]],ax)
        plt.show()
        plt.savefig(FIG_PATH_s+'/'+GRIB_PATH_DES+'_obs_precipAobjs_byhour_'+curdate_currn.strftime('%Y%m%d%H%M')+ \
            '_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'_f'+'{:02d}'.format(int(hr_count)+1)+'.png', bbox_inches='tight',dpi=150)
        plt.close()
        
        #Update figure names to create animated figure
        fig_name=np.append(fig_name,FIG_PATH_s+'/'+GRIB_PATH_DES+'_obs_precipAobjs_byhour_'+ \
            curdate_currn.strftime('%Y%m%d%H%M')+'_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'_f'+'{:02d}'.format(int(hr_count)+1)+'.png')

        hr_count += 1
    #Create an animation from the figures previously saved
    os.system('convert -delay 50 '+' '.join(fig_name)+' -loop 0 '+FIG_PATH_s+ \
        GRIB_PATH_DES+'_obs_precipAobjs_byhour_'+curdate_currn.strftime('%Y%m%d%H%M')+'_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'.gif')
    
    for model in range(0,len(load_data_nc)):
        pair_prop[model][2] = (pair_prop[model][2] * (1 / pre_acc)) - 1

###################################################################################
#mtd_plot_all_fcst creates three plots: 1) forecast ensemble object probability (shading) and 
#object centroid location/magnitude/model type (marker). If observation exists then the
#observed object is plotted (black outline) in addition to centroid location/magnitude.
#2) forecast ensemble object probability (shading) and object centroid location/forecast 
#hour/model type (marker). If observation exists, observed object is plotted (black outline)
#in addition to centroid location/magnitude. 3) An hourly animation of figure 1) is created
#to better convey temporal details of the objects.
#
#Created by MJE. 20170327-20170420
#Updated to include adjustable smoother. 20190604. MJE
#Updated to include NEWSe (in 2019) and HRRR 15 min. plotting. 20201218. MJE
#
#####################INPUT FILES FOR MODEPlotAll####################################
#GRIB_PATH_DES   = unique identifier assigned to figure name
#FIG_PATH_s      = path where figures are to be saved
#latlon_dims     = latitude/longitude dimensions for plotting [WLON,SLAT,ELON,NLAT]
#pre_acc         = accumulation interval
#hrs             = array of forecast hours
#raw_thres       = raw threshold for MODE
#curdate         = datetime attribute
#data_success    = record of which data were succesfully loaded
#load_data_nc    = gridded variable data
#lat             = gridded latitude data
#lon             = gridded longitude data
#simp_bin        = gridded simple data
#simp_prop       = simple cluster centroid info of lat,lon,max. Takes the form of [model][attributes]
#sigma           = sigma value for Gaussian smoother (radius of smoothing in grid points)
####################OUTPUT FILES FOR MODEPlotAllFcst####################################
#Note: "Simple" means unmerged while "cluster" refers to merged
#"Single" refers to unmatched objects while "pair" refers to matched objects.
#
###################################################################################
def mtd_plot_all_fcst(GRIB_PATH_DES,FIG_PATH_s,latlon_dims,pre_acc,hrs,raw_thres,curdate,data_success,load_data_nc,lat,lon,simp_bin,simp_prop,sigma):
    
    #latlon_dims   = latlon_sub[subsets]
    #GRIB_PATH_DES = GRIB_PATH_DES+domain_sub[subsets]
    #raw_thres     = thres

    #Set proper time index depending on ensemble
    if 'NEWSe5min' in ''.join(load_data_nc) or 'HRRRv415min' in ''.join(load_data_nc):
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = (simp_prop[model][2] / 4.0) + 0.25
    else:
        for model in range(0,len(load_data_nc)):

            simp_prop[model][2] = simp_prop[model][2] + 1
    
    #Find and isolate ST4 data
    isolate_st4=[]
    for i in range(0,len(load_data_nc),1):
        if 'ST4' in load_data_nc[i]:
            isolate_st4 = np.append(isolate_st4,False)
        else:
            isolate_st4 = np.append(isolate_st4,True)
    isolate_st4       = isolate_st4 == 1

    #Currently this function only requires a logical matrix
    simp_bin = simp_bin > 0
    
    #Add NaNs to missing data in binary matrix. Convert to float
    simp_bin=simp_bin.astype(float)
    NaN_loc = (data_success == 1).astype(float)
    NaN_loc[NaN_loc == 1] = np.NaN
    
    for j in range(simp_bin.shape[0]):
        for i in range(simp_bin.shape[1]):
            simp_bin[j,i,:,:] = simp_bin[j,i,:,:] + NaN_loc
            
    #Compute raw ensemble mean probability, set 100% to 99% to preserve colorbar
    ens_mean = np.nanmax(np.nanmean(simp_bin[:,:,:,isolate_st4], axis = 3) , axis = 2)
    ens_mean[ens_mean > 0.95] = 0.95
    
    #Compute the proper forecast hour strings
    curdate_currn = curdate + datetime.timedelta(hours=0)
    curdate_start = curdate + datetime.timedelta(hours=hrs[0])
    curdate_stop  = curdate + datetime.timedelta(hours=hrs[-1])
    
    #Establish a smoothing kernel to smooth ensemble probaiblities
    kern=np.hanning(50)
    kern /= kern.sum() 
    
    #Create first colorlist
    colorlist = ["#ffffff", "#02fd02", "#01c501", "#228b22", "#9acd32", "#fdf802", "#f8d826", "#e5bc00", "#fd9500", "#fd0000", "#bc0000"]
    my_cmap1 = mpl.colors.ListedColormap(colorlist)

    #Given a default model name list, build a list based on user specified models
    leg_nam_bymod_default = ['ST4'  ,'HIRESNAM','HIRESW','HRRR','NSSL-OP','NSSL-ARW','NSSL-NMB','HRAM','NEWSe']
    leg_sym_bymod_default = ['s','d','*','^','<','>','v','o','8']
    
    if len(load_data_nc) <= 8: #If list is small enough, specific markers for individual models
        leg_nam_bymod  = load_data_nc
        leg_sym_bymod  = leg_sym_bymod_default[0:len(load_data_nc)]
    else:                      #Create markers by model type
        leg_nam_bymod = []
        leg_sym_bymod = ['d']
#        for bymod in range(0,len(leg_nam_bymod_default)):
#            if np.sum([leg_nam_bymod_default[bymod] in i for i in load_data_nc]) > 0:
#                leg_nam_bymod = np.append(leg_nam_bymod,leg_nam_bymod_default[bymod])
#                leg_sym_bymod = np.append(leg_sym_bymod,'d')
    
    centroid_values = np.round(np.linspace(0,1*10*raw_thres,len(colorlist)),2)
    leg_nam_byint  = ['90th Object Percentile >='+str(x)+"\"" for x in centroid_values]
    leg_nam_bytime = ['Hours '+str(hrs[i])+' - '+str(hrs[i+int(round(len(hrs)/len(colorlist)))]) \
        for i in range(0,len(hrs)-int(round(len(hrs)/len(colorlist))),int(round(len(hrs)/len(colorlist))+1))]
        
    #Shorten initial colorlist for bytime plots
    my_cmap1_short = mpl.colors.ListedColormap(colorlist[0:len(leg_nam_bytime)])
       
    #Create second colorlist, trimming the beginning lighter colors
    interval = np.linspace(0.3, 1)
    colors = plt.cm.Blues(interval)
    my_cmap2 = LinearSegmentedColormap.from_list('name', colors)
    my_cmap2.set_under('white')
     
    ##########################################################################################
    ###PLOT OF OBJ PROB. (SHADED), MODEL TYPE (MARKERTYPE), AND OBJ. INTENSITY (MARKER COLOR)#
    ##########################################################################################
    
    #Create range of values to plot
    values = np.linspace(0,1,11)

    fig = plt.figure(figsize=(12,6)) #Create Figure
    ax  = plt.axes(projection=ccrs.Mercator(),alpha=1) #Create geoaxes to the projection
    props  = dict(boxstyle='round', facecolor='wheat', alpha=1)
    props2 = dict(boxstyle='round', facecolor='tan', alpha=1)

    if '_ALL' not in GRIB_PATH_DES: #Simplify 'ALL' plot of CONUS
        ax.text(0.5, 0.03, str(int(np.sum(np.mean(data_success[:,isolate_st4],axis=0)<1)))+' Members Found', 
            transform=ax.transAxes, fontsize=12,verticalalignment='bottom', horizontalalignment='center', bbox=props)
        ax.text(0.5, 0.97, "Valid Time: "+curdate_start.strftime('%Y%m%d%H%M')+" - "+curdate_stop.strftime('%Y%m%d%H%M'),
            transform=ax.transAxes, fontsize=12,verticalalignment='top', horizontalalignment='center', bbox=props)
    kwargs = {"transform_first": True}
    cs = plt.contourf(lon, lat, gaussian_filter(ens_mean*100, sigma),   levels=values*100, extend='min', cmap=my_cmap2, antialiased=False, vmin=np.spacing(0.0),\
        vmax=np.nanmax(values*100), transform=ccrs.PlateCarree(),**kwargs)
    cb = plt.colorbar(cs,ax=ax) 
    #Outline the observation if it exists
    str_inc = [i for i in range(len(isolate_st4)) if isolate_st4[i] == 0]
    if sum(isolate_st4 == 0) > 0:
        kwargs = {"transform_first": True}
        plt.contour(lon, lat, np.squeeze(np.mean(simp_bin[:,:,:,str_inc],axis=2)) > 0,   levels=values, cmap=mpl.colors.ListedColormap(["#000000"]), linewidths=2, \
                transform=ccrs.PlateCarree(),**kwargs)
        
    #Scatter plot includes intensity (color) and member (symbol)
    for model in range(len(load_data_nc)):
        
        #Specify marker properly 
        if len(load_data_nc) == len(leg_nam_bymod):
            marker = leg_sym_bymod[model]    
        elif len(leg_sym_bymod) == 1:
            marker = leg_sym_bymod[0]
        else:
            for bymod in range(0,len(leg_nam_bymod)):
                if leg_nam_bymod[bymod] in load_data_nc[model]:
                    marker=leg_sym_bymod[bymod]

        #Find modeled/observed objects in metadata
        if np.mean(np.isnan(simp_prop[model])) != 1:
            elements = np.unique(simp_prop[model][0])
            
            #Separate model and observations and set plotting properties
            if isolate_st4[model] == 0:
                elements   = elements[elements < 0]
                linecolor  = 'b'
                linewidth  = 1.5
                markersize = 60
                zorder     = 3
            else:
                elements = elements[elements > 0]
                linecolor  = 'k'
                linewidth  = 0.5
                markersize = 30
                zorder     = 2
    
            for tot_obj in elements: #Through each track
                ind = simp_prop[model][0] == tot_obj
                if len(simp_prop[model][4][ind]) > 2:
                    cs2= plt.plot(simp_prop[model][5][ind], simp_prop[model][4][ind], color='k', linewidth=linewidth, zorder=zorder,transform=ccrs.PlateCarree())
                    cs = plt.scatter(simp_prop[model][5][ind], simp_prop[model][4][ind], c = simp_prop[model][9][ind],marker=marker, \
                        vmin=0, vmax=np.nanmax(centroid_values), s=markersize, linewidth=linewidth, cmap=my_cmap1, zorder=zorder, alpha=1, transform=ccrs.PlateCarree()) 
                else:
                    cs2= plt.plot(simp_prop[model][5][ind][0], simp_prop[model][4][ind][0], color=linecolor, linewidth=1, zorder=zorder,transform=ccrs.PlateCarree())
                    cs = plt.scatter(simp_prop[model][5][ind][0], simp_prop[model][4][ind][0], c = simp_prop[model][9][ind][0],marker=marker, \
                        vmin=0, vmax=np.nanmax(centroid_values), s=markersize, linewidth=linewidth, cmap=my_cmap1, zorder=zorder, alpha=0.8, transform=ccrs.PlateCarree())
                        
            #END model VS obs check
        #END through each track
    #END through the models
         
    if '_ALL' not in GRIB_PATH_DES: #Simplify 'ALL' plot of CONUS
        #Pass line instances to a legend to create centroid maximum legend by color
        line_label=[]
        line_str=[];
        for x in range(len(colorlist)):
            line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color=colorlist[x], linestyle='None', marker='o', markersize=7)))
            line_str   = np.hstack((line_str, leg_nam_byint[x]))
        first_legend=plt.legend(line_label, line_str, fontsize=6, numpoints=1, loc=3, framealpha=1)
        ax2 = plt.gca().add_artist(first_legend)
        
        #Pass line instances to a legend denoting member-type by markersize
        if len(leg_nam_bymod) > 0: #Only plot name legend if > 1 unique name exists
            line_label=[]
            line_str=[];
            for x in range(len(leg_sym_bymod)):
                line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color='k', linestyle='None', marker=leg_sym_bymod[x], markersize=7)))
                line_str   = np.hstack((line_str, leg_nam_bymod[x]))
            plt.legend(line_label, line_str, fontsize=6, numpoints=1, loc=4, framealpha=1)
    
    #Create the title
    ax.set_title("Ensemble Object Probabilities - "+"Precip. >= "+str(raw_thres)+"\" \n "+'{0:.2f}'.format(pre_acc)+ \
        " Hour Acc. Precip. - Initialized "+curdate.strftime('%Y%m%d%H'),fontsize=13)
        
    #Special condition, when plotting 'ALL' of CONUS, draw lines to show distinct regions
    if '_ALL' in GRIB_PATH_DES:
        plt.plot(np.array([-110,-110,-110]), np.array([20,36,52]),color='k',linewidth=3,zorder=3,alpha=0.5, transform=ccrs.PlateCarree())
        plt.plot(np.array([-85,-85,-85]), np.array([20,36,52]),color='k',linewidth=3,zorder=3,alpha=0.5, transform=ccrs.PlateCarree())
        plt.plot(np.array([-130,-97,-64]), np.array([38,38,38]),color='k',linewidth=3,zorder=3,alpha=0.5, transform=ccrs.PlateCarree())

        ax.text((1.0/6.0)-0.02, 0.95, 'North West Domain \n Click to Zoom in',
            transform=ax.transAxes, fontsize=11,verticalalignment='top',zorder=5,horizontalalignment='center',bbox=props2)
        ax.text(0.5, 0.95, 'North Central Domain \n Click to Zoom in',
            transform=ax.transAxes, fontsize=11,verticalalignment='top',zorder=5,horizontalalignment='center',bbox=props2)
        ax.text((5.0/6.0)+0.02, 0.95, 'North East Domain \n Click to Zoom in', 
            transform=ax.transAxes, fontsize=11,verticalalignment='top',zorder=5,horizontalalignment='center',bbox=props2)
                    
        ax.text((1.0/6.0)-0.02, 0.1, 'South West Domain \n Click to Zoom in',
            transform=ax.transAxes, fontsize=11,verticalalignment='top',zorder=5,horizontalalignment='center',bbox=props2)
        ax.text(0.5, 0.1, 'South Central Domain \n Click to Zoom in',
            transform=ax.transAxes, fontsize=11,verticalalignment='top',zorder=5,horizontalalignment='center',bbox=props2)
        ax.text((5.0/6.0)+0.02, 0.1, 'South East Domain \n Click to Zoom in',
            transform=ax.transAxes, fontsize=11,verticalalignment='top',zorder=5,horizontalalignment='center',bbox=props2)
            
    #Show and save the plot
    #plt.hold(True)
    plot_map([latlon_dims[1],latlon_dims[3]],[latlon_dims[0],latlon_dims[2]],ax)
    plt.show()
    plt.savefig(FIG_PATH_s+'/'+GRIB_PATH_DES+'_TOTENS_objprob_bymodel_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+ \
        '_t'+str(raw_thres)+'.png', bbox_inches='tight',dpi=150)
    plt.close()
    
    ##########################################################################################
    ###PLOT OF OBJ PROB. (SHADED), MODEL TYPE (MARKERTYPE), AND OBJ. FCST HR (MARKER COLOR)#
    ##########################################################################################
    
    #Create range of values to plot
    values = np.linspace(0,1,11)

    fig = plt.figure(figsize=(12,6)) #Create Figure
    ax  = plt.axes(projection=ccrs.Mercator(),alpha=1) #Create geoaxes to the projection
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    ax.text(0.5, 0.03, str(int(np.sum(np.mean(data_success[:,isolate_st4],axis=0)<1)))+' Members Found', 
        transform=ax.transAxes, fontsize=12,verticalalignment='bottom', horizontalalignment='center', bbox=props)
    ax.text(0.5, 0.97, "Valid Time: "+curdate_start.strftime('%Y%m%d%H%M')+" - "+curdate_stop.strftime('%Y%m%d%H%M'),
        transform=ax.transAxes, fontsize=12,verticalalignment='top', horizontalalignment='center', bbox=props)

    kwargs = {"transform_first": True}
    cs = plt.contourf(lon, lat, gaussian_filter(ens_mean*100, sigma),   levels=values*100, extend='min', cmap=my_cmap2, antialiased=False, vmin=np.spacing(0.0),\
        vmax=np.nanmax(values*100), transform=ccrs.PlateCarree(),**kwargs)
    cb = plt.colorbar(cs,ax=ax)

    #Outline the observation if it exists
    str_inc = [i for i in range(len(isolate_st4)) if isolate_st4[i] == 0]
    if sum(isolate_st4 == 0) > 0:
        kwargs = {"transform_first": True}
        plt.contour(lon, lat, np.squeeze(np.mean(simp_bin[:,:,:,str_inc],axis=2)) > 0,   levels=values, cmap=mpl.colors.ListedColormap(["#000000"]), linewidths=2, \
            transform=ccrs.PlateCarree(),**kwargs)
        
    #Scatter plot includes intensity (color) and member (symbol)
    for model in range(len(load_data_nc)):

        #Specify marker properly 
        if len(load_data_nc) == len(leg_nam_bymod):
            marker = leg_sym_bymod[model]    
        elif len(leg_sym_bymod) == 1:
            marker = leg_sym_bymod[0]
        else:
            for bymod in range(0,len(leg_nam_bymod)):
                if leg_nam_bymod[bymod] in load_data_nc[model]:
                    marker=leg_sym_bymod[bymod]
        
        #Find modeled objects in metadata
        if np.mean(np.isnan(simp_prop[model])) != 1:
            
            elements = np.unique(simp_prop[model][0])
            #Separate model and observations and set plotting properties
            if isolate_st4[model] == 0:
                elements   = elements[elements < 0]
                linecolor  = 'b'
                linewidth  = 1.5
                markersize = 60
                zorder     = 3
            else:
                elements = elements[elements > 0]
                linecolor  = 'k'
                linewidth  = 0.5
                markersize = 30
                zorder     = 2
    
            for tot_obj in elements: #Through each track
                ind = simp_prop[model][0] == tot_obj
                
                if len(simp_prop[model][5][ind]) > 2:
                    cs2= plt.plot(simp_prop[model][5][ind], simp_prop[model][4][ind], color='k', linewidth=linewidth, zorder=zorder, transform=ccrs.PlateCarree())
                    cs = plt.scatter(simp_prop[model][5][ind], simp_prop[model][4][ind], c = simp_prop[model][2][ind], \
                        marker=marker, vmin=0, vmax=hrs[-1], s=markersize, linewidth=0.25, cmap=my_cmap1_short, zorder=zorder, alpha=1, transform=ccrs.PlateCarree())  
                else:
                    cs2= plt.plot(simp_prop[model][5][ind][0], simp_prop[model][4][ind][0], color='k', linewidth=linewidth, zorder=zorder, transform=ccrs.PlateCarree())
                    cs = plt.scatter(simp_prop[model][5][ind][0], simp_prop[model][4][ind][0], c = simp_prop[model][2][ind][0], \
                        marker=marker, vmin=0, vmax=hrs[-1], s=markersize, linewidth=0.25, cmap=my_cmap1_short, zorder=zorder, alpha=1, transform=ccrs.PlateCarree())

                
        #END through each track
    #END through the models

    #Pass line instances to a legend to create centroid forecast hour legend by color
    line_label=[]
    line_str=[];
    for x in range(0,len(leg_nam_bytime)):
        line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color=colorlist[x], linestyle='None', marker='o', markersize=7)))
        line_str   = np.hstack((line_str, leg_nam_bytime[x]))
    first_legend=plt.legend(line_label, line_str, fontsize=6, numpoints=1, loc=3, framealpha=1)
    ax2 = plt.gca().add_artist(first_legend)
    
    #Pass line instances to a legend denoting member-type by markersize
    line_label=[]
    line_str=[];
    if len(leg_nam_bymod) > 0: #Only plot name legend if > 1 unique name exists
        for x in range(len(leg_sym_bymod)):
            line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color='k', linestyle='None', marker=leg_sym_bymod[x], markersize=7)))
            line_str   = np.hstack((line_str, leg_nam_bymod[x]))
        plt.legend(line_label, line_str, fontsize=6, numpoints=1, loc=4, framealpha=1)
    
    #Create the title
    ax.set_title("Ensemble Object Probabilities - "+"Precip. >= "+str(raw_thres)+"\" \n "+'{0:.2f}'.format(pre_acc)+ \
        " Hour Acc. Precip. - Initialized "+curdate.strftime('%Y%m%d%H'),fontsize=13)
        
    #Show and save the plot
    #plt.hold(True)
    plt.show()
    plot_map([latlon_dims[1],latlon_dims[3]],[latlon_dims[0],latlon_dims[2]],ax)
    plt.savefig(FIG_PATH_s+'/'+GRIB_PATH_DES+'_TOTENS_objprob_bytime_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+ \
        '_t'+str(raw_thres)+'.png', bbox_inches='tight',dpi=150)
    plt.close()

    #############################################################################################
    #ANIMATION OF OBJ PROB. (SHADED), MODEL TYPE (MARKERTYPE), AND OBJ. INTENSITY (MARKER COLOR)#
    #############################################################################################
    fig_name=[];
    
    hr_count = 0
    for hr_inc in hrs:
        
        #Determine time stamp for hour
        curdate_curr = curdate_currn + datetime.timedelta(hours=hr_inc)
        
        #Determine hour matrix that isolates models existing for that hour
        isolate_hr = np.array(isolate_st4 == 1) & np.array(data_success[hr_count,:] == 0)
       
        #Determine hourly object probabilities
        ens_mean = np.nanmean(simp_bin[:,:,hr_count,isolate_hr == 1], axis = 2)
        ens_mean[ens_mean > 0.95] = 0.95
        
        #Create range of values to plot
        values = np.linspace(0,1,11)

        fig = plt.figure(figsize=(12,6)) #Create Figure
        ax  = plt.axes(projection=ccrs.Mercator(),alpha=1) #Create geoaxes to the projection
        props = dict(boxstyle='round', facecolor='wheat', alpha=1)
        ax.text(0.5, 0.03, str(int(np.sum(isolate_hr == 1)))+' Members Found', 
            transform=ax.transAxes, fontsize=12,verticalalignment='bottom', horizontalalignment='center', bbox=props)
        ax.text(0.5, 0.97, "Valid Time: "+curdate_curr.strftime('%Y%m%d%H%M')+" - "+'{0:.2f}'.format(pre_acc)+' Hour Accumulation ',
            transform=ax.transAxes, fontsize=12,verticalalignment='top', horizontalalignment='center', bbox=props)
        kwargs = {"transform_first": False}
        cs = plt.contourf(lon, lat, gaussian_filter(ens_mean*100, sigma),   levels=values*100, extend='min', cmap=my_cmap2, antialiased=False, vmin=np.spacing(0.0),\
            vmax=np.nanmax(values*100), transform=ccrs.PlateCarree(),**kwargs)

        if "_ALL" in GRIB_PATH_DES:
	        geojson_filename = FIG_PATH_s+'/'+GRIB_PATH_DES+'_TOTENS_objprob_byhour_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+ \
	            '_t'+str(raw_thres)+'_f'+'{:02d}'.format(int(hr_count)+1)+'_shadedprobs.geojson'
	        geojsoncontour.contourf_to_geojson(
	            contourf=cs,
	            geojson_filepath=geojson_filename
	        )
 
        #Outline the observation if it exists
        str_inc = [i for i in range(len(isolate_st4)) if isolate_st4[i] == 0]
        if sum(isolate_st4 == 0) > 0:
            kwargs = {"transform_first": True}
            plt.contour(lon, lat, np.squeeze(simp_bin[:,:,hr_count,str_inc],axis=2) > 0,   levels=values, cmap=mpl.colors.ListedColormap(["#000000"]), linewidths=2, \
                transform=ccrs.PlateCarree(),**kwargs)
            
        #Create two plots: 1) includes intensity (color) and member (symbol) at time of concern
        #                  2) total track of object throughout time
        geojson_features = []
        for model in range(len(load_data_nc)):
      
            #Specify marker properly 
            if len(load_data_nc) == len(leg_nam_bymod):
                marker = leg_sym_bymod[model]   
            elif len(leg_sym_bymod) == 1:
                marker = leg_sym_bymod[0] 
            else:
                for bymod in range(0,len(leg_nam_bymod)):
                    if leg_nam_bymod[bymod] in load_data_nc[model]:
                        marker=leg_sym_bymod[bymod]
                
            #Find modeled/observed objects in metadata
            if np.mean(np.isnan(simp_prop[model])) != 1:
                
                #Isolate/identify specific hour object attributes
                hr_loc = simp_prop[model][2] == hr_inc
                
                #If data exists for that hour, isolate and plot
                if len(hr_loc) > 0:
                    elements = np.unique(simp_prop[model][0][hr_loc])
                    
                    #Separate model and observations and set plotting properties
                    if isolate_st4[model] == 0:
                        elements   = elements[elements < 0]
                        linecolor  = 'b'
                        linewidth  = 1.5
                        markersize = 60
                        zorder     = 3
                    else:
                        elements = elements[elements > 0]
                        linecolor  = 'k'
                        linewidth  = 0.5
                        markersize = 30
                        zorder     = 2
                                       
                    for tot_obj in elements: #Through each track
                        #Tracks for instantaneous time and total time
                        ind_snap = np.asarray(simp_prop[model][0] == tot_obj) & np.asarray(simp_prop[model][2] == hr_inc)
                        ind_all  = np.asarray(simp_prop[model][0] == tot_obj)

                        print("VMAX:", np.nanmax(centroid_values))
                        
                        if len(simp_prop[model][5][ind_snap]) > 2:
                            plt.plot(simp_prop[model][5][ind_snap], simp_prop[model][4][ind_snap], color=linecolor, linewidth=linewidth, zorder=zorder, \
                                transform=ccrs.PlateCarree())
                            plt.scatter(simp_prop[model][5][ind_snap], simp_prop[model][4][ind_snap], c = simp_prop[model][9][ind_snap], \
                                marker=marker, vmin=0, vmax=np.nanmax(centroid_values), s=markersize, linewidth=0.25, cmap=my_cmap1, zorder=zorder, \
                                alpha=1, transform=ccrs.PlateCarree())

                            geojson_point = geojson.Point([float(simp_prop[model][5][ind_snap]), float(simp_prop[model][4][ind_snap])])
                            geojson_properties = {
                            	"model": load_data_nc[model].replace('.nc',''), 
                            	"object_id": simp_prop[model][0][ind_snap],
                            	"object_cat:": simp_prop[model][1][ind_snap],
                            	"time_index:": simp_prop[model][2][ind_snap],
                            	"area": simp_prop[model][3][ind_snap],
                            	"axis_ang": simp_prop[model][6][ind_snap],
                            	"10th_percentile": simp_prop[model][7][ind_snap],
                            	"50th_percentile": simp_prop[model][8][ind_snap],
                            	"90th_percentile": simp_prop[model][9][ind_snap],
                            }
                            geojson_feature = geojson.Feature(geometry=geojson_point, properties=geojson_properties)
                            geojson_features.append(geojson_feature)

                        else:
                            plt.plot(simp_prop[model][5][ind_snap][0], simp_prop[model][4][ind_snap][0], color=linecolor, linewidth=linewidth, zorder=zorder, \
                                transform=ccrs.PlateCarree())
                            plt.scatter(simp_prop[model][5][ind_snap][0], simp_prop[model][4][ind_snap][0], c = simp_prop[model][9][ind_snap][0], \
                                marker=marker, vmin=0, vmax=np.nanmax(centroid_values), s=markersize, linewidth=0.25, cmap=my_cmap1, zorder=zorder, alpha=0.8, \
                                transform=ccrs.PlateCarree())

                            geojson_point = geojson.Point([float(simp_prop[model][5][ind_snap][0]), float(simp_prop[model][4][ind_snap][0])])
                            geojson_properties = {
                            	"model": load_data_nc[model].replace('.nc',''), 
                            	"object_id": simp_prop[model][0][ind_snap][0],
                            	"object_cat:": simp_prop[model][1][ind_snap][0],
                            	"time_index:": simp_prop[model][2][ind_snap][0],
                            	"area": simp_prop[model][3][ind_snap][0],
                            	"axis_ang": simp_prop[model][6][ind_snap][0],
                            	"10th_percentile": simp_prop[model][7][ind_snap][0],
                            	"50th_percentile": simp_prop[model][8][ind_snap][0],
                            	"90th_percentile": simp_prop[model][9][ind_snap][0],
                            }
                            geojson_feature = geojson.Feature(geometry=geojson_point, properties=geojson_properties)
                            geojson_features.append(geojson_feature)

                                
                        if len(simp_prop[model][5][ind_all]) > 2:
                            plt.plot(simp_prop[model][5][ind_all], simp_prop[model][4][ind_all], color=linecolor, linewidth=linewidth-0.25, zorder=zorder, \
                                transform=ccrs.PlateCarree())

                            geojson_line = geojson.LineString([[float(lon), float(lat)] for lon, lat in zip(simp_prop[model][5][ind_all],simp_prop[model][4][ind_all])])
                            geojson_feature = geojson.Feature(geometry=geojson_line, properties={"model": load_data_nc[model].replace('.nc','')})
                            geojson_features.append(geojson_feature)

                        else:
                            plt.plot(simp_prop[model][5][ind_all][0], simp_prop[model][4][ind_all][0], color=linecolor, linewidth=linewidth-0.25, zorder=zorder, \
                                transform=ccrs.PlateCarree())

                            geojson_line = geojson.LineString([[float(simp_prop[model][5][ind_all][0]), float(simp_prop[model][4][ind_all][0])]])
                            geojson_feature = geojson.Feature(geometry=geojson_line, properties={"model": load_data_nc[model].replace('.nc','')})
                            geojson_features.append(geojson_feature)
                                    
                    #END through each track 
                #END check for hourly data existing
            #END model VS obs check
        #END through the models

        #Create geojson feature collection from array of track linestring features & save to file
        if "_ALL" in GRIB_PATH_DES:
        	print(geojson_features)
	        geojson_feature_collection = geojson.FeatureCollection(geojson_features)
	        geojson_filename = FIG_PATH_s+'/'+GRIB_PATH_DES+'_TOTENS_objprob_byhour_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+ \
	                '_t'+str(raw_thres)+'_f'+'{:02d}'.format(int(hr_count)+1)+'_tracks.geojson'
	        geojson_object = geojson.dumps(geojson_feature_collection)
	        with open(geojson_filename, "w") as outfile:
	                outfile.write(geojson_object)
    
        #Pass line instances to a legend to create centroid maximum legend by color
        line_label=[]
        line_str=[];
        for x in range(len(colorlist)):
            line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color=colorlist[x], linestyle='None', marker='o', markersize=7)))
            line_str   = np.hstack((line_str, leg_nam_byint[x]))
        first_legend=plt.legend(line_label, line_str, fontsize=6, numpoints=1, loc=3, framealpha=1)
        ax2 = plt.gca().add_artist(first_legend)
        
        #Pass line instances to a legend denoting member-type by markersize
        line_label=[]
        line_str=[];
        if len(leg_nam_bymod) > 0: #Only plot name legend if > 1 unique name exists
            for x in range(len(leg_sym_bymod)):
                line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color='k', linestyle='None', marker=leg_sym_bymod[x], markersize=7)))
                line_str   = np.hstack((line_str, leg_nam_bymod[x]))
            plt.legend(line_label, line_str, fontsize=6, numpoints=1, loc=4, framealpha=1)
        
        #Create the title
        cb = plt.colorbar(cs,ax=ax)
        ax.set_title("Ensemble Object Probabilities - "+"Precip. >= "+str(raw_thres)+"\" \n "+'{0:.2f}'.format(pre_acc)+ \
            " Hour Acc. Precip. - Initialized "+curdate.strftime('%Y%m%d%H'),fontsize=13)
            
        #Show and save the plot
        #plt.hold(True)
        plt.show()
        plot_map([latlon_dims[1],latlon_dims[3]],[latlon_dims[0],latlon_dims[2]],ax)
        plt.savefig(FIG_PATH_s+'/'+GRIB_PATH_DES+'_TOTENS_objprob_byhour_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+ \
            '_t'+str(raw_thres)+'_f'+'{:02d}'.format(int(hr_count)+1)+'.png', bbox_inches='tight',dpi=150)
        plt.close()
        
        #Update figure names to create animated figure
        fig_name=np.append(fig_name,FIG_PATH_s+'/'+GRIB_PATH_DES+'_TOTENS_objprob_byhour_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+ \
            '_t'+str(raw_thres)+'_f'+'{:02d}'.format(int(hr_count)+1)+'.png')
   
        hr_count += 1
 
    #Create an animation from the figures previously saved
    os.system('convert -delay 100 '+' '.join(fig_name)+' -loop 0 '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_TOTENS_objprob_byhour_'+ \
        curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'.gif')

    #Copy to a non-date specific name for website dynamic thumbnail
    os.system('cp '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_TOTENS_objprob_byhour_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+\
        '{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'.gif '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_TOTENS_objprob_byhour_p'+\
        '{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'.gif')

    #Convert data from fraction of an hour back to increments
    if 'NEWSe5min' in ''.join(load_data_nc) or 'HRRRv415min' in ''.join(load_data_nc):
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = (simp_prop[model][2] - 0.25) * 4
    else:
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = simp_prop[model][2] - 1
            
    return None

###################################################################################
#MTDPlotTLEFcst identifies any time lagged ensembles and plots each one separately
#with color/location representing centroid location/magnitude, and marker type representing
#each member in the TLE. This code will plot an unlimited number of TLE's with up to 9
#members within each TLE (restricted by marker type legend). The goal is to assess any
#potential trends in the plots. QPE is also plotted if requested with object shape in
#black outline and color/location representing centroid location/magnitude.
#
#Created by MJE. 20170328-20170420
#Updated to include adjustable smoother. 20190604. MJE
#Updated to include NEWSe (in 2019) and HRRR 15 min. plotting. 20201218. MJE
#
#####################INPUT FILES FOR MODEPlotAll####################################
#GRIB_PATH_DES   = unique identifier assigned to figure name
#FIG_PATH_s      = path where figures are to be saved
#latlon_dims     = latitude/longitude dimensions for plotting [WLON,SLAT,ELON,NLAT]
#pre_acc         = accumulation interval
#hrs             = array of forecast hours
#raw_thres       = raw threshold for MODE
#curdate         = datetime attribute
#data_success    = record of which data were succesfully loaded
#load_data_nc    = list of model/obs data to be loaded
#lat             = gridded latitude data
#lon             = gridded longitude data
#simp_bin        = gridded simple data
#simp_prop       = simple cluster centroid info of lat,lon,max. Takes the form of [model][attributes]
#sigma           = sigma value for Gaussian smoother (radius of smoothing in grid points)
####################OUTPUT FILES FOR MODEPlotAllFcst####################################
#Note: "Simple" means unmerged while "cluster" refers to merged
#"Single" refers to unmatched objects while "pair" refers to matched objects.
###################################################################################

def mtd_plot_tle_fcst(GRIB_PATH_DES,FIG_PATH_s,latlon_dims,pre_acc,hrs,raw_thres,curdate,data_success,load_data_nc,lat,lon,simp_bin,simp_prop,sigma):

    #latlon_dims   = latlon_sub[subsets]
    #GRIB_PATH_DES = GRIB_PATH_DES+domain_sub[subsets]
    #raw_thres     = thres

    #Set proper time index depending on ensemble
    if 'NEWSe5min' in ''.join(load_data_nc) or 'HRRRv415min' in ''.join(load_data_nc):
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = (simp_prop[model][2] / 4.0) + 0.25
    else:
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = simp_prop[model][2] + 1

    #Remove ST4 data since ST4 objects are not plotted here
    isolate_st4=[]
    for i in range(0,len(load_data_nc),1):
        if 'ST4' in load_data_nc[i]:
            isolate_st4 = np.append(isolate_st4,False)
        else:
            isolate_st4 = np.append(isolate_st4,True)

    isolate_st4      = isolate_st4 == 1
    isolate_st4      = np.array(isolate_st4 == 1)

    #Currently this function only requires a logical matrix
    simp_bin = simp_bin > 0

    #Add NaNs to missing data in binary matrix. Convert to float
    simp_bin=simp_bin.astype(float)
    NaN_loc = (data_success == 1).astype(float)
    NaN_loc[NaN_loc == 1] = np.NaN

    for j in range(simp_bin.shape[0]):
        for i in range(simp_bin.shape[1]):
            simp_bin[j,i,:,:] = simp_bin[j,i,:,:] + NaN_loc

    #Compute the proper forecast hour strings
    curdate_currn = curdate + datetime.timedelta(hours=0)
    curdate_start = curdate + datetime.timedelta(hours=hrs[0])
    curdate_stop  = curdate + datetime.timedelta(hours=hrs[-1])

    #Establish a smoothing kernel to smooth ensemble probaiblities
    kern=np.hanning(50)
    kern /= kern.sum()

    #Create colorlist attributes and legend information if possible
    colorlist = ["#ffffff", "#02fd02", "#01c501", "#228b22", "#9acd32", "#fdf802", "#f8d826", "#e5bc00", "#fd9500", "#fd0000", "#bc0000"]
    my_cmap1 = mpl.colors.ListedColormap(colorlist)

    #Second colorlist is a trimmed default python Blues
    interval = np.linspace(0.3, 1)
    colors = plt.cm.Blues(interval)
    my_cmap2 = LinearSegmentedColormap.from_list('name', colors)
    my_cmap2.set_under('white')

    centroid_values = np.round(np.linspace(0,1*10*raw_thres,len(colorlist)),2)
    leg_nam_byint  = ['90th Object Percentile >='+str(x)+"\"" for x in centroid_values]
    if sum(isolate_st4 == 0) > 0: #If obs exists, change legend
        leg_sym_bymem  = ['s','d','o','*','^','1','2','3','4']
        mod_inc = 1 #When plotting marker, add 1 for models if st4 exists
    else:
        leg_sym_bymem  = ['d','o','*','^','1','2','3','4']
        mod_inc = 0 #If ST4 does not exist, no offset needed

    #Find unique models with different lags
    load_data_short = [load_data_nc[i][:-9] for i in range(0,len(load_data_nc))]
    load_data = [load_data_nc[i][:-3] for i in range(0,len(load_data_nc))]
    lag = np.array([int(load_data[i][load_data[i].index("lag")+3::]) for i in range(0,len(load_data))])
    [u, indices] = np.unique(load_data_short, return_inverse=True)

    #Loop through each unique model grouping. Create a separate plot for each TLE
    ind_count = 0
    for ind in range(0,len(u),1):

        #Initialize variables
        simp_prop_k        = [[] for i in range(0,20,1)]
        leg_sym_use        = []

        #Consider the current TLE and subset data
        current_ind        = indices == ind
        data_success_p     = data_success[:,current_ind]
        load_data_nc_p     = [load_data_nc[i] for i in range(0,int(len(current_ind))) if current_ind[i] == True]
        simp_bin_p         = simp_bin[:,:,:,current_ind]
        simp_prop_p        = [simp_prop[i] for i in range(0,int(len(current_ind))) if current_ind[i] == True]
        lag_p              = lag[current_ind]

        if sum(indices==ind) > 1 and np.mean(data_success_p) != 1: #check to see if a TLE exists with data

            #Sort the current TLE from newest to oldest
            old2new = np.argsort(lag_p)

            #Cat all TLEs by model
            data_success_k     = data_success_p[:,old2new[::-1]]
            load_data_nc_k     = [load_data_nc_p[old2new[i]] for i in range(int(len(old2new))-1,-1,-1)]
            #simp_bin_k         = np.squeeze(simp_bin_p[:,:,:,old2new[::-1]])
            count_elem         = 0

            for model in range(int(len(old2new))-1,-1,-1):
                if np.mean(np.isnan(simp_prop_p[old2new[model]])) != 1:
                    simp_prop_k[count_elem] = simp_prop_p[old2new[model]][:,simp_prop_p[old2new[model]][0,:]>0]
                else:
                    simp_prop_k[count_elem] = np.nan

                count_elem += 1

            #Compute raw ensemble mean probability, set 100% to 95% to preserve colorbar
            ens_mean = np.nanmax(np.nanmean(simp_bin_p, axis = 3) , axis = 2)
            ens_mean[ens_mean > 0.95] = 0.95

            #Legend information for time lagged ensembles
            if sum(isolate_st4 == 0) > 0:
                leg_nam_bymem = np.append('ST4',[str(i)+' Hours Lag' for i in lag_p])
            else:
                leg_nam_bymem = np.array([str(i)+' Hours Lag' for i in lag_p])

            #Create range of values to plot and colorbar
            values = np.linspace(0,1,11)

            fig = plt.figure(figsize=(12,6)) #Create Figure
            ax  = plt.axes(projection=ccrs.Mercator(),alpha=1) #Create geoaxes to the projection
            props  = dict(boxstyle='round', facecolor='wheat', alpha=1)

            ax.text(0.5, 0.03, load_data_nc_k[0][:-9]+' - '+str(int(np.sum(np.mean(data_success_k==0,axis=0)>0)))+' Members Found',
                transform=ax.transAxes, fontsize=12,verticalalignment='bottom', horizontalalignment='center', bbox=props)
            ax.text(0.5, 0.97, "Valid Time: "+curdate_start.strftime('%Y%m%d%H%M')+" - "+curdate_stop.strftime('%Y%m%d%H%M'),
                transform=ax.transAxes, fontsize=12,verticalalignment='top', horizontalalignment='center', bbox=props)
            #norm = mpl.colors.BoundaryNorm(values, 0)

            #Plot smoothed probability of observation within 40 km of a point given model
            kwargs = {"transform_first": True}
            cs = plt.contourf(lon, lat, gaussian_filter(ens_mean*100, sigma),   \
                levels=values*100, extend='min', cmap=my_cmap2, antialiased=False, vmin=np.spacing(0.0),vmax=np.nanmax(values)*100, transform=ccrs.PlateCarree(),**kwargs)

            #If observation exists, plot outline
            if sum(isolate_st4 == 0) > 0:
                kwargs = {"transform_first": True}
                plt.contour(lon, lat, np.squeeze(np.mean(simp_bin[:,:,:,isolate_st4 == 0],axis=2)) > 0,   levels=values, cmap=mpl.colors.ListedColormap(["#000000"]), \
                    linewidths=2, transform=ccrs.PlateCarree(),**kwargs)

            #Plot the raw TLE objects
            line_label=[]
            line_str=[]

            #Array to properly index marker legend
            marker_inc = range(int(len(old2new))-1,-1,-1)

            #Scatter plot the observation if it exists
            if sum(isolate_st4 == 0) > 0:

                #Element location for observation
                obs_inc = [int(i) for i in range(len(isolate_st4)) if isolate_st4[i] == 0]
                obs_inc = obs_inc[0]

                #Find observed objects in metadata
                if np.mean(np.isnan(simp_prop[obs_inc])) != 1:

                    elements = np.unique(simp_prop[obs_inc][0])
                    elements = elements[elements < 0]

                    for tot_obj in elements: #Through each track
                        obj = simp_prop[obs_inc][0] == tot_obj

                        if len(simp_prop[obs_inc][5][obj]) > 2:
                            plt.plot(simp_prop[obs_inc][5][obj], simp_prop[obs_inc][4][obj], color='b', linewidth=1.5, zorder = 3, transform=ccrs.PlateCarree())
                            plt.scatter(simp_prop[obs_inc][5][obj], simp_prop[obs_inc][4][obj], c = simp_prop[obs_inc][9][obj], \
                                marker=leg_sym_bymem[obs_inc], vmin=0, vmax=np.nanmax(centroid_values), s=60, linewidth=0.25, cmap=my_cmap1, zorder = 3, alpha = 0.8, \
                                transform=ccrs.PlateCarree())
                        else:
                            plt.plot(simp_prop[obs_inc][5][obj][0], simp_prop[obs_inc][4][obj][0], color='b', linewidth=1.5, zorder = 3, transform=ccrs.PlateCarree())
                            plt.scatter(simp_prop[obs_inc][5][obj][0], simp_prop[obs_inc][4][obj][0], c = simp_prop[obs_inc][9][obj][0], \
                                marker=leg_sym_bymem[obs_inc], vmin=0, vmax=np.nanmax(centroid_values), s=60, linewidth=0.25, cmap=my_cmap1, zorder = 3, alpha = 0.8, \
                                transform=ccrs.PlateCarree())
                    leg_sym_use = np.append(leg_sym_use,leg_sym_bymem[obs_inc])

            #Scatter plot includes intensity (color) and member (symbol)
            for model in range(len(load_data_nc_k)):

                #Find modeled objects in metadata
                if np.mean(np.isnan(simp_prop_k[model])) != 1:

                    elements = np.unique(simp_prop_k[model][0])
                    elements = elements[elements >= 0]

                    for tot_obj in elements: #Through each track
                        obj = simp_prop_k[model][0] == tot_obj

                        if len(simp_prop_k[model][5][obj]) > 2:
                            plt.plot(simp_prop_k[model][5][obj], simp_prop_k[model][4][obj], color='k', linewidth=0.5, zorder = 2, transform=ccrs.PlateCarree())
                            plt.scatter(simp_prop_k[model][5][obj], simp_prop_k[model][4][obj], c = simp_prop_k[model][9][obj], \
                                marker=leg_sym_bymem[marker_inc[model]+mod_inc], vmin=0, vmax=np.nanmax(centroid_values), s=30, linewidth=0.25, cmap=my_cmap1, zorder = 2, \
                                alpha = 0.8, transform=ccrs.PlateCarree())
                        else:
                            plt.plot(simp_prop_k[model][5][obj][0], simp_prop_k[model][4][obj][0], color='k', linewidth=0.5, zorder = 2, transform=ccrs.PlateCarree())
                            plt.scatter(simp_prop_k[model][5][obj][0], simp_prop_k[model][4][obj][0], c = simp_prop_k[model][9][obj][0], \
                                marker=leg_sym_bymem[marker_inc[model]+mod_inc], vmin=0, vmax=np.nanmax(centroid_values), s=30, linewidth=0.25, cmap=my_cmap1, zorder = 2, \
                                alpha = 0.8, transform=ccrs.PlateCarree())
                leg_sym_use = np.append(leg_sym_use,leg_sym_bymem[marker_inc[model]+mod_inc])
            #END through model to create scatterplot
            #plt.hold(True)



            #Pass line instances to a legend to create centroid maximum legend by color
            line_label = []
            line_str   = []
            for x in range(len(colorlist)):
                line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color=colorlist[x], linestyle='None', marker='o', markersize=5)))
                line_str   = np.hstack((line_str, leg_nam_byint[x]))
            first_legend  = plt.legend(line_label, line_str, fontsize=6, numpoints=1, loc=3, framealpha=1)
            plt.gca().add_artist(first_legend)

            #Create legend of markertype by member
            line_label = []
            line_str   = []
            leg_count  = 0
            for x in range(np.sum(np.mean(data_success_k,axis=0) < 1) + mod_inc):
                line_label = np.hstack((line_label, mpl.lines.Line2D(range(1), range(1), color='k', linestyle='None', marker=leg_sym_use[x], markersize=5)))
                line_str   = np.hstack((line_str, leg_nam_bymem[x]))
                leg_count += 1
            second_legend = plt.legend(line_label, line_str, fontsize=6, numpoints=1, loc=4, framealpha=1)
            plt.gca().add_artist(second_legend)

            cb = plt.colorbar(cs,ax=ax)
            ax.set_title("Objects/Max. Locations - "+"Precip. >= "+str(raw_thres)+"\" \n "+'{0:.2f}'.format(pre_acc)+ \
                " Hour Acc. Precip. - Initialized "+curdate.strftime('%Y%m%d%H'),fontsize=13)
            #plt.hold(True)
            plot_map([latlon_dims[1],latlon_dims[3]],[latlon_dims[0],latlon_dims[2]],ax)
            plt.show()
            plt.savefig(FIG_PATH_s+'/'+GRIB_PATH_DES+'_TLE_obj_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+ \
            '_t'+str(raw_thres)+'_group'+str(ind_count)+'.png', bbox_inches='tight',dpi=150)
            plt.close()

            ind_count += 1

    #Convert data from fraction of an hour back to increments
    if 'NEWSe5min' in ''.join(load_data_nc) or 'HRRRv415min' in ''.join(load_data_nc):
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = (simp_prop[model][2] - 0.25) * 4
    else:
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = simp_prop[model][2] - 1


###################################################################################
#mtd_plot_all_snow_fcst creates three plots: 1) forecast ensemble object probability (shading) and 
#object centroid location/magnitude/model type (marker). If observation exists then the
#observed object is plotted (black outline) in addition to centroid location/magnitude.
#2) forecast ensemble object probability (shading) and object centroid location/forecast 
#hour/model type (marker). If observation exists, observed object is plotted (black outline)
#in addition to centroid location/magnitude. 3) An hourly animation of figure 1) is created
#to better convey temporal details of the objects.
#
#Created by MJE. 20170327-20170420
#Updated to include NEWSe (in 2019) and HRRR 15 min. plotting. 20201218. MJE
#
#####################INPUT FILES FOR MTDPlotAllSnowFcst####################################
#GRIB_PATH_DES   = unique identifier assigned to figure name
#FIG_PATH_s      = path where figures are to be saved
#latlon_dims     = latitude/longitude dimensions for plotting [WLON,SLAT,ELON,NLAT]
#pre_acc         = accumulation interval
#hrs             = array of forecast hours
#raw_thres       = raw threshold for MODE
#curdate         = datetime attribute
#data_success    = record of which data were succesfully loaded
#load_data_nc    = list of model/obs data to be loaded
#lat             = gridded latitude data
#lon             = gridded longitude data
#simp_bin        = gridded simple data
#simp_prop       = simple cluster centroid info of lat,lon,max. Takes the form of [model][attributes]
####################OUTPUT FILES FOR MODEPlotAllFcst####################################
#Note: "Simple" means unmerged while "cluster" refers to merged
#"Single" refers to unmatched objects while "pair" refers to matched objects.
###################################################################################

def mtd_plot_all_snow_fcst(GRIB_PATH_DES,FIG_PATH_s,latlon_dims,pre_acc,hrs,raw_thres,curdate,data_success,load_data_nc,lat,lon,simp_bin,simp_prop):

    #data_success  = data_success_new
    #simp_bin      = simp_bin
    #simp_prop     = simp_prop
    #latlon_dims   = latlon_sub[subsets]
    #GRIB_PATH_DES = GRIB_PATH_DES+domain_sub[subsets]
    #raw_thres     = thres

    #Set proper time index depending on ensemble
    if 'NEWSe5min' in ''.join(load_data_nc) or 'HRRRv415min' in ''.join(load_data_nc):
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = (simp_prop[model][2] / 4.0) + 0.25
    else:
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = simp_prop[model][2] + 1
        
    #Find and isolate ST4 data
    isolate_st4=[]
    for i in range(0,len(load_data_nc),1):
        if 'ST4' in load_data_nc[i]:
            isolate_st4 = np.append(isolate_st4,False)
        else:
            isolate_st4 = np.append(isolate_st4,True)
    isolate_st4       = isolate_st4 == 1

    #Add NaNs to missing data in binary matrix. Convert to float
    simp_bin=simp_bin.astype(float)
    NaN_loc = (data_success == 1).astype(float)
    NaN_loc[NaN_loc == 1] = np.NaN
    
    for j in range(simp_bin.shape[0]):
        for i in range(simp_bin.shape[1]):
            simp_bin[j,i,:,:] = simp_bin[j,i,:,:] + NaN_loc
            
    #Isolate model only to plot outline of all objects
    simp_bin_mod = simp_bin[:,:,:,isolate_st4]
    
    #Compute the proper forecast hour strings
    curdate_currn = curdate + datetime.timedelta(hours=0)
    
    #Establish a smoothing kernel to smooth ensemble probaiblities
    kern=np.hanning(50)
    kern /= kern.sum() 
    
    #Create first colorlist
    colorlist = ["#ffffff", "#95FF9F", "#0BC31D", "#006400", "#FAE94C", "#F5AD31", "#FF4500", "#CC0000", "#8B0000", "#4B0082", "#FF00FF"]
    #["#ffffff", "#02fd02", "#01c501", "#228b22", "#9acd32", "#fdf802", "#f8d826", "#e5bc00", "#fd9500", "#fd0000", "#bc0000"]
    my_cmap1 = mpl.colors.ListedColormap(colorlist)

    #Given a default model name list, build a list based on user specified models
    leg_nam_bymod_default = ['ST4'  ,'HIRESNAM','HIRESW','HRRR','NSSL-OP','NSSL-ARW','NSSL-NMB','HRAM']
    leg_sym_bymod_default = ['s','d','*','^','<','>','v','o']
    
    if len(load_data_nc) <= 8: #If list is small enough, specific markers for individual models
        leg_nam_bymod  = load_data_nc
        leg_sym_bymod  = leg_sym_bymod_default[0:len(load_data_nc)]
    else:                      #Create markers by model type
        leg_nam_bymod = []
        leg_sym_bymod = ['d']
    
    centroid_values = np.round(np.linspace(0,1*10*raw_thres,len(colorlist)),2)
    leg_nam_byint  = ['>='+str(x)+"\"" for x in centroid_values]
    leg_nam_bytime = ['Hours '+str(hrs[i])+' - '+str(hrs[i+int(round(len(hrs)/len(colorlist)))]) \
        for i in range(0,len(hrs)-int(round(len(hrs)/len(colorlist))),int(round(len(hrs)/len(colorlist))+1))]
    
    #Shorten initial colorlist for bytime plots
    my_cmap1_short = mpl.colors.ListedColormap(colorlist[0:len(leg_nam_bytime)])
       
    #Create second colorlist, replace first element with white
    my_cmap2 = copy.copy( mpl.cm.get_cmap('cool'))
    my_cmap2.set_under('white')
    
    values = np.linspace(0,0.55,12)
 
    ##########################################################################################
    ########################CREATE HOURLY PLOTS OF SNOWBAND OBJECTS###########################
    ##########################################################################################
    #['OBJ_ID','CLUS_ID','TIME_INDEX','AREA','CENTROID_LAT','CENTROID_LON','AXIS_ANG','INTENSITY_90'] 
    
    fig_name = []
    hr_count = 0
    for hr_inc in hrs:

        #Create datestring for forecast hour
        curdate_cur = curdate + datetime.timedelta(hours=hr_inc)
        
        #Create range of values to plot
        values = np.linspace(0,0.55,12)

        fig = plt.figure(figsize=(12,6)) #Create Figure
        ax  = plt.axes(projection=ccrs.Mercator(),alpha=1) #Create geoaxes to the projection
        props  = dict(boxstyle='round', facecolor='wheat', alpha=1)
        props2 = dict(boxstyle='round', facecolor='tan', alpha=1)

        ax.text(0.5, 0.03, str(int(np.sum((data_success[hr_count,isolate_st4]==0)*1,axis=0)))+' Members Found', 
            transform=ax.transAxes, fontsize=12,verticalalignment='bottom', horizontalalignment='center', bbox=props)
        ax.text(0.5, 0.97, "Valid Time: "+curdate_cur.strftime('%Y%m%d%H%M'),
            transform=ax.transAxes, fontsize=12,verticalalignment='top', horizontalalignment='center', bbox=props)

        #plt.savefig('/export/hpc-lw-dtbdev5/merickson/Desktop/test1.png',bbox_inches='tight',dpi=150)
        #os.system('scp /export/hpc-lw-dtbdev5/merickson/Desktop/test1.png hpc@wpcrzdm:/home/people/hpc/ftp/erickson/test15')
       
        for model in range(0,simp_bin_mod.shape[3]):
            
            if np.sum(simp_bin_mod[:,:,hr_count,model]) == 0: #If no data, insert single value point
                simp_bin_mod[0,0,hr_count,model] = 1
                    
            #Capture 90th percentile of object intensity through time matching
            try:
                inten  = simp_prop[model][9][simp_prop[model][2] == hr_inc]
                obj_id = np.unique(simp_bin_mod[:,:,hr_count,model])
                
            except TypeError:
                inten  = [0]
                obj_id = [0,0]
            
            if len(inten) == 0:
                inten  = [0]
                obj_id = [0,0]
           
            #Loop through the objects
            obj_count = 1
            for obj in inten:

                colorlist = ["#ffffff", "#95FF9F", "#0BC31D", "#006400", "#FAE94C", "#F5AD31", "#FF4500", "#CC0000", "#8B0000", "#4B0082", "#FF00FF"] 
                if   obj >= 0.00 and obj < 0.05: 
                    c_outline = "#ffffff"
                elif obj >= 0.05 and obj < 0.10:
                    c_outline = "#95FF9F"
                elif obj >= 0.10 and obj < 0.15:
                    c_outline = "#0BC31D"
                elif obj >= 0.15 and obj < 0.20:
                    c_outline = "#006400"
                elif obj >= 0.20 and obj < 0.25:
                    c_outline = "#FAE94C"
                elif obj >= 0.25 and obj < 0.30:
                    c_outline = "#F5AD31"
                elif obj >= 0.30 and obj < 0.35:
                    c_outline = "#FF4500"
                elif obj >= 0.35 and obj < 0.40:
                    c_outline = "#CC0000"
                elif obj >= 0.40 and obj < 0.45:
                    c_outline = "#8B0000"
                elif obj >= 0.45 and obj < 0.50:
                    c_outline = "#4B0082"
                elif obj >= 0.50:
                    c_outline = "#FF00FF"
                       
                try: #Catch error if plotting objects outside of regional subset
                    #Contour creates bad legends, so use contour plot outline but contourf attributes for legend
                    kwargs = {"transform_first": True}
                    cs1 = plt.contourf(lon, lat, (simp_bin_mod[:,:,hr_count,model]==obj_id[obj_count]*1)*0,   levels=values,  cmap=my_cmap1, \
                        antialiased=False, vmin=np.nanmin(values),vmax=np.nanmax(values), transform=ccrs.PlateCarree(),**kwargs)
                    cs2 = plt.contour(lon, lat, (simp_bin_mod[:,:,hr_count,model]==obj_id[obj_count]*1),   antialiased=False, colors=c_outline,linewidths=0.8, \
                        transform=ccrs.PlateCarree(),**kwargs)
                except:
                    pass
                    
                obj_count +=1

        #Create the title
        cb = plt.colorbar(cs1,ax=ax)
        ax.set_title("All Snow Precip. Objects >= "+str(raw_thres)+"\" \n "+'{0:.2f}'.format(pre_acc)+ \
            " Hour Acc. Precip. at Hour "+'{0:.2f}'.format(hr_inc)+" - Initialized "+curdate.strftime('%Y%m%d%H'),fontsize=13)

        plot_map([latlon_dims[1],latlon_dims[3]],[latlon_dims[0],latlon_dims[2]],ax)
        plt.show()
        plt.savefig(FIG_PATH_s+'/'+GRIB_PATH_DES+'_snowobj_byhour_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+ \
            '_t'+str(raw_thres)+'_h'+'{:02d}'.format(int(hr_count)+1)+'.png', bbox_inches='tight',dpi=150)
        plt.close()

        #Update figure names to create animated figure
        fig_name=np.append(fig_name,FIG_PATH_s+'/'+GRIB_PATH_DES+'_snowobj_byhour_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+ \
            '_t'+str(raw_thres)+'_h'+'{:02d}'.format(int(hr_count)+1)+'.png')

        hr_count += 1
      
    #Create an animation from the figures previously saved
    os.system('convert -delay 100 '+' '.join(fig_name)+' -loop 0 '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_snowobj_byhour_'+ \
        curdate_currn.strftime('%Y%m%d%H')+'_p'+'{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'.gif')

    #Copy to a non-date specific name for website dynamic thumbnail
    os.system('cp '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_snowobj_byhour_'+curdate_currn.strftime('%Y%m%d%H')+'_p'+\
        '{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'.gif '+FIG_PATH_s+'/'+GRIB_PATH_DES+'_snowobj_byhour_p'+\
        '{0:.2f}'.format(pre_acc)+'_t'+str(raw_thres)+'.gif')
 
    #Convert data from fraction of an hour back to increments 
    if 'NEWSe5min' in ''.join(load_data_nc) or 'HRRRv415min' in ''.join(load_data_nc):
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = (simp_prop[model][2] - 0.25) * 4
    else:
        for model in range(0,len(load_data_nc)):
            simp_prop[model][2] = simp_prop[model][2] - 1
