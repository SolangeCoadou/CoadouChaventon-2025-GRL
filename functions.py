#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 10:53:10 2026

@author: solangecoadou
"""

import numpy as np 
import seawater as sw
from sklearn.linear_model import LinearRegression
from scipy.interpolate import PchipInterpolator
from shapely.geometry import LineString
from scipy.signal import argrelextrema
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import cmocean 
import xarray as xr 
from datetime import datetime
from datetime import timedelta  

import sys
sys.path.insert(1, '/Users/solangecoadou/Desktop/Desktop/Thesis/mcssa/')
from mcssa import MCSSA

#############################
# Functions for Figure_1 #
#############################

def deriv_2D(field, lat, lon, axis):
    
    """
    Compute the spatial derivative of a 2D array in longitude/latitude coordinate
    along one dimension using central difference.

    Parameters:
    ----------
    field : array_like (M x N)
        2D array of the data to be derived.
    lat : array_like (M x N)
        2D array of the latitude coordinates.
    lon : array_like (M x N)
        2D array of the longitude coordinates.
    axis : float
        Axis along which the derivative is computed:
    
    Returns:
    --------
    grad_field : array_like (M-2 x N-2)
        2D array of the spatial derivative of field along the given 
        axis
    """
        
    if axis == 0:
        grad_field = np.zeros([len(field[:,0])-2,len(field[0,:])-2])
        for i in range(1,len(field[:,0])-1):
            dist_points = sw.dist(lat[i,:], lon[i,:], units='km')[0]
            dist_2points = dist_points[:-1] + dist_points[1:]
            grad_field[i-1,:] = (field[i,2:] - field[i,:-2]) / (1000 * dist_2points)
    elif axis == 1:
        grad_field = np.zeros([len(field[:,0])-2,len(field[0,:])-2])
        for i in range(1,len(field[0,:])-1):
            dist_points = sw.dist(lat[:,i], lon[:,i], units='km')[0]
            dist_2points = dist_points[:-1] + dist_points[1:]
            grad_field[:,i-1] = (field[2:,i] - field[:-2,i]) / (1000 * dist_2points)
            
    return grad_field

def Rossby_DUACS(ds_sat, lat, lon):
    
    """
    Compute the relative vorticity or Rossby number from an xarray

    Parameters:
    ----------
    ds_sat : xarray
        xarray containing the geostrophic velocities asugos and vgos
    lat : array_like
        2D array of the latitude coordinates.
    lon : array_like
        2D array of the longitude coordinates.
    Returns:
    --------
        : array_like
        2D array of the relative vorticity
    """

    grad_y_u_fonction = deriv_2D(ds_sat.ugos.data.compute(), lat, lon, 1)
    grad_x_v_fonction = deriv_2D(ds_sat.vgos.data.compute(), lat, lon, 0)
    fcoriolis = sw.f(lat[1:-1,1:-1])
    
    return (grad_x_v_fonction-grad_y_u_fonction)/fcoriolis

def diameter_from_contour(dat0):
    
    """
    Compute an approximate diameter from an array containing
    a list of points in longitude-latitude cordinate. The function 
    looks, for each point, at the maximum distance to another point
    and takes the median of these distances as an estimate of the
    diameter of the shape defined by this list of points

    Parameters:
    ----------
    dat0 : 2D array (M x 2)
        array containing longitude-latitude cordinates of a list
        of points
    Returns:
    --------
    np.median(dist_list) : float
        estimate of the diameter of the shape defined by dat0
    """
    
    nb_points = len(dat0[:,0])
    dist_list = []
    for i in range(nb_points):
        ref_x, ref_y = dat0[i,0], dat0[i,1]
        dist0 = 0
        for j in range(nb_points):
            d = sw.dist([ref_y, dat0[j,1]], [ref_x, dat0[j,0]],units='km')[0][0]
            if d>dist0:
                dist0 = d
        dist_list.append(dist0)
    return np.median(dist_list)

#############################
# Functions for Figure_3_S2 #
#############################

def moving_average(arr, window):
    
    """
    Compute the moving average of a 1D array over a given window

    Parameters:
    ----------
    arr : 1D array (M)
        array 
    window : number of points over which the moving average is computed
    Returns:
    --------
        : 1D array (M-window)
        the moving average of arr
    """
    
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]

    return ret[window:] / window

def compute_grad_DUACS(dist, field, side, thres=0.5, window=15):
    
    """
    Approximate the slope of the ADT field from DUACS where the ADT
    gradient is maximum. The function looks at the maximum ADT gradient to
    define the front location. The front extension is given by the points where 
    the ADT gradient has fallen by a fraction (thres) from its maximum value. The slope 
    is then approximated by a linear regression over the front extension

    Parameters:
    ----------
    dist : 1D array (M)
        array of the cumulative distance in km between the points in field
    field : 1D array (M)
        array of the ADT field from DUACS
    side : str
        'north' or 'south'. If 'north' ('south') we look for a maximum (minimum)
    thres : float between 0 and 1
         fraction used to defined the drop in the ADT gradient that will defined 
         the front extension
    window : int
        window of the moving average applied on the derivative of field
    Returns:
    --------
        lin_reg : 
            contains the parameters derived from the linear regression  
        x : 1D array
            the dist variable sub-setted over the front extension
        moving_average(dist, window)[id_max_extrema] : int
            the index of the front location
    """
    
    grad = np.gradient(field, dist)
    grad_smooth = moving_average(grad, window)
    
    if side == 'north':
        
        #look at the maximum among the local maxima (to avoid selecting
        # a point on the edges)
        id_list_extrema = np.array(argrelextrema(grad_smooth, np.greater))[0]
        id_max_extrema = id_list_extrema[np.argmax(grad_smooth[id_list_extrema])]
        
        # look for the indexes (in both direction) from where the gradient has decreased by half (thres)
        # from its maximum value 
        
        lim_grad = thres*grad_smooth[id_max_extrema]
        if len(np.where(np.flip(grad_smooth[:id_max_extrema])<lim_grad)[0])>0:
            id1 = id_max_extrema - np.where(np.flip(grad_smooth[:id_max_extrema])<lim_grad)[0][0]+window//2
        else:
            id1 = window//2
        if len(np.where(grad_smooth[id_max_extrema:]<lim_grad)[0])>0:
            id2 = id_max_extrema + np.where(grad_smooth[id_max_extrema:]<lim_grad)[0][0]+window//2+1
        else:
            id2=-window//2

    elif side == 'south':
        id_list_extrema = np.array(argrelextrema(grad_smooth, np.less))[0]
        id_max_extrema = id_list_extrema[np.argmin(grad_smooth[id_list_extrema])]
        lim_grad = thres*grad_smooth[id_max_extrema]
        
        if len(np.where(grad_smooth[id_max_extrema:]>lim_grad)[0])>0:
            id2 = id_max_extrema + np.where(grad_smooth[id_max_extrema:]>lim_grad)[0][0]+window//2+1
        else:
            id2 = -window//2
        if len(np.where(np.flip(grad_smooth[:id_max_extrema])>lim_grad)[0])>0:
            id1 = id_max_extrema - np.where(np.flip(grad_smooth[:id_max_extrema])>lim_grad)[0][0]+window//2
        else:
            id1=window//2
    
    x = dist[id1:id2]
    y = field[id1:id2]
    
    lin_reg = LinearRegression()
    lin_reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    
    return lin_reg, x, moving_average(dist, window)[id_max_extrema]

def compute_grad_SWOT(dist, field, side, loc_front, dist_front=100, thres=0.5, window=15):
    
    """
    Same as compute_grad_DUACS except that the front location in field is looked among the 
    extrema that remain below dist_front from loc_front. 

    Parameters:
    ----------
    dist : 1D array (M)
        array of the cumulative distance in km between the points in field
    field : 1D array (M)
        array of the ADT field from SWOT
    side : str
        'north' or 'south'. If 'north' ('south') we look for a maximum (minimum)
    loc_front : int
        the index of the front location found in DUACS
    dist_front : float
        maximum distance allowed to the front location found in DUACS. This is to ensure that the front detected in SWOT corresponds
        to the same front detected in DUACS.
    thres : float between 0 and 1
         fraction used to defined the drop in the ADT gradient that will defined 
         the front extension
    window : int
        window of the moving average applied on the derivative of field
    Returns:
    --------
    lin_reg : 
        contains the parameters derived from the linear regression  
    x : 1D array
        the dist variable sub-setted over the front extension
    """
    
    grad = np.gradient(field,dist)
    grad_smooth = moving_average(grad, window)

    if side == 'north':
        
        # look at the maximum among the local maxima (to avoid selecting
        # a point on the edges)
        id_list_extrema = np.array(argrelextrema(grad_smooth, np.greater))[0]
        id_max_extrema = id_list_extrema[np.argmax(grad_smooth[id_list_extrema])]
        
        # SWOT has more frontal features than DUACS. As we want to try to compare
        # the same features in both products, we compare the distance between the 
        # front derived in DUACS (given by loc_front) with the maximum extrema that 
        # we detect in SWOT. If this distance is above dist_front, we consider the next
        # extrema until we found an extrema that is found below dist_front to loc_front
        # and that we consider is thus representative of the same front
        
        if abs(moving_average(dist, window)[id_max_extrema]-loc_front)>dist_front:
            id_extre_sort = np.array(argrelextrema(grad_smooth, np.greater))[0][np.flip(np.argsort(grad_smooth[argrelextrema(grad_smooth, np.greater)]))[1:]]
            pot = id_extre_sort[0]
            while (abs(moving_average(dist, window)[pot]-loc_front)>dist_front) & (len(id_extre_sort)>1):
                id_extre_sort = id_extre_sort[1:]
                pot = id_extre_sort[0]
            if abs(moving_average(dist, window)[pot]-loc_front)<dist_front:
                id_max_extrema = pot
            else:
                id_max_extrema = np.argmax(grad_smooth)
        
        # Once we've found the front location, we look for the indexes (in both direction) 
        # from where the gradient has decreased by half (thres)
        # from its value at the front position
        
        lim_grad = thres*grad_smooth[id_max_extrema]
        if len(np.where(np.flip(grad_smooth[:id_max_extrema])<lim_grad)[0])>0:
            id1 = id_max_extrema - np.where(np.flip(grad_smooth[:id_max_extrema])<lim_grad)[0][0]+window//2
        else:
            id1 = window//2
        if len(np.where(grad_smooth[id_max_extrema:]<lim_grad)[0])>0:
            id2 = id_max_extrema + np.where(grad_smooth[id_max_extrema:]<lim_grad)[0][0]+window//2+1
        else:
            id2=-window//2
    elif side == 'south':
        id_list_extrema = np.array(argrelextrema(grad_smooth, np.less))[0]
        id_max_extrema = id_list_extrema[np.argmin(grad_smooth[id_list_extrema])]
        if abs(moving_average(dist, window)[id_max_extrema]-loc_front)>dist_front:
            id_extre_sort = np.array(argrelextrema(grad_smooth, np.less))[0][np.argsort(grad_smooth[argrelextrema(grad_smooth, np.less)])[1:]]
            pot = id_extre_sort[0]
            while (abs(moving_average(dist, window)[pot]-loc_front)>dist_front) & (len(id_extre_sort)>1):
                id_extre_sort = id_extre_sort[1:]
                pot = id_extre_sort[0]
            if abs(moving_average(dist, window)[pot]-loc_front)<dist_front:
                id_max_extrema = pot
            else:
                id_max_extrema = np.argmin(grad_smooth)
        lim_grad = thres*grad_smooth[id_max_extrema]
        if len(np.where(grad_smooth[id_max_extrema:]>lim_grad)[0])>0:
            id2 = id_max_extrema + np.where(grad_smooth[id_max_extrema:]>lim_grad)[0][0]+window//2+1
        else:
            id2 = -window//2
        if len(np.where(np.flip(grad_smooth[:id_max_extrema])>lim_grad)[0])>0:
            id1 = id_max_extrema - np.where(np.flip(grad_smooth[:id_max_extrema])>lim_grad)[0][0]+window//2
        else:
            id1=window//2
    x = dist[id1:id2]
    y = field[id1:id2]
    
    lin_reg = LinearRegression()
    lin_reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    
    return lin_reg, x

def map_fig_one(fig, ax0, ds_sat, retro_thresh=0.6):
    
    """
    Plot a map of the ADT field from DUACS for a given day 

    Parameters:
    ----------
    fig : Figure
        Figure where to add the map 
    ax0 : axis
        axis that must be suitable for maps (add a projection argument)
    ds_sat : xarray
        xarray of DUACS ADT field for a given day
    retro_thresh : float
        thresh used for highlithing the Agulhas Retroflection 

    """
    
    proj = ccrs.Mercator(central_longitude=16,min_latitude=-43.25,max_latitude=-32)
    crs = ccrs.PlateCarree()
    pos_west,pos_east,pos_south,pos_north  = 10,25,-43.25,-32  #Reset the area
    extent=[pos_west, pos_east, pos_south, pos_north]
    
        #Set the colorbar properties
    Tinc=np.arange(-0.8,1.5, 0.05)
    cmap = cmocean.tools.crop(cmocean.cm.curl, vmin=Tinc[0], vmax=Tinc[-1], pivot=0)
    levels = MaxNLocator(nbins=len(Tinc)).tick_values(Tinc[0], Tinc[-1])
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
        #Set the map properties
    ax0.add_feature(cfeature.LAND,linewidth=0.5,zorder=4, edgecolor='black', facecolor=(0.85,0.85,0.85))#facecolor=(0.95,0.95,0.95))#,alpha=0.1)
    ax0.coastlines(resolution='10m', color='k', linestyle='-', alpha=1,zorder=5)
    ax0.set_extent(extent)
    map_grid=ax0.gridlines(linewidth=1, color='black', alpha=0.2, linestyle='--', draw_labels=True,zorder=6)
    map_grid.top_labels = False
    map_grid.left_labels = False
    map_grid.right_labels = False
    map_grid.bottom_labels = True        
    #map_grid.xlabel_style = {'size': ft}
    #map_grid.ylabel_style = {'size': ft}
    

        #Plot the field for the given day
    ax0.pcolormesh(ds_sat.longitude, ds_sat.latitude, ds_sat.adt, cmap=cmap, norm=norm, transform=crs)
    ax0.contour(ds_sat.longitude, ds_sat.latitude, ds_sat.adt, levels=np.arange(-1,1.2,0.2), colors='k', linewidths=1.5,transform=crs)
    ax0.contour(ds_sat.longitude, ds_sat.latitude, ds_sat.adt, levels=[retro_thresh], colors='k', linewidths=3,transform=crs)
    
def monotonic(x):
    
    """
    Test if an array is strictly monotonic

    Parameters:
    ----------
    x : array
    Returns:
   --------
       : boolean
       True is the array is strictly monotonic
    """
    
    dx = np.diff(x)
    return np.all(dx < 0) or np.all(dx > 0)

def compute_angle_DUACS(lon_c, lat_c, adt_c, ds_sat, track_vect_unit, delta=0.5, radius=10):
    
    """
    Return the angle between the SWOT track with direction vector track_vect_unit and the contour
    defining the front (adt_c) in DUACS

    Parameters:
    ----------
    lon_c : float
        longitude of the front location
    lat_c : float
        latitude of the front location
    adt_c : float
        adt level at the front location
    ds_sat : xarray
        xarray of DUACS ADT field for a given day
    track_vect_unit : array
        normalized direction vector of the SWOT track
    delta : float 
         in degree, to define the area around the front location (lon_c, lat_c) over which to
         look for the front contour
    radius : int
        in km, the distance to the front location over which the front contour is interpolated before
        looking at the tangent vector to this interpolated section of the contour
    Returns:
    --------
    angle : float
        
    """

    ds_sel = ds_sat.sel(longitude=slice(lon_c-delta,lon_c+delta), latitude=slice(lat_c-delta,lat_c+delta))

    #Get ADT contours
    plt.figure()
    cs = plt.contour(ds_sel.longitude, ds_sel.latitude, ds_sel.adt, levels=[adt_c])    
    plt.close()

    #Make regression along contour level cs to have more points to compute the tangent
    # to that contour at the location of the front

    #Contains the length of each contour
    length_contour = [len(cs.allsegs[0][j][:,0]) for j in range(len(cs.allsegs[0]))]

    #We condider only the longuest contour as the one representing the retroflection
    # Might have to change in different configuration
    id_cont = np.argsort(length_contour)[-1]

    #Interpolation will be conducted over the latitudes. As the interpolation we use
    # only works on monotonous curves, we will apply it to successive monotonous 
    # sections of the contours

    if monotonic(cs.allsegs[0][id_cont][:,0]):
        id_split = np.array([0,len(cs.allsegs[0][id_cont][:,0])-1])
    else:
        delta_sign = np.sign(np.diff(cs.allsegs[0][id_cont][:-1,0]))-np.sign(np.diff(cs.allsegs[0][id_cont][1:,0]))
        id_change_sign = np.where(delta_sign!=0)[0]
        id_split = np.concatenate(([0],id_change_sign+1,[len(cs.allsegs[0][id_cont][:,0])-1]))

    xsel=[]
    ysel=[]

    # PchipInterpolator only works on stricly increasing points

    for p in range(len(id_split)-1):
        nb_points = int(np.round((id_split[p+1]-id_split[p])*50/(len(cs.allsegs[0][id_cont][:,0])-1)))
        if cs.allsegs[0][id_cont][id_split[p],0]<cs.allsegs[0][id_cont][id_split[p+1],0]:
            xorg = cs.allsegs[0][id_cont][id_split[p]:id_split[p+1]+1,0]
            yorg = cs.allsegs[0][id_cont][id_split[p]:id_split[p+1]+1,1]
            xinterp = np.linspace(cs.allsegs[0][id_cont][id_split[p],0],cs.allsegs[0][id_cont][id_split[p+1],0],nb_points)
        else:
            xorg = np.flip(cs.allsegs[0][id_cont][id_split[p]:id_split[p+1]+1,0])
            yorg = np.flip(cs.allsegs[0][id_cont][id_split[p]:id_split[p+1]+1,1])
            xinterp = np.linspace(cs.allsegs[0][id_cont][id_split[p+1],0], cs.allsegs[0][id_cont][id_split[p],0],nb_points)
        spl = PchipInterpolator(xorg, yorg)

        #Only 'remember' points that are less than radius away from the front location

        for r in range(nb_points):
            if sw.dist([lat_c, spl(xinterp)[r]],[lon_c, xinterp[r]],units='km')[0]<radius:
                xsel.append(xinterp[r])
                ysel.append(spl(xinterp)[r])

    xsel=np.array(xsel)
    ysel=np.array(ysel)

    #Compute linear regression on the selected points of the interpolated contour

    lin_reg = LinearRegression()
    lin_reg.fit(xsel.reshape(-1, 1), ysel.reshape(-1, 1))
    y_pred = lin_reg.predict(np.array([xsel[0], xsel[-1]]).reshape(-1, 1))[:,0]

    # Define direction vector of the tangent vector to the contour
    reg_vect = np.array([xsel[0] - xsel[-1], y_pred[0] - y_pred[-1]])
    reg_vect_unit = reg_vect/np.linalg.norm(reg_vect)

    # Compute the angle between the contour and the SWOT track (always below 90 degree)
    angle = np.rad2deg(np.arccos(np.dot(track_vect_unit,reg_vect_unit)))
    if angle >90:
        angle = 180 - angle

    return angle

def compute_angle_SWOT(lon_c, lat_c, adt_c, ds_swot, lon_swot, lat_swot, id_track, track_vect_unit, delta=0.5, radius=10):
    
    """
    Same as compute_angle_DUACS but for SWOT data

    Parameters:
    ----------
    lon_c : float
        longitude of the front location
    lat_c : float
        latitude of the front location
    adt_c : float
        adt level at the front location
    ds_swot : xarray
        xarray of SWOT ADT field for a given day
    lon_swot : array
        1D array of the longitude of the SWOT track
    lat_swot : array
        1D array of the latitude of the SWOT track
    id_track : int
        index of the num_pixels within SWOT pass
    track_vect_unit : array
        normalized direction vector of the SWOT track
    delta : float 
         in degree, to define the area around the front location (lon_c, lat_c) over which to
         look for the front contour
    radius : int
        in km, the distance to the front location over which the tangent vector to the front contour is computed
    Returns:
    --------
    angle : float
        
    """
    
    id_sel = np.unique(np.where((ds_swot.longitude>lon_c-delta) & (ds_swot.longitude<lon_c+delta) & (ds_swot.latitude>lat_c-delta) & (ds_swot.latitude<lat_c+delta))[0])

    ds_sel_adt = ds_swot.ssha_filtered.data[id_sel] + ds_swot.mdt.data[id_sel]
    row, col = np.where(~np.isnan(ds_sel_adt)==1)
    
    #Plot angle figure:
    plt.figure(figsize=(9,11))
    cs = plt.tricontour(ds_swot.longitude.data[id_sel][row,col].flatten(), ds_swot.latitude.data[id_sel][row,col].flatten(), ds_sel_adt[row,col].flatten(), [adt_c])
    plt.close()

    #Find contour that intersect SWOT track and is the closest to the front location
    list_inters = []
    idx_ct = []
    for j in range(len(cs.allsegs[0])):
        lines = LineString(cs.allsegs[0][j]).intersection(LineString(np.vstack((lon_swot,lat_swot)).T))
        if not(lines.is_empty):
            if lines.geom_type == 'MultiPoint':
                lines = lines.geoms[0]
            list_inters.append(sw.dist([lat_c, lines.y],[lon_c, lines.x],units='km')[0])
            idx_ct.append(j)

    contour_select = idx_ct[np.argmin(list_inters)]

    #Select only a few points of that contour to make the linear regression
    xsel=[]
    ysel=[]
    if id_track==7:
        rr=radius/2
    else:
        rr=radius
    for p in range(len(cs.allsegs[0][contour_select][:,0])):
        if sw.dist([lat_c, cs.allsegs[0][contour_select][p,1]],[lon_c, cs.allsegs[0][contour_select][p,0]],units='km')[0]<rr:
            xsel.append(cs.allsegs[0][contour_select][p,0])
            ysel.append(cs.allsegs[0][contour_select][p,1])
    xsel=np.array(xsel)
    ysel=np.array(ysel)

    lin_reg = LinearRegression()
    lin_reg.fit(xsel.reshape(-1, 1), ysel.reshape(-1, 1))
    y_pred = lin_reg.predict(np.array([xsel[0], xsel[-1]]).reshape(-1, 1))[:,0]

    reg_vect = np.array([xsel[0] - xsel[-1], y_pred[0] - y_pred[-1]])
    reg_vect_unit = reg_vect/np.linalg.norm(reg_vect)

    angle = np.rad2deg(np.arccos(np.dot(track_vect_unit,reg_vect_unit)))
    if angle >90:
        angle = 180 - angle

    return angle

def distribution(arr_x, arr_y, x_bin, y_bin):
    
    """
    Compute the distribution in % of two variables over a 2D space

    Parameters:
    ----------
    arr_x : array
        1D array of the first variable
    arr_y : array
        1D array of the second variable
    x_bin : array
        1D array of the binning cells for the first variable
    y_bin : array
        1D array of the binning cells for the second variable
    Returns:
    --------
    count : float
        2D array of the distribution in % of the two variables in the 
        different bins
    """
    
    count=np.zeros([len(y_bin)-1,len(x_bin)-1])
    count[:]=np.nan
    for i in range(len(y_bin)-1):
        for j in range(len(x_bin)-1):
            id_select = np.where((arr_x>=x_bin[j]) & (arr_x<x_bin[j+1]) & (arr_y>=y_bin[i]) & (arr_y<y_bin[i+1]))[0]
            if len(id_select)>0:
                count[i,j]=len(id_select)
    return count*100/len(arr_x)


################################
# Functions for Figure_4_S1_S3 #
################################


def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False
    
def inertial_frequency_PCs(finert, var, ds_drifter, lim_lag):
    
    """
    Extract the velocity component associated to inertial oscillations from a 
    velocity timeseries using Singular Spectral Analysis. The statistical significance
    of this component is evaluated through a Monte-Carlo test.

    Parameters:
    ----------
    finert : float
        Inertial frequency
    var : str
        'utot' or 'vtot'
    ds_drifter : xarray
        xarray of one drifter trajectory regularly spaced in time
    lim_lag : int
        window length (embedding dimension) for the SSA
    Returns:
    --------
       : array
        1D array of the inertial component of var
    """

    #Range of frenquencies around finert
    fup = finert + 1/(2*lim_lag)
    fdw = finert - 1/(2*lim_lag)

    #Compue SSA on u (with Monte-Carlo test with 1000 samples)
    x_avg = ds_drifter[var].data
    mssa = MCSSA(x_avg)
    mssa.run_mcssa(lim_lag)

    ff = mssa.freqs
    errors = np.array(mssa.stats.iloc[3:5, mssa.freq_rank])
    mean_suro = np.array(mssa.stats.iloc[0,  mssa.freq_rank])
    y = mssa.values[mssa.freq_rank]

    #Find pair principal components (PC) with frequencies close to finert
    id_select = np.intersect1d(np.where(ff>=fdw)[0], np.where(ff<=fup)[0])

    #Only consider the PCs whose variance is significative (estimated through the Monte-Carlo test)
    id_potential = []
    for idx in id_select:
        thresh = errors[1,np.where(mssa.freq_rank==idx)[0]] + mean_suro[np.where(mssa.freq_rank==idx)[0]]
        position_PC = y[np.where(mssa.freq_rank==idx)[0]]
        if np.real(position_PC)[0]>thresh[0]:
            id_potential.append(idx)

    #We are looking for a pair of PCs. They have similar variance and are thus usually of successive order (freq_rank)
    if len(id_potential)<2:
        id_select=[]
    elif len(id_potential)>2:
        diff = np.diff(id_potential)
        if len(np.where(diff==1)[0])>0:
            idx = np.where(diff==1)[0][0]
            id_select = id_potential[idx:idx+2]
        else:
            id_select = []
    else:
        if np.diff(id_potential)>2:
            delta = np.real(y[np.where(mssa.freq_rank==id_potential[0])[0]]-y[np.where(mssa.freq_rank==id_potential[1])[0]])[0]
            if (delta / np.real(y[np.where(mssa.freq_rank==id_potential[0])[0]]))>0.1:
                id_select = []
        else:
            id_select = id_potential
    return mssa.reconstruct(id_select)


def interpolate_insitu(platform, step):
    
    """
    Interpolate several variables stored in an xarray with a time dimension on a 
    regularly spaced along-track path
    
    Parameters
    ----------
    platform : xarray
        xarray containing the field to be interpolated from time to space
    step : float
        spatial interpolation step
    Returns
    -------
       : xarray
        the initial xarray with fields now interpolated in space
    """
    
    dist = np.hstack([0,np.cumsum(sw.dist(platform.latitude,platform.longitude,units='km')[0])])
    #Remove dulpicates
    dist, idx = np.unique(dist, return_index=True)
    platform = platform.isel(time=idx)
    platform['dist_total'] = xr.DataArray(dist,dims=('time'),coords={'time':platform.time})
    
    #Create new variable: time in second since first point of the dataset
    timeinsec = (platform.time.data - platform.time.data[0])/np.timedelta64(1, 's')
    platform['timeinsec']=xr.DataArray(timeinsec,dims=('time'),coords={'time':platform.time})
    
    #Interpolate in space - swap dimensions
    ds2 = platform.assign_coords(dist_total = platform.dist_total)  #add dist traveled coordinate
    ds3 = ds2.swap_dims({'time':'dist_total'})                  #swap from time to distance traveled
    dist_interp = np.arange(ds2.dist_total[0],ds2.dist_total[-1],step) #Interpolate the distance with the corresponding lenght
    ds4 = ds3.interp(dist_total=dist_interp)
    
    
    #Swap back dimensions to time
    time = np.zeros_like(ds4.timeinsec,dtype='datetime64[ns]')
    day0 = dt2cal(platform.time.data[0])
    for i in range(len(time)):
        time[i] = np.datetime64(datetime(day0[0], day0[1], day0[2], day0[3], day0[4], day0[5]) + timedelta(seconds=ds4.timeinsec.data[i]))
    ds4['time'] = xr.DataArray(time,dims=('dist_total'),coords={'dist_total':ds4.dist_total})
    ds5 = ds4.swap_dims({'dist_total':'time'})               

    return ds5


def dt2cal(dt):
    
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond.
    Parameters
    ----------
    dt : datetime64 array (M)
        1D array of datetimes 
    Returns
    -------
    cal : array (M x 7)
        2D array of integers with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # allocate output 
    out = np.empty(dt.shape + (7,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 2] = (D - M) + 1 # dat
    out[..., 3] = (dt - D).astype("m8[h]") # hour
    out[..., 4] = (dt - h).astype("m8[m]") # minute
    out[..., 5] = (dt - m).astype("m8[s]") # second
    out[..., 6] = (dt - s).astype("m8[us]") # microsecond
    return out

def velocity_and_time_swot(platform):
    
    """
    Co-locate the platform trajectory with the closest SWOT observations in time
    
    Parameters
    ----------
    platform : xarray
        xarray containing the platform trajectory
    Returns
    -------
    swot_geos : array 
        2D array of co-located SWOT geostrophic velocity component with the last axis 
        representing u and v
    time_diff_check : array
        1D array of the time difference in hours between the platform observations
        and the SWOT colocated field.
    """
    
    #Initialization array to save SWOT velocity components
    
    swot_geos = np.zeros([len(platform.time.data),2])
    swot_geos[:] = np.nan
    
    time_diff_check = np.zeros(len(platform.time.data))
    time_diff_check[:] = np.nan

    for p in range(len(platform.time.data)):

        point = Point(platform.longitude.data[p],platform.latitude.data[p]) # create point
        time_platform=platform.time.data[p]

        day = np.datetime64(datetime(dt2cal(platform.time.data[p])[0],dt2cal(platform.time.data[p])[1],dt2cal(platform.time.data[p])[2]))
        
        if dt2cal(day)[2]<10:
            date_str = str(dt2cal(day)[0]) + '0' + str(dt2cal(day)[1]) + '0' + str(dt2cal(day)[2])
        else:
            date_str = str(dt2cal(day)[0]) + '0' + str(dt2cal(day)[1]) + str(dt2cal(day)[2])
        
        #The platform is under pass 16 only
        if (polygon1.contains(point)) & (not(polygon2.contains(point))):
            file_SWOT_pot = [x for x in glob(swot_dir+'*' + '_016_' + '*' + date_str+'*.nc')]
            if len(file_SWOT_pot)>0:
                
                #Select the first potential file
                file_SWOT_pot = [x for x in glob(swot_dir+'*' + '_016_' + '*' + date_str+'*.nc')][0]
                idx = np.where(trace_016 == file_SWOT_pot)[0][0]
                
                #To be sure we select the SWOT observations the closest in time to the in situ, 
                # we look at the SWOT observations from +/- 2 days before (could be reduced)
                
                if idx<3:
                    SWOT_test = trace_016[:idx+3]
                else:
                    SWOT_test = trace_016[idx-2:idx+3]

                swot_array = xr.open_dataset(file_SWOT_pot)

                # First, find the index of the grid point nearest a specific lat/lon.   
                abslat = np.abs(swot_array.latitude-platform.latitude.data[p])
                abslon = np.abs(swot_array.longitude-platform.longitude.data[p])
                c = np.maximum(abslon, abslat)

                ([xloc], [yloc]) = np.where(c == np.min(c))
                
                time_list = np.zeros(len(SWOT_test))
                geos_list = np.zeros(len(SWOT_test))
                time_list[:]=np.nan
                geos_list[:]=np.nan
            
                for i in range(len(time_list)):
                    swot_array = xr.open_dataset(SWOT_test[i])
                    if swot_array.sel(num_lines=xloc, num_pixels=yloc).quality_flag.data==0:

                        time_list[i] = abs(time_platform - swot_array.time.data[xloc])/np.timedelta64(1, 'h')
                        point_ds = swot_array.sel(num_lines=xloc, num_pixels=yloc) 
                        geos_list[i] = np.sqrt(point_ds.ugos_filtered.data**2 + point_ds.vgos_filtered.data**2)
            
                idnan = np.where(np.isnan(geos_list)==0)[0]
            
                if len(idnan)>0:
                    #Select SWOT observation that is the closest to the time of the in situ observations
                    id_select = idnan[np.argmin(time_list[idnan])]
                    
                    #Only consider the SWOT observation if it's less than 12h away from the in situ 
                    if abs(time_list[id_select])<12:
                        swot_array = xr.open_dataset(SWOT_test[id_select])
                        point_ds = swot_array.sel(num_lines=xloc, num_pixels=yloc) 
                        swot_geos[p,0] = point_ds.ugos_filtered.data
                        swot_geos[p,1] = point_ds.vgos_filtered.data
                        time_diff_check[p] = abs(time_platform - point_ds.time)/np.timedelta64(1, 'h')

        #Same as before but the platform is under pass 1 only

        elif (polygon2.contains(point)) & (not(polygon1.contains(point))):
            file_SWOT_pot = [x for x in glob(swot_dir+'*' + '_001_' + '*' + date_str+'*.nc')]
            if len(file_SWOT_pot)>0:
                file_SWOT_pot = [x for x in glob(swot_dir+'*' + '_001_' + '*' + date_str+'*.nc')][0]
                idx = np.where(trace_001 == file_SWOT_pot)[0][0]
                if idx<3:
                    SWOT_test = trace_001[:idx+3]
                else:
                    SWOT_test = trace_001[idx-2:idx+3]


                swot_array = xr.open_dataset(file_SWOT_pot)

                # First, find the index of the grid point nearest a specific lat/lon.   
                abslat = np.abs(swot_array.latitude-platform.latitude.data[p])
                abslon = np.abs(swot_array.longitude-platform.longitude.data[p])
                c = np.maximum(abslon, abslat)

                ([xloc], [yloc]) = np.where(c == np.min(c))
            
                time_list = np.zeros(len(SWOT_test))
                geos_list = np.zeros(len(SWOT_test))
                time_list[:]=np.nan
                geos_list[:]=np.nan
            
                for i in range(len(time_list)):
                    swot_array = xr.open_dataset(SWOT_test[i])
                    if swot_array.sel(num_lines=xloc, num_pixels=yloc).quality_flag.data==0:
                        time_list[i] = abs(time_platform - swot_array.time.data[xloc])/np.timedelta64(1, 'h')
                        point_ds = swot_array.sel(num_lines=xloc, num_pixels=yloc) 
                        geos_list[i] = np.sqrt(point_ds.ugos_filtered.data**2 + point_ds.vgos_filtered.data**2)
            
                idnan = np.where(np.isnan(geos_list)==0)[0]
            
                if len(idnan)>0:
                    id_select = idnan[np.argmin(time_list[idnan])]
               
                    if abs(time_list[id_select])<12:
            
                        swot_array = xr.open_dataset(SWOT_test[id_select])
                        point_ds = swot_array.sel(num_lines=xloc, num_pixels=yloc) 
                        swot_geos[p,0] = point_ds.ugos_filtered.data
                        swot_geos[p,1] = point_ds.vgos_filtered.data
                        time_diff_check[p] = abs(time_platform - point_ds.time)/np.timedelta64(1, 'h')
        
        #Same as before but the platform is under the crossover
        elif (polygon2.contains(point)) & (polygon1.contains(point)):
            
            #for polygon1
            file_SWOT_pot1 = [x for x in glob(swot_dir+'*' + '_016_' + '*' + date_str+'*.nc')]
            if len(file_SWOT_pot1):
                file_SWOT_pot1 = [x for x in glob(swot_dir+'*' + '_016_' + '*' + date_str+'*.nc')][0]
                idx = np.where(trace_016 == file_SWOT_pot1)[0][0] 
                if idx<3:
                    SWOT_test1 = trace_016[:idx+3]
                else:
                    SWOT_test1 = trace_016[idx-2:idx+3]

                swot_array = xr.open_dataset(file_SWOT_pot1)

                # First, find the index of the grid point nearest a specific lat/lon.   
                abslat = np.abs(swot_array.latitude-platform.latitude.data[p])
                abslon = np.abs(swot_array.longitude-platform.longitude.data[p])
                c = np.maximum(abslon, abslat)

                ([xloc1], [yloc1]) = np.where(c == np.min(c))
                
                time_list1 = np.zeros(len(SWOT_test1))
                geos_list1 = np.zeros(len(SWOT_test1))
                time_list1[:]=np.nan
                geos_list1[:]=np.nan
            
                for i in range(len(time_list1)):
                    swot_array = xr.open_dataset(SWOT_test1[i])
                    if swot_array.sel(num_lines=xloc1, num_pixels=yloc1).quality_flag.data==0:
                        time_list1[i] = abs(time_platform - swot_array.time.data[xloc1])/np.timedelta64(1, 'h')
                        point_ds = swot_array.sel(num_lines=xloc1, num_pixels=yloc1) 
                        geos_list1[i] = np.sqrt(point_ds.ugos_filtered.data**2 + point_ds.vgos_filtered.data**2)
            
                idnan = np.where(np.isnan(geos_list1)==0)[0]
                if len(idnan)>0:
                    id_select1 = idnan[np.argmin(time_list1[idnan])]
                else:
                    id_select1 = np.nan
            else:
                id_select1=np.nan
            
            #for polygon2
            file_SWOT_pot2 = [x for x in glob(swot_dir+'*' + '_001_' + '*' + date_str+'*.nc')]
            if len(file_SWOT_pot2)>0:
                file_SWOT_pot2 = [x for x in glob(swot_dir+'*' + '_001_' + '*' + date_str+'*.nc')][0]
                idx = np.where(trace_001 == file_SWOT_pot2)[0][0]  
                if idx<3:
                    SWOT_test2 = trace_001[:idx+3]
                else:
                    SWOT_test2 = trace_001[idx-2:idx+3]

                swot_array = xr.open_dataset(file_SWOT_pot2)

                # First, find the index of the grid point nearest a specific lat/lon.   
                abslat = np.abs(swot_array.latitude-platform.latitude.data[p])
                abslon = np.abs(swot_array.longitude-platform.longitude.data[p])
                c = np.maximum(abslon, abslat)

                xloc2, yloc2 = np.where(c == np.min(c))[0][0], np.where(c == np.min(c))[1][0]
            
                time_list2 = np.zeros(len(SWOT_test2))
                geos_list2 = np.zeros(len(SWOT_test2))
                time_list2[:]=np.nan
                geos_list2[:]=np.nan
            
                for i in range(len(time_list2)):
                    swot_array = xr.open_dataset(SWOT_test2[i])
                    if swot_array.sel(num_lines=xloc2, num_pixels=yloc2).quality_flag.data==0:
                        time_list2[i] = abs(time_platform - swot_array.time.data[xloc2])/np.timedelta64(1, 'h')
                        point_ds = swot_array.sel(num_lines=xloc2, num_pixels=yloc2) 
                        geos_list2[i] = np.sqrt(point_ds.ugos_filtered.data**2 + point_ds.vgos_filtered.data**2)
            
                idnan = np.where(np.isnan(geos_list2)==0)[0]
                if len(idnan)>0:
                    id_select2 = idnan[np.argmin(time_list2[idnan])]
                else:
                    id_select2 = np.nan
            else:
                id_select2=np.nan
            
            if (np.isnan(id_select1)) & (not(np.isnan(id_select2))):
                id_select = 1
            elif (np.isnan(id_select2)) & (not(np.isnan(id_select1))):
                id_select = 0
            elif (np.isnan(id_select1)) & (np.isnan(id_select2)):
                id_select = np.nan
            else:
                id_select = np.argmin([time_list1[id_select1], time_list2[id_select2]])
            if  id_select == 0:             
                if abs(time_list1[id_select1])<12:
                    swot_array = xr.open_dataset(SWOT_test1[id_select1])
                    point_ds = swot_array.sel(num_lines=xloc1, num_pixels=yloc1) 
                    swot_geos[p,0] = point_ds.ugos_filtered.data
                    swot_geos[p,1] = point_ds.vgos_filtered.data
                    time_diff_check[p] = abs(time_platform - point_ds.time)/np.timedelta64(1, 'h')
            elif id_select == 1:
                if abs(time_list2[id_select2])<12:
                    swot_array = xr.open_dataset(SWOT_test2[id_select2])
                    point_ds = swot_array.sel(num_lines=xloc2, num_pixels=yloc2) 
                    swot_geos[p,0] = point_ds.ugos_filtered.data
                    swot_geos[p,1] = point_ds.vgos_filtered.data
                    time_diff_check[p] = abs(time_platform - point_ds.time)/np.timedelta64(1, 'h')
                
    return swot_geos, time_diff_check

def colocation_DUACS(platform):
    
    """
    Co-locate the platform trajectory with the closest DUACS observations in time
    
    Parameters
    ----------
    platform : xarray
        xarray containing the platform trajectory
    Returns
    -------
    extract : array 
        2D array of co-located DUACS geostrophic velocity component with the last axis 
        representing u and v
    """
    
    timestart = platform.time[0].values
    nb_days = (platform.time[-1].values - timestart)//np.timedelta64(24, 'h')
    extract = np.zeros([len(platform.time),2])

    for j in tqdm(range(nb_days+3)):
        day = np.datetime64(datetime(dt2cal(platform.time.data)[0,0],dt2cal(platform.time.data)[0,1],dt2cal(platform.time.data)[0,2]) + timedelta(days=np.float64(j)))
        ds_sat_select = ds_ssh.sel(time=day, method='nearest')
        
        day_low = np.datetime64(datetime(dt2cal(platform.time.data)[0,0],dt2cal(platform.time.data)[0,1],dt2cal(platform.time.data)[0,2]) + timedelta(days=np.float64(j)) - timedelta(hours=12))
        day_up = np.datetime64(datetime(dt2cal(platform.time.data)[0,0],dt2cal(platform.time.data)[0,1],dt2cal(platform.time.data)[0,2]) + timedelta(days=np.float64(j)) + timedelta(hours=12))

        id_select = np.where((platform.time>=day_low) & (platform.time<=day_up))[0]
        if len(id_select)>0:
            platform_select = platform.isel(time=id_select)
            extract[id_select,0] = ds_sat_select['ugos'].interp(longitude=('z', platform_select.longitude.data), latitude=('z', platform_select.latitude.data))
            extract[id_select,1] = ds_sat_select['vgos'].interp(longitude=('z', platform_select.longitude.data), latitude=('z', platform_select.latitude.data))
   
    return extract

def colocation_other(platform, product):
    
    """
    Co-locate the platform trajectory with the closest observations in time from
    products available hourly
    
    Parameters
    ----------
    platform : xarray
        xarray containing the platform trajectory
    product : str
        'ekman' or 'stokes'
    Returns
    -------
    extract : array 
        2D array of co-located velocity components with the last axis 
        representing u and v
    """
    
    timestart = platform.time[0].values
    nb_hours = (platform.time[-1].values - timestart)//np.timedelta64(1, 'h')
    extract = np.zeros([len(platform.time),2])

    for j in tqdm(range(nb_hours+3)):
        day = np.datetime64(datetime(dt2cal(platform.time.data)[0,0],dt2cal(platform.time.data)[0,1],dt2cal(platform.time.data)[0,2],dt2cal(platform.time.data)[0,3]) + timedelta(hours=np.float64(j)))
        if product == 'ekman':
            ds_sat_select = ds_ekman.sel(time=day,depth=0)
        elif product=='stokes':
            ds_sat_select = ds_stokes.sel(valid_time=day)

        day_low = np.datetime64(datetime(dt2cal(platform.time.data)[0,0],dt2cal(platform.time.data)[0,1],dt2cal(platform.time.data)[0,2],dt2cal(platform.time.data)[0,3]) + timedelta(hours=np.float64(j)) - timedelta(minutes=30))
        day_up = np.datetime64(datetime(dt2cal(platform.time.data)[0,0],dt2cal(platform.time.data)[0,1],dt2cal(platform.time.data)[0,2],dt2cal(platform.time.data)[0,3]) + timedelta(hours=np.float64(j)) + timedelta(minutes=30))

        id_select = np.where((platform.time>=day_low) & (platform.time<=day_up))[0]
        if len(id_select)>0:
            platform_select = platform.isel(time=id_select)
            if product == 'ekman':
                uvar='ue'
                vvar='ve'
            elif product=='stokes':
                uvar='ust'
                vvar='vst'            
            extract[id_select,0] = ds_sat_select[uvar].interp(longitude=('z', platform_select.longitude.data), latitude=('z', platform_select.latitude.data))
            extract[id_select,1] = ds_sat_select[vvar].interp(longitude=('z', platform_select.longitude.data), latitude=('z', platform_select.latitude.data))

    return extract





