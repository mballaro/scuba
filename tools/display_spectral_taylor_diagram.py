#!/usr/bin/python
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic
from math import sin, cos, atan2, pi, sqrt
from netCDF4 import Dataset
from sys import argv
from numpy import arange, zeros, meshgrid, ma
from matplotlib import rc, colors
#rc('text', usetex=True)
from scipy.fftpack import fft
import scipy.interpolate
from scipy.optimize import curve_fit

INPUT_FILE = argv[1]

NC = Dataset(INPUT_FILE)
FREQ = NC.variables['frequency'][:, :, :].filled(0)
PS_along_track = NC.variables['spectrum_along_track'][:, :, :]
PS_map = NC.variables['spectrum_map'][:, :, :]
PSD_along_track = NC.variables['psd_along_track'][:, :, :]
PSD_map = NC.variables['psd_map'][:, :, :]
COHERENCE = NC.variables['coherence'][:, :, :]
LAT = NC.variables['lat2D'][:, :]
LON = NC.variables['lon2D'][:, :]
NB_SEG_COMPUTATION = NC.variables['nb_segment'][:, :]
MAP_EFFECTIVE_RESOLUTION = NC.variables['effective_resolution'][:, :]
NC.close()

LON = np.where(LON >= 360, LON-360, LON)
LON = np.where(LON < 0, LON+360, LON)

def find_nearest_index(array_lon, array_lat, value_lon, value_lat):
    """
    Function find nearest index
    :param array: input array
    :param value: value
    :return: nearest index in array equal to value
    """
    idy = np.argmin(np.abs(array_lat - value_lat), axis=0)
    idx = np.argmin(np.abs(array_lon[idy[0], :] - value_lon))
    return idx, idy[0]


def find_nearest_index1D(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def updateThetaAxis(ax):
    labels=[]
    for f in ax.get_xticks():
        labels.append(np.round(np.cos(f),2))
    ax.set_xticklabels(labels[::2],fontsize=10)
    return labels

def plt_spectrum_coherence(cix, ciy):
    """

    """
    ilon, ilat = find_nearest_index(LON, LAT, cix, ciy)

    normalized_std = PSD_map[:, ilat, ilon]/PSD_along_track[:, ilat, ilon] # in m
    coherence = COHERENCE[:, ilat, ilon]
    angle_coherence = np.arccos(COHERENCE[:, ilat, ilon])

    plt.ion()
    plt.show()

    r_theta05=np.arange(0,1.2,0.01)
    
    r_theta_area = np.linspace(0.5,1,120)
    
    theta05 = np.arccos(0.5)*np.ones(np.shape(r_theta05))
    theta0 = 0*np.ones(np.shape(r_theta05))
    theta_r05 = np.arccos(np.arange(0,1.5,0.01))
    r05 = 0.5*np.ones(np.shape(theta_r05))
    r1 = np.ones(np.shape(theta_r05))

    tt = np.arange(0,np.arccos(0.49), 0.01)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, polar=True)
    ax1.fill_between(tt, 0.5, 1, color='0.3', alpha=0.5)
    #ax1.fill_betweenx(r_theta_area, theta0, theta05, color='0.3', alpha=0.5)
    ax1.plot(angle_coherence, normalized_std, color='0.5', alpha=0.5)
    c = ax1.scatter(angle_coherence, normalized_std, c=1/FREQ[:, ilat, ilon], s=50, cmap='Spectral_r', lw=0, vmin=100, vmax=800)
    ax1.scatter(0, 1, c='white', s=150, marker=(5, 1), zorder=10, facecolors='none', edgecolors='r')
    ax1.plot(theta05, r_theta05, color='k', lw=2)
    ax1.plot(theta_r05, r05, color='k', lw=2)
    ax1.plot(theta_r05, r1, color='k', lw=2)
    ax1.set_thetamin(0)
    ax1.set_thetamax(90)
    ax1.set_rmax(1.2)
    ax1.grid(True)
    #updateThetaAxis(ax1)
    # ax1.set_ylabel("Normalized standard deviation of power (m)", fontweight='bold', fontsize=12)
    ax1.set_ylabel("Normalized PSD", fontweight='bold', fontsize=12)
    plt.annotate("Spectral Taylor Diagram at lon = %s and lat = %s" %(str(np.round(LON[ilat, ilon],2)), str(np.round(LAT[ilat, ilon],2))), xy=(0.3, 1.07), xycoords='axes fraction', color='black',
                    bbox=dict(facecolor='none', edgecolor='black', pad=10.0))
#    ax1.set_title("Spectral Taylor Diagram at lon = %s and lat = %s" %(str(np.round(LON[ilat, ilon],2)), str(np.round(LAT[ilat, ilon],2))) )
    cbar = plt.colorbar(c)
    cbar.set_label('Wavelength (km)', fontweight='bold')
    plt.annotate('Coherence', xy=(0.75, 0.85), # theta, radius
                             xycoords='axes fraction', rotation=-50, fontweight='bold', fontsize=12
                             )
    plt.annotate('REF', xy=(0.85, 0.04), # theta, radius
                 xycoords='axes fraction', fontweight='bold', color='red', fontsize=12
    )
    plt.annotate('ratio = 0.5', xy=(pi/3.5, 0.45), # theta, radius
                 xycoords='data', rotation=-50, fontsize=8
    )
    plt.annotate('ratio = 1.0', xy=(pi/4, 1.01), # theta, radius
                 xycoords='data', rotation=-50, fontsize=8
    )
    plt.annotate('Coherence = 0.5', xy=(pi/2.1, 0.3), # theta, radius
                 xycoords='data', rotation=60, fontsize=8
    )
    labels = updateThetaAxis(ax1)
    #ax1.set_xticks(labels)
    plt.show()


def onclick(event):
    """
    Function onclick
    :param event:
    :return:
    """
    toolbar = plt.get_current_fig_manager().toolbar
    if event.button == 1 and event.inaxes and toolbar.mode == '':
        #global indexx, indexy
        indexx, indexy = event.xdata, event.ydata
        print(' ')
        print('lon = ', indexx, ';', 'lat = ', indexy)
        plt_spectrum_coherence(indexx, indexy)
    else:
        print('Clicked ouside axes bounds but inside plot window \n')


        
FIG, AX = plt.subplots()
MAP = Basemap(projection='cyl',
              llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-10, urcrnrlon=365,
              resolution='i')
MAP.drawcoastlines(zorder=5, linewidth=0.25)
MAP.fillcontinents(color='grey', lake_color='white')
MAP.drawparallels(arange(-90, 90, 30), labels=[1, 0, 0, 0], fontsize=20, dashes=[4, 4], linewidth=0.25)
MAP.drawmeridians(arange(-180, 360, 45), labels=[0, 0, 0, 1], fontsize=20, dashes=[4, 4], linewidth=0.25)
X,Y = MAP(LON, LAT)
C1 = MAP.scatter(X, Y, c=NB_SEG_COMPUTATION, s=2, lw=0)#, norm=colors.LogNorm())
CBAR = MAP.colorbar(C1, location='bottom', pad='15%')
CBAR.set_label('Nb segment in computation')
CID = FIG.canvas.callbacks.connect('button_press_event', onclick)
plt.show()
