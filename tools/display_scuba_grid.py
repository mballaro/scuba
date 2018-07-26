"""
Display SCUBA analysis for spectral lovers...
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic
from math import sin, cos, atan2, pi, sqrt
from netCDF4 import Dataset
from sys import argv, path
from numpy import arange, zeros, meshgrid, ma
from matplotlib import rc, colors
from scipy.fftpack import fft
import scipy.interpolate
from scipy.optimize import curve_fit

path.insert(0, '../src/')
from mod_geo import *


input_file_x_direction = argv[1]
input_file_y_direction = argv[2]

nc = Dataset(input_file_x_direction)
wavenumber_x = nc.variables['frequency'][:, :, :].filled(0)
lat = nc.variables['lat2D'][:, :]
lon = nc.variables['lon2D'][:, :]
nb_segment = nc.variables['nb_segment'][:, :]
psd_ref_x = nc.variables['psd_ref'][:, :, :]
autocorrelation_ref_x = nc.variables['autocorrelation_ref'][:, :, :]
distance_x = nc.variables['distance'][:, :, :]
distance_units_x = nc.variables['distance'].units
zero_crossing_ref_x = nc.variables['zero_crossing_ref'][:, :]

flag_study_x = True
try:
    psd_study_x = nc.variables['psd_study'][:, :, :]
    coherence_x = nc.variables['coherence'][:, :, :]
    effective_resolution_x = nc.variables['effective_resolution'][:, :]
    useful_resolution_x = nc.variables['useful_resolution'][:, :]
    autocorrelation_study_x = nc.variables['autocorrelation_study'][:, :, :]
    zero_crossing_study_x = nc.variables['zero_crossing_study'][:, :]

except:
    flag_study_x = False
    psd_study_x = np.zeros(np.shape(psd_ref_x))
    coherence_x = np.zeros(np.shape(psd_ref_x))
    effective_resolution_x = np.zeros(np.shape(lon))
    useful_resolution_x = np.zeros(np.shape(lon))
    autocorrelation_study_x = np.zeros(np.shape(psd_ref_x))
    zero_crossing_study_x = np.zeros(np.shape(lon))

nc.close()

nc = Dataset(input_file_y_direction)
wavenumber_y = nc.variables['frequency'][:, :, :].filled(0)
lat = nc.variables['lat2D'][:, :]
lon = nc.variables['lon2D'][:, :]
nb_segment = nc.variables['nb_segment'][:, :]
psd_ref_y = nc.variables['psd_ref'][:, :, :]
autocorrelation_ref_y = nc.variables['autocorrelation_ref'][:, :, :]
distance_y = nc.variables['distance'][:, :, :]
distance_units_y = nc.variables['distance'].units
zero_crossing_ref_y = nc.variables['zero_crossing_ref'][:, :]

flag_study_y = True
try:
    psd_study_y = nc.variables['psd_study'][:, :, :]
    coherence_y = nc.variables['coherence'][:, :, :]
    effective_resolution_y = nc.variables['effective_resolution'][:, :]
    useful_resolution_y = nc.variables['useful_resolution'][:, :]
    autocorrelation_study_y = nc.variables['autocorrelation_study'][:, :, :]
    zero_crossing_study_y = nc.variables['zero_crossing_study'][:, :]

except:
    flag_study_y = False
    psd_study_y = np.zeros(np.shape(psd_ref_y))
    coherence_y = np.zeros(np.shape(psd_ref_y))
    effective_resolution_y = np.zeros(np.shape(lon))
    useful_resolution_y = np.zeros(np.shape(lon))
    autocorrelation_study_y = np.zeros(np.shape(psd_ref_y))
    zero_crossing_study_y = np.zeros(np.shape(lon))

nc.close()

lon = np.where(lon >= 360, lon-360, lon)
lon = np.where(lon < 0, lon+360, lon)


def func_y0(x, a):
    return 0*x + a


def func_y1(x, a, b):
    return a*x + b


def func_y2slope(x, a1, b1, a2, b2):
    return a1*x + b1 + a2*x + b2 


def plt_spectral_analysis(cix, ciy):
    """
    Plot spectrum and coherence at a specific (lon, lat) coordinates
    :param cix:
    :param ciy:
    :return:
    """

    ilon, ilat = find_nearest_index_lonlat(lon, lat, cix, ciy)

    if flag_study_x:
        study_zero_crossing_x = zero_crossing_study_x[ilat, ilon]
        study_effective_resolution_x = effective_resolution_x[ilat, ilon]
        study_useful_resolution_x = useful_resolution_x[ilat, ilon]

    if flag_study_y:
        study_zero_crossing_y = zero_crossing_study_y[ilat, ilon]
        study_effective_resolution_y = effective_resolution_y[ilat, ilon]
        study_useful_resolution_y = useful_resolution_y[ilat, ilon]

    ref_zero_crossing_x = zero_crossing_ref_x[ilat, ilon]
    ref_zero_crossing_y = zero_crossing_ref_y[ilat, ilon]
    
    plt.ion()
    plt.show()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(221)
    ax1.plot(wavenumber_x[:, ilat, ilon], psd_ref_x[:, ilat, ilon], color='r',
             label='Zonal PSD reference field', lw=2)
    if flag_study_x:
        ax1.plot(wavenumber_x[:, ilat, ilon], psd_study_x[:, ilat, ilon], color='orange',
                 label='Zonal PSD study field', lw=2)
    ax1.set_xlabel("Wavenumber (cy/%s)" % distance_units_x, fontweight='bold')
    ax1.set_ylabel("Power spectral density (m2/(cy/%s))" % distance_units_x, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.legend(loc='best')
    fig1.suptitle("Spectral analysis at lon = %s and lat = %s"
                  % (str(np.round(lon[ilat, ilon], 2)), str(np.round(lat[ilat, ilon], 2))),
                  fontweight='bold', fontsize=35)
    ax2 = ax1.twiny()
    ax2.plot(1/wavenumber_x[:, ilat, ilon], psd_ref_x[:, ilat, ilon], color='r')
    ax2.invert_xaxis()
    ax2.set_xscale('log')
    ax2.set_xlabel("1/%s" % distance_units_x)
    plt.grid()

    ax3 = fig1.add_subplot(222)
    ax3.plot(distance_x[:, ilat, ilon], autocorrelation_ref_x[:, ilat, ilon], color='r',
             label='Zonal autocorrelation function reference field (zero-crossing = %5.0f %s )'
                   % (ref_zero_crossing_x, distance_units_x), lw=2)
    if flag_study_x:
        ax3.plot(distance_x[:, ilat, ilon], autocorrelation_study_x[:, ilat, ilon], color='orange',
             label='Zonal autocorrelation function study field (zero-crossing = %5.0f %s )'
                   % (study_zero_crossing_x, distance_units_x), lw=2)
    plt.axhline(y=0., color='k')
    ax3.set_xlabel("Distance (%s)" % distance_units_x, fontweight='bold')
    ax3.set_ylabel("Autocorrelation", fontweight='bold')
    plt.legend(loc='best')

    ax4 = fig1.add_subplot(223)
    ax4.plot(wavenumber_y[:, ilat, ilon], psd_ref_y[:, ilat, ilon], color='b',
             label='Meridional PSD reference field', lw=2)
    if flag_study_y:
        ax4.plot(wavenumber_y[:, ilat, ilon], psd_study_y[:, ilat, ilon], color='lightblue',
                 label='Meridional PSD study field', lw=2)
    ax4.set_xlabel("Wavenumber (cy/km)", fontweight='bold')
    ax4.set_ylabel("Power spectral density (m2/(cy/km))", fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    plt.legend(loc='best')
    ax5 = ax4.twiny()
    ax5.plot(1/wavenumber_y[:, ilat, ilon], psd_ref_y[:, ilat, ilon], color='b')
    ax5.invert_xaxis()
    ax5.set_xscale('log')
    ax5.set_xlabel("km")
    plt.grid()

    ax6 = fig1.add_subplot(224)
    ax6.plot(distance_y[:, ilat, ilon], autocorrelation_ref_y[:, ilat, ilon], color='b',
             label='Meridional autocorrelation function reference field (zero-crossing = %5.0f %s )'
                   % (ref_zero_crossing_y, distance_units_y), lw=2)
    if flag_study_y:
        ax6.plot(distance_y[:, ilat, ilon], autocorrelation_ref_y[:, ilat, ilon], color='lightblue',
                 label='Meridional autocorrelation function study field (zero-crossing = %5.0f %s )'
                       % (study_zero_crossing_y, distance_units_y), lw=2)
    plt.axhline(y=0., color='k')
    ax6.set_xlabel("Distance (%s)" % distance_units_y, fontweight='bold')
    ax6.set_ylabel("Autocorrelation", fontweight='bold')
    plt.legend(loc='best')

    plt.show()  


def onclick(event):
    """
    Function onclick
    :param event:
    :return:
    """
    toolbar = plt.get_current_fig_manager().toolbar
    if event.button == 1 and event.inaxes and toolbar.mode == '':
        # global indexx, indexy
        indexx, indexy = event.xdata, event.ydata
        print ' '
        print 'lon = ', indexx, ';', 'lat = ', indexy
        plt_spectral_analysis(indexx, indexy)
    else:
        print 'Clicked ouside axes bounds but inside plot window \n'


fig, ax = plt.subplots()
bmap = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-10, urcrnrlon=365, resolution='i')
bmap.drawcoastlines(zorder=5, linewidth=0.25)
bmap.fillcontinents(color='grey', lake_color='white')
bmap.drawparallels(arange(-90, 90, 30), labels=[1, 0, 0, 0], fontsize=20, dashes=[4, 4], linewidth=0.25)
bmap.drawmeridians(arange(-180, 360, 45), labels=[0, 0, 0, 1], fontsize=20, dashes=[4, 4], linewidth=0.25)
xx, yy = bmap(lon, lat)
scatter = bmap.scatter(xx, yy, c=nb_segment, s=2, lw=0)
cbar_scatter = bmap.colorbar(scatter, location='bottom', pad='15%')
cbar_scatter.set_label('Nb segment in computation')
ax.set_title("-- CLICK ON THE MAP TO DISPLAY SPECTRAL ANALYSIS --", fontweight='bold')
fig.canvas.callbacks.connect('button_press_event', onclick)
plt.show()
