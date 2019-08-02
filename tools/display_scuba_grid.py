"""
Display SCUBA analysis for spectral lovers...
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from math import sin, cos, atan2, pi, sqrt
from netCDF4 import Dataset
from sys import argv, path
from numpy import arange, zeros, meshgrid, ma
from matplotlib import rc, colors
from scipy.fftpack import fft
import scipy.interpolate
from scipy.optimize import curve_fit
from scipy.stats import chi2

path.insert(0, '../src/')
from mod_geo import *


input_file_x_direction = argv[1]
input_file_y_direction = argv[2]

nc = Dataset(input_file_x_direction)
wavenumber_x = nc.variables['freq'][:].filled(0)
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
nb_segment = nc.variables['nb_segment'][:, :]
psd_ref_x = nc.variables['psd_ref'][:, :, :]

flag_study_x = True

try:
    psd_study_x = nc.variables['psd_study'][:, :, :]
    coherence_x = nc.variables['coherence'][:, :, :]

except KeyError:
    flag_study_x = False
    psd_study_x = np.zeros(np.shape(psd_ref_x))
    coherence_x = np.zeros(np.shape(psd_ref_x))

nc.close()

nc = Dataset(input_file_y_direction)
wavenumber_y = nc.variables['freq'][:].filled(0)
nb_segment_y = nc.variables['nb_segment'][:, :]
psd_ref_y = nc.variables['psd_ref'][:, :, :]

flag_study_y = True

try:
    psd_study_y = nc.variables['psd_study'][:, :, :]
    coherence_y = nc.variables['coherence'][:, :, :]

except KeyError:
    flag_study_y = False
    psd_study_y = np.zeros(np.shape(psd_ref_y))
    coherence_y = np.zeros(np.shape(psd_ref_y))

nc.close()

if flag_study_y or flag_study_x:
    nb_subplot1 = 2
    nb_subplot2 = 2
else:
    nb_subplot1 = 1
    nb_subplot2 = 2


def confidence_interval(psd, nb_seg):
    probability = 0.95
    alfa = 1. - probability
    v = 2. * nb_seg
    c = chi2.ppf([1 - 0.5 * alfa, 0.5 * alfa], v)
    c = v / c
    psd_lower = psd * c[0]
    psd_upper = psd * c[1]

    return psd_lower, psd_upper


def plt_spectral_analysis(cix, ciy):
    """
    Plot spectrum and coherence at a specific (lon, lat) coordinates
    :param cix:
    :param ciy:
    :return:
    """

    ilon = find_nearest_index(lon, cix)
    ilat = find_nearest_index(lat, ciy)

    plt.ion()
    plt.show()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(nb_subplot1, nb_subplot2, 1)
    ax1.plot(wavenumber_x, psd_ref_x[:, ilat, ilon], color='r',
             label='Zonal PSD reference field', lw=2)

    psd_lower, psd_upper = confidence_interval(psd_ref_x[:, ilat, ilon], nb_segment[ilat, ilon])
    ax1.fill_between(wavenumber_x, psd_lower, psd_upper, color='r', alpha=0.2)
    ax1.plot(wavenumber_x, psd_lower, color='r', lw=0.5)
    ax1.plot(wavenumber_x, psd_upper, color='r', lw=0.5)

    if flag_study_x:
        ax1.plot(wavenumber_x, psd_study_x[:, ilat, ilon], color='orange',
                 label='Zonal PSD study field', lw=2)
        psd_lower, psd_upper = confidence_interval(psd_study_x[:, ilat, ilon], nb_segment[ilat, ilon])
        ax1.fill_between(wavenumber_x, psd_lower, psd_upper, color='r', alpha=0.2)
        ax1.plot(wavenumber_x, psd_lower, color='orange', lw=0.5)
        ax1.plot(wavenumber_x, psd_upper, color='orange', lw=0.5)

    ax1.set_xlabel("Wavenumber (cy/km)", fontweight='bold')
    ax1.set_ylabel("Power spectral density (m2/(cy/km))", fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.legend(loc='best')
    fig1.suptitle("Spectral analysis at lon = %s and lat = %s"
                  % (str(np.round(lon[ilon], 2)), str(np.round(lat[ilat], 2))),
                  fontweight='bold', fontsize=25)
    ax2 = ax1.twiny()
    ax2.plot(1/wavenumber_x, psd_ref_x[:, ilat, ilon], color='r')
    ax2.invert_xaxis()
    ax2.set_xscale('log')
    ax2.set_xlabel("Wavenumber (km)", fontweight='bold')
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax1.grid(which='major', axis='both')
    ax1.grid(ls='--', which='minor', axis='both')

    ax4 = fig1.add_subplot(nb_subplot1, nb_subplot2, 2)
    ax4.plot(wavenumber_y, psd_ref_y[:, ilat, ilon], color='b',
             label='Meridional PSD reference field', lw=2)
    psd_lower, psd_upper = confidence_interval(psd_ref_y[:, ilat, ilon], nb_segment_y[ilat, ilon])
    ax4.fill_between(wavenumber_y, psd_lower, psd_upper, color='b', alpha=0.2)
    ax4.plot(wavenumber_y, psd_lower, color='b', lw=0.5)
    ax4.plot(wavenumber_y, psd_upper, color='b', lw=0.5)

    if flag_study_y:
        ax4.plot(wavenumber_y, psd_study_y[:, ilat, ilon], color='lightblue',
                 label='Meridional PSD study field', lw=2)
        psd_lower, psd_upper = confidence_interval(psd_study_y[:, ilat, ilon], nb_segment[ilat, ilon])
        ax4.fill_between(wavenumber_y, psd_lower, psd_upper, color='r', alpha=0.2)
        ax4.plot(wavenumber_y, psd_lower, color='lightblue', lw=0.5)
        ax4.plot(wavenumber_y, psd_upper, color='lightblue', lw=0.5)
    ax4.set_xlabel("Wavenumber (cy/km)", fontweight='bold')
    ax4.set_ylabel("Power spectral density (m2/(cy/km))", fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    plt.legend(loc='best')
    ax5 = ax4.twiny()
    ax5.plot(1/wavenumber_y, psd_ref_y[:, ilat, ilon], color='b')
    ax5.invert_xaxis()
    ax5.set_xscale('log')
    ax5.set_xlabel("Wavenumber (km)", fontweight='bold')
    ax5.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax4.grid(which='major', axis='both')
    ax4.grid(ls='--', which='minor', axis='both')

    if flag_study_y:
        ax7 = fig1.add_subplot(nb_subplot1, nb_subplot2, 3)
        ax7.plot(wavenumber_x, coherence_x[:, ilat, ilon], color='r',
                 label='MSC zonal direction', lw=2)
        ax7.set_xlabel("Wavenumber (cy/km)", fontweight='bold')
        ax7.set_ylabel("Magnitude Squared Coherence", fontweight='bold')
        ax7.set_xscale('log')
        ax7.set_ylim(0, 1)
        ax7.grid(which='major', axis='both')
        plt.legend(loc='best')
        ax8 = ax7.twiny()
        ax8.plot(1 / wavenumber_x, coherence_x[:, ilat, ilon], color='r')
        ax8.invert_xaxis()
        ax8.set_xscale('log')
        ax8.set_xlabel("Wavenumber (km)", fontweight='bold')
        ax8.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax8.grid(which='major', axis='both')
        ax8.grid(ls='--', which='minor', axis='both')
        ax8.set_ylim(0, 1)

        ax9 = fig1.add_subplot(nb_subplot1, nb_subplot2, 4)
        ax9.plot(wavenumber_y, coherence_y[:, ilat, ilon], color='b',
                 label='MSC meridional direction', lw=2)
        ax9.set_xlabel("Wavenumber (cy/km)", fontweight='bold')
        ax9.set_ylabel("Magnitude Squared Coherence", fontweight='bold')
        ax9.set_xscale('log')
        ax9.set_ylim(0, 1)
        ax9.grid(which='major', axis='both')
        plt.legend(loc='best')
        ax10 = ax9.twiny()
        ax10.plot(1 / wavenumber_y, coherence_y[:, ilat, ilon], color='b')
        ax10.invert_xaxis()
        ax10.set_xscale('log')
        ax10.set_xlabel("Wavenumber (km)", fontweight='bold')
        ax10.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax10.grid(which='major', axis='both')
        ax10.grid(ls='--', which='minor', axis='both')
        ax10.set_ylim(0, 1)

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
        print(' ')
        print('lon = ', indexx, ';', 'lat = ', indexy)
        plt_spectral_analysis(indexx, indexy)
    else:
        print('Clicked ouside axes bounds but inside plot window \n')


fig, ax0 = plt.subplots()
projection = ccrs.PlateCarree()
ax = plt.axes(projection=projection)
ax.coastlines()
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.LAND)
ax.set_xticks(np.linspace(np.min(lon), np.max(lon), 5), crs=projection)
ax.set_yticks(np.linspace(np.min(lat), np.max(lat), 5), crs=projection)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
lons, lats = np.meshgrid(lon, lat)
delta_lon = 0.5*np.abs(lon[0]-lon[1])
delta_lat = 0.5*np.abs(lat[0]-lat[1])
pcolor = ax.pcolormesh(lons-delta_lon, lats-delta_lat, nb_segment, transform=projection)
cbar = fig.colorbar(pcolor)
cbar.set_label('Nb segment in computation', fontweight='bold')
ax.set_title("-- CLICK ON THE MAP TO DISPLAY SPECTRAL ANALYSIS --", fontweight='bold')
fig.canvas.callbacks.connect('button_press_event', onclick)
plt.show()
