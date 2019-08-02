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


input_file = argv[1]

nc = Dataset(input_file)
wavenumber = nc.variables['frequency'][:, :]
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
psd_ref = nc.variables['power_spectrum_TG'][:, :]
psd_map = nc.variables['power_spectrum_alti'][:, :]
coherence = nc.variables['coherence'][:, :]
effective_resolution = nc.variables['effective_resolution'][:]
useful_resolution = nc.variables['useful_resolution'][:]
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

    index = find_nearest_common_index(lon, lat, cix, ciy)

    map_effective_resolution = effective_resolution[index]
    map_useful_resolution = useful_resolution[index]

    plt.ion()
    plt.show()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(221)
    ax1.plot(wavenumber[index, :], psd_ref[index, :], color='r', label='PSD tide gauge', lw=2)
    ax1.plot(wavenumber[index, :], psd_map[index, :], color='b', label='PSD map', lw=2)
    ax1.axvline(x=1./map_effective_resolution, color='lightgreen',
                    label='Map effective resolution =%5.0f days' % map_effective_resolution, lw=3)
    ax1.axvline(x=1./map_useful_resolution, color='lime',
                    label='Map useful resolution =%5.0f days' % map_useful_resolution, lw=3)
    ax1.set_xlabel("Wavenumber (cy/days)", fontweight='bold')
    ax1.set_ylabel("Power spectral density (m2/(cy/days))", fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.legend(loc='best')
    fig1.suptitle("Spectral analysis at lon = %s and lat = %s"
                  % (str(np.round(lon[index], 2)), str(np.round(lat[index], 2))),
                  fontweight='bold', fontsize=35)
    ax2 = ax1.twiny()
    ax2.plot(1/wavenumber[index, :], psd_ref[index, :], color='r')
    ax2.plot(1/wavenumber[index, :], psd_map[index, :], color='b', lw=1)
    ax2.invert_xaxis()
    ax2.set_xscale('log')
    ax2.set_xlabel("days")
    plt.grid()

    ax3 = fig1.add_subplot(222)
    ax3.plot(wavenumber[index, :], coherence[index, :], color='k',
        label='Magnitude squared coherence along-track / map', lw=2)
    ax3.axvline(x=1./map_effective_resolution, color='lightgreen',
        label='Map effective resolution =%5.0f km' % map_effective_resolution, lw=3)
    ax3.axvline(x=1./map_useful_resolution, color='lime',
        label='Map useful resolution =%5.0f km' % map_useful_resolution, lw=3)
    plt.axhline(y=0.5, color='r')
    ax3.set_xscale('log')
    ax3.set_xlabel("Wavenumber (cy/days)", fontweight='bold')
    ax3.set_ylabel("Coherence", fontweight='bold')
    plt.legend(loc='best')
    ax4 = ax3.twiny()
    ax4.plot(1/wavenumber[index, :], coherence[index, :], color='k', lw=1)
    ax4.invert_xaxis()
    ax4.set_xscale('log')
    ax4.set_xlabel("days")
    plt.legend(loc='best')
    plt.grid()

    ax6 = fig1.add_subplot(224)
    ax6.plot(wavenumber[index, :], psd_map[index, :]/psd_ref[index, :], color='k',
        label='Spectral ratio (PSD map / PSD tide gauge)', lw=2)
    ax6.axvline(x=1./map_effective_resolution, color='lightgreen',
        label='Map effective resolution =%5.0f days' % map_effective_resolution, lw=3)
    ax6.axvline(x=1./map_useful_resolution, color='lime',
        label='Map useful resolution =%5.0f days' % map_useful_resolution, lw=3)
    plt.axhline(y=0.5, color='r')
    ax6.set_xscale('log')
    ax6.set_xlabel("Wavenumber (cy/days)", fontweight='bold')
    ax6.set_ylabel("PSD map / PSD tide gauge", fontweight='bold')
    plt.legend(loc='best')
    ax7 = ax6.twiny()
    ax7.plot(1/wavenumber[index, :], psd_map[index, :]/psd_ref[index, :], color='k', lw=1)
    ax7.invert_xaxis()
    ax7.set_xscale('log')
    ax7.set_xlabel("days")
    plt.legend(loc='best')
    plt.grid()

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
bmap = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-5, urcrnrlon=365, resolution='i')
bmap.drawcoastlines(zorder=5, linewidth=0.25)
bmap.fillcontinents(color='grey', lake_color='white', zorder=0)
bmap.drawparallels(arange(-90, 90, 30), labels=[1, 0, 0, 0], fontsize=20, dashes=[4, 4], linewidth=0.25)
bmap.drawmeridians(arange(-180, 360, 45), labels=[0, 0, 0, 1], fontsize=20, dashes=[4, 4], linewidth=0.25)
xx, yy = bmap(lon, lat)
scatter = bmap.scatter(xx, yy, c=effective_resolution, s=40, lw=0, vmin = 10, vmax=40)
cbar_scatter = bmap.colorbar(scatter, location='bottom', pad='15%')
cbar_scatter.set_label('Effective_resolution (days)', fontweight='bold')
ax.set_title("-- CLICK ON THE MAP TO DISPLAY SPECTRAL ANALYSIS --", fontweight='bold')
fig.canvas.callbacks.connect('button_press_event', onclick)
plt.show()
