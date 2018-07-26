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
wavenumber = nc.variables['frequency'][:, :, :].filled(0)
lat = nc.variables['lat2D'][:, :]
lon = nc.variables['lon2D'][:, :]
nb_segment = nc.variables['nb_segment'][:, :]
psd_ref = nc.variables['psd_ref'][:, :, :]
autocorrelation_ref = nc.variables['autocorrelation_ref'][:, :, :]
distance = nc.variables['distance'][:, :, :]
distance_units = nc.variables['distance'].units
zero_crossing_ref = nc.variables['zero_crossing_ref'][:, :]

flag_map = True
try:
    psd_map = nc.variables['psd_study'][:, :, :]
    coherence = nc.variables['coherence'][:, :, :]
    effective_resolution = nc.variables['effective_resolution'][:, :]
    useful_resolution = nc.variables['useful_resolution'][:, :]
    autocorrelation_map = nc.variables['autocorrelation_study'][:, :, :]
    zero_crossing_map = nc.variables['zero_crossing_study'][:, :]

except:
    flag_map = False
    psd_map = np.zeros(np.shape(psd_ref))
    coherence = np.zeros(np.shape(psd_ref))
    effective_resolution = np.zeros(np.shape(lon))
    useful_resolution = np.zeros(np.shape(lon))
    autocorrelation_map = np.zeros(np.shape(psd_ref))
    zero_crossing_map = np.zeros(np.shape(lon))

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

    # Fit noise level
    spectrum_noise_limit = 23.5  # km
    index_frequency_23km = find_nearest_index(wavenumber[:, ilat, ilon], 1.0/spectrum_noise_limit)
    xdata = wavenumber[index_frequency_23km:, ilat, ilon]
    ydata = psd_ref[index_frequency_23km:, ilat, ilon]
    popt, pcov = curve_fit(func_y0, xdata, ydata)
    # print "Noise level PSD (m2/km)= ", popt
    # print "Noise level PSD np.sqrt((m2/km))= ", np.sqrt(popt)
    # print "Noise level PSD np.sqrt((m2/km)/100)= ", np.sqrt(popt/100)
    
    # index_frequency_100km = find_nearest_index(wavenumber[:, ilat, ilon], 1.0/100.)
    # delta_frequency = np.diff(wavenumber[:, ilat, ilon])
    # area = np.trapz(psd_ref[index_frequency_100km:, ilat, ilon], x=wavenumber[index_frequency_100km:, ilat, ilon],
    #                 dx=delta_frequency)
    # print "Area Noise level PSD (m) = " , np.sqrt(area)
    
    # xdata2 = wavenumber[index_frequency_23km:, ilat, ilon]
    # ydata2 = PS_along_track[index_frequency_23km:, ilat, ilon]
    # popt2, pcov2 = curve_fit(func_y0, xdata2, ydata2)
    # print "Noise level PS (m)= ", np.sqrt(popt2)
    
    # Denoised spectrum
    psd_ref_denoised = np.ma.masked_invalid(psd_ref[:, ilat, ilon] - popt[0])

    # Fit two slope denoised spectrum
    # light_freq = np.ma.masked_where(wavenumber[:, ilat, ilon] < 0.004, wavenumber[:, ilat, ilon])
    # xdata_2slope = np.log(np.ma.masked_where(wavenumber[:, ilat, ilon] < 0.004, wavenumber[:, ilat, ilon]))
    # ydata_2slope = np.log(np.ma.masked_where(wavenumber[:, ilat, ilon] < 0.004, psd_ref_denoised))
    # popt_2slope, pcov_2slope = curve_fit(func_y2slope, xdata_2slope, ydata_2slope,
    #                                      bounds=([-4, -100, -4, -100], [-2, 100, -2, 100]))
        
    # Fit slope
    freq1 = 1./100.
    freq2 = 1./250.
    index_frequency_100km = find_nearest_index(wavenumber[:, ilat, ilon], freq1)
    index_frequency_250km = find_nearest_index(wavenumber[:, ilat, ilon], freq2)
    xdata_slope = np.log(wavenumber[index_frequency_250km:index_frequency_100km, ilat, ilon])
    ydata_slope = np.log(psd_ref[index_frequency_250km:index_frequency_100km, ilat, ilon]-popt[0])
    popt_slope, pcov_slope = curve_fit(func_y1, xdata_slope, ydata_slope)
    
    # Fit slope2
    # freq1 = 1./30.
    # freq2 = 1./100.
    # index_frequency_10km = find_nearest_index(wavenumber[:, ilat, ilon], freq1)
    # index_frequency_100km = find_nearest_index(wavenumber[:, ilat, ilon], freq2)
    # xdata_slope = np.log(wavenumber[index_frequency_100km:index_frequency_10km, ilat, ilon])
    # ydata_slope = np.log(psd_ref[index_frequency_100km:index_frequency_10km, ilat, ilon]-popt[0])
    # popt_slope2, pcov_slope2 = curve_fit(func_y1, xdata_slope, ydata_slope)
            
    along_track_resolution = np.exp(-(np.log(popt[0]) - popt_slope[1])/popt_slope[0])

    map_effective_resolution = effective_resolution[ilat, ilon]
    map_useful_resolution = useful_resolution[ilat, ilon]

    along_track_zero_crossing = zero_crossing_ref[ilat, ilon]
    map_zero_crossing = zero_crossing_map[ilat, ilon]
    
    plt.ion()
    plt.show()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(221)
    ax1.plot(wavenumber[:, ilat, ilon], psd_ref[:, ilat, ilon], color='r',
             label='PSD along-track', lw=2)
    ax1.plot(wavenumber[:, ilat, ilon], psd_ref_denoised, color='orange',
             label='PSD along-track denoised', lw=2)
    if flag_map:
        ax1.plot(wavenumber[:, ilat, ilon], psd_map[:, ilat, ilon], color='b',
                 label='PSD map', lw=2)
    ax1.plot(wavenumber[:, ilat, ilon], func_y0(wavenumber[:, ilat, ilon], *popt), 'r',
             label='Noise fit: y =%5.3f' % tuple(popt), lw=2, ls='--')
    ax1.plot(wavenumber[:, ilat, ilon], np.exp(popt_slope[0]*np.log(wavenumber[:, ilat, ilon])+popt_slope[1]), 'r',
             label='Slope fit: y =%5.3fx + %5.3f' % tuple(popt_slope), lw=2, ls=':')

    # ax1.plot(wavenumber[:, ilat, ilon], np.exp(popt_slope2[0]*np.log(wavenumber[:, ilat, ilon])+popt_slope2[1]), '',
    #          label='slope2 fit: y =%5.3fx + %5.3f' % (popt_slope2[0], popt_slope2[1]), lw=2, ls=':')
    
    ax1.axvline(x=1./along_track_resolution, color='g',
                label='Along-track resolution =%5.0f km' % along_track_resolution, lw=3)
    if flag_map:
        ax1.axvline(x=1./map_effective_resolution, color='lightgreen',
                    label='Map effective resolution =%5.0f km' % map_effective_resolution, lw=3)
        ax1.axvline(x=1./map_useful_resolution, color='lime',
                    label='Map useful resolution =%5.0f km' % map_useful_resolution, lw=3)
    ax1.set_xlabel("Wavenumber (cy/km)", fontweight='bold')
    ax1.set_ylabel("Power spectral density (m2/(cy/km))", fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.legend(loc='best')
    fig1.suptitle("Spectral analysis at lon = %s and lat = %s"
                  % (str(np.round(lon[ilat, ilon], 2)), str(np.round(lat[ilat, ilon], 2))),
                  fontweight='bold', fontsize=35)
    ax2 = ax1.twiny()
    ax2.plot(1/wavenumber[:, ilat, ilon], psd_ref[:, ilat, ilon], color='r')
    if flag_map:
        ax2.plot(1/wavenumber[:, ilat, ilon], psd_map[:, ilat, ilon], color='b', lw=1)
    ax2.invert_xaxis()
    ax2.set_xscale('log')
    ax2.set_xlabel("km")
    plt.grid()

    ax3 = fig1.add_subplot(222)
    if flag_map:
        ax3.plot(wavenumber[:, ilat, ilon], coherence[:, ilat, ilon], color='k',
                 label='Magnitude squared coherence along-track / map', lw=2)
        ax3.axvline(x=1./along_track_resolution, color='g',
                    label='Along-track resolution =%5.0f km' % along_track_resolution, lw=3)
        ax3.axvline(x=1./map_effective_resolution, color='lightgreen',
                    label='Map effective resolution =%5.0f km' % map_effective_resolution, lw=3)
        ax3.axvline(x=1./map_useful_resolution, color='lime',
                    label='Map useful resolution =%5.0f km' % map_useful_resolution, lw=3)
    plt.axhline(y=0.5, color='r')
    ax3.set_xscale('log')
    ax3.set_xlabel("Wavenumber (cy/km)", fontweight='bold')
    ax3.set_ylabel("Coherence", fontweight='bold')
    plt.legend(loc='best')
    ax4 = ax3.twiny()
    ax4.plot(1/wavenumber[:, ilat, ilon], coherence[:, ilat, ilon], color='k', lw=1)
    ax4.invert_xaxis()
    ax4.set_xscale('log')
    ax4.set_xlabel("km")
    plt.legend(loc='best')
    plt.grid()

    ax5 = fig1.add_subplot(223)
    ax5.plot(distance[:, ilat, ilon], autocorrelation_ref[:, ilat, ilon], color='r',
             label='Autocorrelation function along-track (zero-crossing = %5.0f %s )'
                   % (along_track_zero_crossing, distance_units), lw=2)
    if flag_map:
        ax5.plot(distance[:, ilat, ilon], autocorrelation_map[:, ilat, ilon], color='b',
                 label='Autocorrelation function map (zero-crossing = %5.0f %s)'
                       % (map_zero_crossing, distance_units), lw=2)
    plt.axhline(y=0., color='k')
    ax5.set_xlabel("Distance (%s)" % distance_units, fontweight='bold')
    ax5.set_ylabel("Autocorrelation", fontweight='bold')
    plt.legend(loc='best')

    ax6 = fig1.add_subplot(224)
    if flag_map:
        ax6.plot(wavenumber[:, ilat, ilon], psd_map[:, ilat, ilon]/psd_ref[:, ilat, ilon], color='k',
                 label='Spectral ratio (PSD map / PSD along-track)', lw=2)
        ax6.axvline(x=1./along_track_resolution, color='g',
                    label='Along-track resolution =%5.0f km' % along_track_resolution, lw=3)
        ax6.axvline(x=1./map_effective_resolution, color='lightgreen',
                    label='Map effective resolution =%5.0f km' % map_effective_resolution, lw=3)
        ax6.axvline(x=1./map_useful_resolution, color='lime',
                    label='Map useful resolution =%5.0f km' % map_useful_resolution, lw=3)
    plt.axhline(y=0.5, color='r')
    ax6.set_xscale('log')
    ax6.set_xlabel("Wavenumber (cy/km)", fontweight='bold')
    ax6.set_ylabel("PSD map / PSD along-track", fontweight='bold')
    plt.legend(loc='best')
    ax7 = ax6.twiny()
    if flag_map:
        ax7.plot(1/wavenumber[:, ilat, ilon], psd_map[:, ilat, ilon]/psd_ref[:, ilat, ilon], color='k', lw=1)
    ax7.invert_xaxis()
    ax7.set_xscale('log')
    ax7.set_xlabel("km")
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
