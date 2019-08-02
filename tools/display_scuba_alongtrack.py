"""
Display SCUBA analysis for spectral lovers...
"""
import numpy as np
from matplotlib import pyplot as plt

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

from math import sin, cos, atan2, pi, sqrt
from netCDF4 import Dataset
from sys import argv, path
from numpy import arange, zeros, meshgrid, ma
from matplotlib import rc, colors
from scipy.fftpack import fft
import scipy.interpolate
from scipy.optimize import curve_fit
import cmocean
from scipy import interpolate

path.insert(0, '../src/')
from mod_geo import *


input_file = argv[1]

nc = Dataset(input_file)
wavenumber = nc.variables['wavenumber'][:]
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
nb_segment = nc.variables['nb_segment'][:, :]
psd_ref = nc.variables['psd_ref'][:, :, :]

flag_map = True
try:
    psd_map = nc.variables['psd_study'][:, :, :]
    coherence = nc.variables['coherence'][:, :, :]
    psd_diff = nc.variables['psd_diff'][:, :, :]

except:
    flag_map = False
    psd_map = np.zeros(np.shape(psd_ref))
    coherence = np.zeros(np.shape(psd_ref))
    psd_diff = np.zeros(np.shape(psd_ref))

nc.close()

lon = np.where(lon >= 360, lon-360, lon)
lon = np.where(lon < 0, lon+360, lon)


def func_y0(x, a):
    return 0*x + a


def func_y1(x, a, b):
    return a*x + b


def func_y2slope(x, a1, b1, a2, b2):
    return a1*x + b1 + a2*x + b2


wavenumber_hr = np.linspace(np.min(wavenumber), np.max(wavenumber), 10000)


def updatethetaaxis(axis):
    """

    :param axis:
    :return:
    """
    labels = []
    for f in axis.get_xticks():
        labels.append(np.round(np.cos(f), 2))
    axis.set_xticklabels(labels[::1], fontsize=10)
    return labels


def plt_spectral_taylor_diagram(ilon, ilat, ax):

    normalized_std = psd_map[:, ilat, ilon] / psd_ref[:, ilat, ilon]
    angle_coherence = np.arccos(coherence[:, ilat, ilon])

    plt.ion()
    plt.show()

    f = interpolate.interp1d(wavenumber, angle_coherence)
    angle_coherence_hr = f(wavenumber_hr)
    f = interpolate.interp1d(wavenumber, normalized_std)
    normalized_std_hr = f(wavenumber_hr)

    r_theta05 = np.arange(0, 1.2, 0.01)

    theta05 = np.arccos(0.5) * np.ones(np.shape(r_theta05))
    # theta0 = 0*np.ones(np.shape(r_theta05))
    theta_r05 = np.arccos(np.arange(0, 1.5, 0.01))
    r05 = 0.5 * np.ones(np.shape(theta_r05))
    r1 = np.ones(np.shape(theta_r05))
    tt = np.arange(0, np.arccos(0.49), 0.01)

    ax.fill_between(tt, 0.5, 1, color='0.3', alpha=0.3)
    # ax1.fill_betweenx(r_theta_area, theta0, theta05, color='0.3', alpha=0.5)
    # ax1.plot(angle_coherence, normalized_std, color='0.5', alpha=0.5)
    c = ax.scatter(angle_coherence_hr, normalized_std_hr,
                   c=1 / np.ma.masked_where(wavenumber_hr > 1. / 60., wavenumber_hr),
                   s=30, cmap='Spectral_r', lw=0, vmin=100, vmax=800)
    ax.scatter(0, 1, c='r', s=150, marker=(5, 1), zorder=10, facecolors='none', edgecolors='k')
    ax.plot(theta05, r_theta05, color='k', lw=2)
    ax.plot(theta_r05, r05, color='k', lw=2)
    ax.plot(theta_r05, r1, color='k', lw=2)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_rmax(1.2)
    plt.rc('grid', linestyle="--")
    ax.xaxis.grid(linestyle="--")
    ax.grid(True)
    updatethetaaxis(ax)
    ax.set_ylabel("Normalized PSD", fontweight='bold', fontsize=12)
    # ax.annotate("Spectral Taylor Diagram at lon = %s and lat = %s"
    #             % (str(np.round(lon[ilon], 2)), str(np.round(lat[ilat], 2))),
    #             xy=(0.3, 1.07), xycoords='axes fraction', color='black',
    #             bbox=dict(facecolor='none', edgecolor='black', pad=10.0))
    cbar = plt.colorbar(c, pad=0.1)
    cbar.set_label('Wavelength (km)', fontweight='bold')
    ax.annotate('Coherence', xy=(0.75, 0.85), xycoords='axes fraction', rotation=-50, fontweight='bold', fontsize=12)
    ax.annotate('REF', xy=(0.85, 0.04), xycoords='axes fraction', fontweight='bold', color='red', fontsize=12)
    ax.annotate('ratio = 0.5', xy=(pi / 3.5, 0.45), xycoords='data', rotation=-50, fontsize=8)
    ax.annotate('ratio = 1.0', xy=(pi / 4, 1.01), xycoords='data', rotation=-50, fontsize=8)
    ax.annotate('Coherence = 0.5', xy=(pi / 2.5, 0.82), xycoords='data', rotation=60, fontsize=8)


def plt_spectral_analysis(cix, ciy):
    """
    Plot spectrum and coherence at a specific (lon, lat) coordinates
    :param cix:
    :param ciy:
    :return:
    """

    ilon = find_nearest_index(lon % 360, cix % 360)
    ilat = find_nearest_index(lat, ciy)

    # Fit noise level
    spectrum_noise_limit = 23.5  # km
    index_frequency_23km = find_nearest_index(wavenumber, 1.0/spectrum_noise_limit)
    xdata = wavenumber[index_frequency_23km:]
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
    index_frequency_100km = find_nearest_index(wavenumber, freq1)
    index_frequency_250km = find_nearest_index(wavenumber, freq2)
    xdata_slope = np.log(wavenumber[index_frequency_250km:index_frequency_100km])
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
    min_wavenumber = np.min(1./wavenumber)
    #
    # map_effective_resolution = effective_resolution[ilat, ilon]
    # map_useful_resolution = useful_resolution[ilat, ilon]
    #
    # along_track_zero_crossing = zero_crossing_ref[ilat, ilon]
    # map_zero_crossing = zero_crossing_map[ilat, ilon]
    
    plt.ion()
    plt.show()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(221)
    ax1.axvspan(min_wavenumber, along_track_resolution, alpha=0.5, color='grey')
    ax1.plot(1/wavenumber, psd_ref[:, ilat, ilon], color='r', label='PSD along-track', lw=2)
    ax1.plot(1/wavenumber, psd_ref_denoised, color='orange', label='PSD along-track denoised', lw=2)
    if flag_map:
        ax1.plot(1/wavenumber, psd_map[:, ilat, ilon], color='b', label='PSD map', lw=2)
    ax1.plot(1/wavenumber, func_y0(wavenumber, *popt), 'r', label='Noise fit: y =%5.3f' % tuple(popt), lw=1, ls='--')
    ax1.plot(1/wavenumber, np.exp(popt_slope[0]*np.log(wavenumber)+popt_slope[1]), 'r',
             label='Slope fit: y =%5.3fx + %5.3f' % tuple(popt_slope), lw=1, ls=':')

    # ax1.plot(wavenumber[:, ilat, ilon], np.exp(popt_slope2[0]*np.log(wavenumber[:, ilat, ilon])+popt_slope2[1]), '',
    #          label='slope2 fit: y =%5.3fx + %5.3f' % (popt_slope2[0], popt_slope2[1]), lw=2, ls=':')
    
    # ax1.axvline(x=1./along_track_resolution, color='g',
    #             label='Along-track resolution =%5.0f km' % along_track_resolution, lw=3)
    # if flag_map:
    #     ax1.axvline(x=1./map_effective_resolution, color='lightgreen',
    #                 label='Map effective resolution =%5.0f km' % map_effective_resolution, lw=3)
    #     ax1.axvline(x=1./map_useful_resolution, color='lime',
    #                 label='Map useful resolution =%5.0f km' % map_useful_resolution, lw=3)
    ax1.set_xlabel("Wavenumber", fontweight='bold')
    ax1.set_ylabel("Power spectral density (m2/(cy/km))", fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.legend(loc='best')
    plt.xticks([50, 100, 200, 500, 1000], ["50km", "100km", "200km", "500km", "1000km"])
    ax1.invert_xaxis()
    plt.grid(which='both', linestyle='--')
    fig1.suptitle("Spectral analysis at lon = %s and lat = %s"
                  % (str(np.round(lon[ilon], 2)), str(np.round(lat[ilat], 2))),
                  fontweight='bold', fontsize=22)

    ax3 = fig1.add_subplot(222)
    if flag_map:
        ax3.plot(1/wavenumber, coherence[:, ilat, ilon], color='k',
                 label='MSC along-track / map', lw=2)
        # ax3.axvline(x=1./along_track_resolution, color='g',
        #             label='Along-track resolution =%5.0f km' % along_track_resolution, lw=3)
        # ax3.axvline(x=1./map_effective_resolution, color='lightgreen',
        #             label='Map effective resolution =%5.0f km' % map_effective_resolution, lw=3)
        # ax3.axvline(x=1./map_useful_resolution, color='lime',
        #             label='Map useful resolution =%5.0f km' % map_useful_resolution, lw=3)
    plt.axhline(y=0.5, color='r')
    ax3.set_xscale('log')
    ax3.set_xlabel("Wavenumber", fontweight='bold')
    ax3.set_ylabel("Magnitude squared coherence", fontweight='bold')
    ax3.invert_xaxis()
    plt.legend(loc='best')
    plt.grid(which='both', linestyle='--')
    plt.xticks([50, 100, 200, 500, 1000], ["50km", "100km", "200km", "500km", "1000km"])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ax5 = fig1.add_subplot(223, polar=True)
    plt_spectral_taylor_diagram(ilon, ilat, ax5)

    ax6 = fig1.add_subplot(224)
    if flag_map:
        ax6.plot(1/wavenumber, psd_map[:, ilat, ilon]/psd_ref[:, ilat, ilon], color='k',
                 label='PSD map / PSD along-track', lw=2)
        # ax6.axvline(x=1./along_track_resolution, color='g',
        #             label='Along-track resolution =%5.0f km' % along_track_resolution, lw=3)
        # ax6.axvline(x=1./map_effective_resolution, color='lightgreen',
        #             label='Map effective resolution =%5.0f km' % map_effective_resolution, lw=3)
        # ax6.axvline(x=1./map_useful_resolution, color='lime',
        #             label='Map useful resolution =%5.0f km' % map_useful_resolution, lw=3)
    plt.axhline(y=0.5, color='r')
    ax6.set_xscale('log')
    ax6.set_xlabel("Wavenumber", fontweight='bold')
    ax6.set_ylabel("Spectral ratio", fontweight='bold')
    plt.legend(loc='best')
    plt.grid(which='both', linestyle='--')
    ax6.invert_xaxis()
    plt.xticks([50, 100, 200, 500, 1000], ["50km", "100km", "200km", "500km", "1000km"])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
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
        print(' ')
        print('lon = ', indexx, ';', 'lat = ', indexy)
        plt_spectral_analysis(indexx, indexy)
    else:
        print('Clicked ouside axes bounds but inside plot window \n')


fig, _ = plt.subplots()
projection = ccrs.PlateCarree(central_longitude=0)
ax0 = plt.axes(projection=projection)
ax0.set_global()
ax0.coastlines(resolution='50m', lw=0.5, zorder=4)
gl = ax0.gridlines(draw_labels=True, linestyle='--', xlocs=[-120, -60, 0, 60, 120, 180, 240], zorder=6)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

ax0.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face',
                                            facecolor=cfeature.COLORS['land_alt1']),
                zorder=3)
pc1 = ax0.pcolormesh(lon, lat, nb_segment)
cb = plt.colorbar(pc1, orientation='vertical', fraction=0.024, pad=0.01)
cb.set_label('Nb segment in computation')
ax0.set_title("-- CLICK ON THE MAP TO DISPLAY SPECTRAL ANALYSIS --", fontweight='bold')
fig.canvas.callbacks.connect('button_press_event', onclick)
plt.show()
