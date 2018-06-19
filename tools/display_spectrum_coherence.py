#!/usr/bin/python
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic
from math import sin, cos, atan2, pi, sqrt
from netCDF4 import Dataset
from sys import argv
from numpy import arange, zeros, meshgrid, ma
from matplotlib import rc, colors
rc('text', usetex=True)
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


def func_y0(x, a):
    return 0*x + a


def func_y1(x, a, b):
    return a*x + b


def func_y2slope(x, a1, b1, a2, b2):
    return a1*x + b1 + a2*x + b2 


def compute_reachable_resolution(coherence, frequency):
    """                                                                                                                                                                                                            
    Given a coherence profile, compute the effective resolution (i.e., where coherence = 0.5)
    :param coherence:
    :param frequency:
    :return:
    """
    # find index where coherence = 0.25
    try:
        ii025 = np.where(coherence[:] > 0.25)[0][-1]
    except:
        ii025 = -1
        
    if (ii025 + 1 < coherence[:].size) and (ii025 + 1 >= 0):
        d1 = coherence[ii025] - 0.25
        d2 = 0.25 - coherence[ii025 + 1]
        logres = np.log(frequency[ii025]) * (d2 / (d1 + d2)) + np.log(frequency[ii025 + 1]) * (
            d1 / (d1 + d2))
        reachable_resolution = 1. / np.exp(logres)
        
    else:
        # print "Error: ", ii, jj
        # plt.plot(1/frequency[:, jj, ii], coherence[:, jj, ii])
        # plt.show()
        reachable_resolution = 0.
    
    return reachable_resolution


def plt_spectrum_coherence(cix, ciy):
    """

    """
    ilon, ilat = find_nearest_index(LON, LAT, cix, ciy)

    # Compute reachable resolution
    reachable_resolution = compute_reachable_resolution(COHERENCE[:, ilat, ilon], FREQ[:, ilat, ilon])
        
    # Fit noise level
    spectrum_noise_limit = 23.5
    index_frequency_23km = find_nearest_index1D(FREQ[:, ilat, ilon], 1.0/spectrum_noise_limit)
    xdata = FREQ[index_frequency_23km:, ilat, ilon]
    ydata = PSD_along_track[index_frequency_23km:, ilat, ilon]
    popt, pcov = curve_fit(func_y0, xdata, ydata)
    print "Noise level PSD (m2/km)= ", popt
    print "Noise level PSD np.sqrt((m2/km))= ", np.sqrt(popt)
    print "Noise level PSD np.sqrt((m2/km)/100)= ", np.sqrt(popt/100)
    
    index_frequency_100km = find_nearest_index1D(FREQ[:, ilat, ilon], 1.0/100.)
    delta_frequency = np.diff(FREQ[:, ilat, ilon])
    area = np.trapz(PSD_along_track[index_frequency_100km:, ilat, ilon], x=FREQ[index_frequency_100km:, ilat, ilon], dx=delta_frequency)
    print "Area Noise level PSD (m) = " , np.sqrt(area)
    
    xdata2 = FREQ[index_frequency_23km:, ilat, ilon]
    ydata2 = PS_along_track[index_frequency_23km:, ilat, ilon]
    popt2, pcov2 = curve_fit(func_y0, xdata2, ydata2)
    print "Noise level PS (m)= ", np.sqrt(popt2)
    
    # Denoised spectrum
    PSD_along_track_denoised = np.ma.masked_invalid(PSD_along_track[:, ilat, ilon] - popt[0])

    # Fit two slope denoised spectrum
    light_freq = np.ma.masked_where(FREQ[:, ilat, ilon]<0.004, FREQ[:, ilat, ilon])
    xdata_2slope = np.log(np.ma.masked_where(FREQ[:, ilat, ilon]<0.004, FREQ[:, ilat, ilon]))
    ydata_2slope = np.log(np.ma.masked_where(FREQ[:, ilat, ilon]<0.004, PSD_along_track_denoised))
    popt_2slope, pcov_2slope = curve_fit(func_y2slope, xdata_2slope, ydata_2slope, bounds=([-4, -100, -4, -100], [-2, 100, -2, 100]))
        
    # Fit slope
    freq1 = 1./100.
    freq2 = 1./250.
    index_frequency_100km = find_nearest_index1D(FREQ[:, ilat, ilon], freq1)
    index_frequency_250km = find_nearest_index1D(FREQ[:, ilat, ilon], freq2)
    xdata_slope = np.log(FREQ[index_frequency_250km:index_frequency_100km, ilat, ilon])
    ydata_slope = np.log(PSD_along_track[index_frequency_250km:index_frequency_100km, ilat, ilon]-popt[0])
    popt_slope, pcov_slope = curve_fit(func_y1, xdata_slope, ydata_slope)
    
    # Fit slope2
    freq1 = 1./30.
    freq2 = 1./100.
    index_frequency_10km = find_nearest_index1D(FREQ[:, ilat, ilon], freq1)
    index_frequency_100km = find_nearest_index1D(FREQ[:, ilat, ilon], freq2)
    xdata_slope = np.log(FREQ[index_frequency_100km:index_frequency_10km, ilat, ilon])
    ydata_slope = np.log(PSD_along_track[index_frequency_100km:index_frequency_10km, ilat, ilon]-popt[0])
    popt_slope2, pcov_slope2 = curve_fit(func_y1, xdata_slope, ydata_slope)
            
    along_track_resolution = np.exp(-(np.log(popt[0]) - popt_slope[1])/popt_slope[0] )

    map_resolution = MAP_EFFECTIVE_RESOLUTION[ilat, ilon]
    
    plt.ion()
    plt.show()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)    
    ax1.plot(FREQ[:, ilat, ilon], PSD_along_track[:, ilat, ilon], color= 'r', label='Spectrum along-track', lw=2)
    ax1.plot(FREQ[:, ilat, ilon], PSD_along_track_denoised, color= 'r', label='Spectrum along-track denoised', lw=1)
    ax1.plot(FREQ[:, ilat, ilon], PSD_map[:, ilat, ilon], color= 'b', label='Spectrum map', lw=2)
    ax1.plot(FREQ[:, ilat, ilon], func_y0(FREQ[:, ilat, ilon], *popt), 'k', label='noise fit: y =%5.3f' % tuple(popt), lw=2, ls='--')
    ax1.plot(FREQ[:, ilat, ilon], np.exp(popt_slope[0]*np.log(FREQ[:, ilat, ilon])+popt_slope[1]), 'k', label='slope fit: y =%5.3fx + %5.3f' % tuple(popt_slope), lw=2, ls=':')
    
    ax1.plot(FREQ[:, ilat, ilon], np.exp(popt_slope[0]*np.log(FREQ[:, ilat, ilon])+popt_slope[1]), 'k', label='slope1 fit: y =%5.3fx + %5.3f' % (popt_slope[0], popt_slope[1]), lw=2, ls=':')
    ax1.plot(FREQ[:, ilat, ilon], np.exp(popt_slope2[0]*np.log(FREQ[:, ilat, ilon])+popt_slope2[1]), 'k', label='slope2 fit: y =%5.3fx + %5.3f' % (popt_slope2[0], popt_slope2[1]), lw=2, ls=':')
    
    ax1.axvline(x=1./along_track_resolution, color='g', label='along-track resolution=%5.0f km' %along_track_resolution, lw=3)
    ax1.axvline(x=1./map_resolution, color='lightgreen', label='map resolution=%5.0f km' %map_resolution, lw=3)
    ax1.axvline(x=1./reachable_resolution,  color='lime', label='map limit resolution=%5.0f km' %reachable_resolution, lw=3)
    ax1.set_xlabel("Wavenumber (cy/km)", fontweight='bold')
    ax1.set_ylabel("Spectral power density (m2/(cy/km))", fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    plt.legend(loc='best')
    fig1.suptitle("Map and along-track spectum at lon = %s and lat = %s" %(str(np.round(LON[ilat, ilon],2)), str(np.round(LAT[ilat, ilon],2))) )
    ax2 = ax1.twiny()
    ax2.plot(1/FREQ[:, ilat, ilon], PSD_along_track[:, ilat, ilon], color= 'r')
    ax2.plot(1/FREQ[:, ilat, ilon], PSD_map[:, ilat, ilon], color= 'b', lw=1)
    ax2.invert_xaxis()
    ax2.set_xscale('log')
    ax2.set_xlabel("km")
    plt.grid()

    #fig2 = plt.figure(3)
    ax3 = fig1.add_subplot(212)
    ax3.plot(FREQ[:, ilat, ilon], COHERENCE[:, ilat, ilon], color= 'k', label='Coherence Spectrum along-track / map', lw=2)
    ax3.axvline(x=1./along_track_resolution, color='g', label='along-track resolution=%5.0f km' %along_track_resolution, lw=3)
    ax3.axvline(x=1./map_resolution, color='lightgreen', label='map resolution=%5.0f km' %map_resolution, lw=3)
    ax3.axvline(x=1./reachable_resolution,  color='lime', label='map limit resolution=%5.0f km' %reachable_resolution, lw=3)
    plt.axhline(y=0.25, color='r')
    plt.axhline(y=0.5, color='r')
    ax3.set_xscale('log')
    ax3.set_xlabel("Wavenumber (cy/km)", fontweight='bold')
    ax3.set_ylabel("Coherence", fontweight='bold')
    plt.legend(loc='best')
    ax4 = ax3.twiny()
    ax4.plot(1/FREQ[:, ilat, ilon], COHERENCE[:, ilat, ilon], color= 'k', lw=1)
    ax4.invert_xaxis()
    #fig2.suptitle("Coherence between map and along-track spectum at lon = %s and lat = %s" %(str(np.round(LON[ilat, ilon],2)), str(np.round(LAT[ilat, ilon],2))))
    ax4.set_xscale('log')
    ax4.set_xlabel("km")
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
        #global indexx, indexy
        indexx, indexy = event.xdata, event.ydata
        print ' '
        print 'lon = ', indexx, ';', 'lat = ', indexy
        plt_spectrum_coherence(indexx, indexy)
    else:
        print 'Clicked ouside axes bounds but inside plot window \n'


        
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
