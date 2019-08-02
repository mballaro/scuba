"""
Display SCUBA analysis for spectral lovers...
"""
import numpy as np
from matplotlib import pyplot as plt
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
output_file = argv[2]

nc = Dataset(input_file)
wavenumber = nc.variables['wavenumber'][:]
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
psd_ref = nc.variables['psd_ref'][:, :, :]
nc.close()

lon = np.where(lon >= 360, lon-360, lon)
lon = np.where(lon < 0, lon+360, lon)

spectrum_noise_limit = 23.5  # in km

debug = False


def func_y0(x, a):
    return 0*x + a


def func_y1(x, a, b):
    return a*x + b


def func_y2slope(x, a1, b1, a2, b2):
    return a1*x + b1 + a2*x + b2


def compute_resolution_limit(wavenumber, psd):
    # Fit noise level
    index_frequency_23km = find_nearest_index(wavenumber, 1.0 / spectrum_noise_limit)
    xdata = wavenumber[index_frequency_23km:]
    ydata = psd[index_frequency_23km:]
    popt, pcov = curve_fit(func_y0, xdata, ydata)

    # Denoised spectrum
    psd_ref_denoised = np.ma.masked_invalid(psd - popt[0])

    # Fit slope
    freq1 = 1. / 100.
    freq2 = 1. / 250.
    index_frequency_100km = find_nearest_index(wavenumber, freq1)
    index_frequency_250km = find_nearest_index(wavenumber, freq2)
    xdata_slope = np.log(wavenumber[index_frequency_250km:index_frequency_100km])
    ydata_slope = np.log(psd[index_frequency_250km:index_frequency_100km] - popt[0])
    popt_slope, pcov_slope = curve_fit(func_y1, xdata_slope, ydata_slope)

    along_track_resolution = np.exp(-(np.log(popt[0]) - popt_slope[1]) / popt_slope[0])

    if debug:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(wavenumber, psd, color='r', label='PSD along-track', lw=2)
        ax1.plot(wavenumber, psd_ref_denoised, color='orange', label='PSD along-track denoised', lw=2)
        ax1.plot(wavenumber, func_y0(wavenumber, *popt), 'k', label='Noise fit: y =%5.3f' % tuple(popt), lw=1, ls='--')
        ax1.plot(wavenumber, np.exp(popt_slope[0] * np.log(wavenumber) + popt_slope[1]), 'k',
                 label='Slope fit: y =%5.3fx + %5.3f' % tuple(popt_slope), lw=1, ls=':')

        ax1.axvline(x=1. / along_track_resolution, color='g',
                    label='Along-track resolution =%5.0f km' % along_track_resolution, lw=3)
        ax1.set_xlabel("Wavenumber (cy/km)", fontweight='bold')
        ax1.set_ylabel("Power spectral density (m2/(cy/km))", fontweight='bold')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        plt.legend(loc='best')
        ax2 = ax1.twiny()
        ax2.plot(1 / wavenumber, psd, color='r')
        ax2.invert_xaxis()
        ax2.set_xscale('log')
        ax2.set_xlabel("km")
        plt.grid()
        plt.show()

    return along_track_resolution


resolution_limit = np.zeros((lat.size, lon.size))
for jj in range(lat.size):
    for ii in range(lon.size):
        if len(psd_ref[:, jj, ii].compressed()) > 0:
            resolution_limit[jj, ii] = compute_resolution_limit(wavenumber, psd_ref[:, jj, ii])


nc_out = Dataset(output_file, 'w', format='NETCDF4')
nc_out.createDimension('lat', lat.size)
nc_out.createDimension('lon', lon.size)
lat_out = nc_out.createVariable('lat', 'f8', ('lat',))
lat_out[:] = lat
lon_out = nc_out.createVariable('lon', 'f8', ('lon',))
lon_out[:] = lon
resolution_out = nc_out.createVariable('resolution_limit', 'f8', ('lat', 'lon'))
resolution_out.coordinates = "lat lon"
resolution_out.long_name = "resolution limit of alongtrack as defined by Dufau et al(2016)"
resolution_out[:, :] = np.ma.masked_invalid(np.ma.masked_outside(resolution_limit, 1, 500))

nc_out.close()
