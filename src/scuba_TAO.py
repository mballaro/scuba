from netCDF4 import Dataset, date2num, num2date
import numpy as np
import matplotlib.pylab as plt
import datetime
import glob
from sys import argv, exit
from os import path, walk
import re
from mpl_toolkits.basemap import Basemap
import datetime
import math
from scipy.fftpack import fft
import scipy.signal
import scipy.interpolate
from math import sqrt, cos, sin, asin, radians
from yaml import load
import fnmatch

from mod_io import *
from mod_constant import *
from mod_geo import *
from mod_segmentation import *
from mod_spectral import *
from mod_interpolation import *

if len(argv) != 2:
    print(' \n')
    print('USAGE   : %s <file.YAML> \n' % path.basename(argv[0]))
    print('Purpose : Spectral and spectral coherence computation tools for altimeter data \n')
    exit(0)

# Load analysis information from YAML file
YAML = load(open(str(argv[1])))
input_tao_file_directory = YAML['inputs']['input_tao_file_directory']
input_map_file = YAML['inputs']['input_map_file']
coef_corr_criterion = YAML['properties']['corr_coef']
lenght_scale = YAML['properties']['spectral_parameters']['lenght_scale']
delta_t = YAML['properties']['spectral_parameters']['delta_t']
output_nc_file = YAML['outputs']['output_filename']

debug = False

effective_resolution_out = []
useful_resolution_out = []
lat_out = []
lon_out = []
coherence_out = []
spectrum_TAO_out = []
spectrum_SSH_at_TAO = []
power_spectrum_TAO_out = []
power_spectrum_SSH_at_TAO = []
frequency_out = []

# Read map lon, lat, time vector
nc = Dataset(input_map_file, 'r')
lon_map = nc.variables['longitude'][:]
lat_map = nc.variables['latitude'][:]
time_map = nc.variables['time'][:]

for root, dirnames, filenames in walk(input_tao_file_directory):

    number_of_timeseries = 1

    for filename in fnmatch.filter(filenames, '*.cdf.gz'):

        # Read SSH TAO
        print("Start reading TAO file : %s" % filename)
        ssh_tao, time_tao, lat_tao, lon_tao = read_tao(path.join(root, filename))
        print("End reading TAO file")

        # Cleaning
        ssh_tao = np.ma.masked_invalid(np.ma.masked_where(np.abs(ssh_tao) > 1.E10, ssh_tao))
        # convert date to julian day
        date_time_tao = num2date(time_tao, units="days since 1950-01-01", calendar='julian')
        date_JD_tao = date2num(date_time_tao, units="days since 1950-01-01", calendar='julian')
        # mask date > 20170101
        index_max_time_tao = find_nearest_index(time_tao, 24472)
        date_JD_tao = date_JD_tao[:index_max_time_tao]
        ssh_tao = ssh_tao[:index_max_time_tao]

        # select closest maps point to the TAO
        index_lon = find_nearest_index(lon_map, lon_tao)
        index_lat = find_nearest_index(lat_map, lat_tao)
        index_min_time = find_nearest_index(time_map, np.min(date_JD_tao))
        index_max_time = find_nearest_index(time_map, np.max(date_JD_tao) + 1)

        print("Start reading SSH map timeserie")
        ssh_map = nc.variables['adt'][index_min_time:index_max_time, index_lat, index_lon]
        print("End reading SSH map timeserie")

        #ssh_map = np.ma.masked_where(ssh_tao.filled(-999) == -999, ssh_map)
        #date_JD_tao = np.ma.masked_where(ssh_tao.filled(-999) == -999, date_JD_tao)

        #ssh_map = ssh_map.compressed()
        #ssh_tao = ssh_tao.compressed()
        #time = date_JD_tao.compressed()

        # Compute correlation
        rcoeff = np.ma.corrcoef(ssh_map, ssh_tao)[0, 1]

        if debug:
            print rcoeff

        if rcoeff > coef_corr_criterion:

            print(" ")
            print('Analyzing file %s' % filename)
            # make coherent timeseries between TG and alti (remove missing date)
            try:
                ssh_tao = ssh_tao.filled(-999.)
            except:
                print("1 No missing value SSH TAO in file %s" % filename)
            try:
                ssh_map = ssh_map.filled(-999.)
            except:
                print("1 No missing value SSH map in file %s" % filename)

            ssh_map = np.ma.masked_where(((ssh_tao == -999.) | (ssh_map == -999.)), ssh_map)
            ssh_tao = np.ma.masked_where(((ssh_tao == -999.) | (ssh_map == -999.)), ssh_tao)
            date_JD_tao = np.ma.masked_where((ssh_tao == -999) | (ssh_map == -999), date_JD_tao)

            try:
                ssh_tao = ssh_tao.compressed()
            except:
                print("2 No missing value SSH TAO in file %s" % filename)
            try:
                ssh_map = ssh_map.compressed()
            except:
                print("2 No missing value SSH map")
            try:
                date_JD_tao = date_JD_tao.compressed()
            except:
                print("2 No missing value date_JD_tao %s" % filename)

            if debug:
                plt.plot(date_JD_tao, ssh_tao, label='SSH TAO')
                plt.plot(date_JD_tao, ssh_map, color='r', label='SSH map')
                plt.legend(loc='best')
                plt.show()

            # Prepare segment
            print("start segment computation", str(datetime.datetime.now()))

            computed_sla_segment, computed_msla_segment = \
                compute_segment_tide_gauge(ssh_tao, date_JD_tao, lenght_scale, delta_t, ssh_map)

            print("end segment computation", str(datetime.datetime.now()))

            # compute spectrum on grid
            print("start spectral computation", str(datetime.datetime.now()))

            output_mean_frequency, output_mean_psd_sla_ref, output_mean_ps_sla_ref, \
            output_autocorrelation_ref, output_autocorrelation_ref_zero_crossing, output_autocorrelation_distance, \
            output_mean_ps_sla_study, output_mean_ps_diff_sla_study_sla_ref, \
            output_mean_psd_sla_study, output_mean_psd_diff_sla_study_sla_ref, \
            output_mean_coherence, output_effective_resolution, output_useful_resolution, \
            output_autocorrelation_study, output_autocorrelation_study_zero_crossing = \
                spectral_computation_tide_gauge(np.asarray(computed_sla_segment).flatten(),
                                     delta_t, lenght_scale, sla_study_segments=np.asarray(computed_msla_segment).flatten())

            print("end spectral computation", str(datetime.datetime.now()))
            print(" ")

            if output_mean_frequency.size:
                effective_resolution_out.append(output_effective_resolution)
                useful_resolution_out.append(output_useful_resolution)
                lat_out.append(lat_tao)
                lon_out.append(lon_tao)
                coherence_out.append(output_mean_coherence)
                power_spectrum_TAO_out.append(output_mean_psd_sla_ref)
                power_spectrum_SSH_at_TAO.append(output_mean_psd_sla_study)
                spectrum_TAO_out.append(output_mean_psd_sla_ref)
                spectrum_SSH_at_TAO.append(output_mean_ps_sla_study)
                frequency_out.append(output_mean_frequency[:])

nc.close()

print("start writing", str(datetime.datetime.now()))
write_netcdf_TAO(output_nc_file, frequency_out, effective_resolution_out, useful_resolution_out,
                    lat_out, lon_out,
                    spectrum_TAO_out, spectrum_SSH_at_TAO,
                    power_spectrum_TAO_out, power_spectrum_SSH_at_TAO,
                    coherence_out)
print("end writing", str(datetime.datetime.now()))
