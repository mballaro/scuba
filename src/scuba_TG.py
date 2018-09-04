from netCDF4 import Dataset, date2num
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
input_file_directory = YAML['inputs']['input_file_directory']
coef_corr_criterion = YAML['properties']['corr_coef']
lenght_scale = YAML['properties']['spectral_parameters']['lenght_scale']
delta_t = YAML['properties']['spectral_parameters']['delta_t']
output_nc_file = YAML['outputs']['output_filename']

effective_resolution_out = []
useful_resolution_out = []
lat_out = []
lon_out = []
coherence_out = []
spectrum_TG_out = []
spectrum_SLA_at_TG = []
power_spectrum_TG_out = []
power_spectrum_SLA_at_TG = []
frequency_out = []

for root, dirnames, filenames in walk(input_file_directory):

    number_of_timeseries = 1

    for filename in fnmatch.filter(filenames, 'SLA*.nc'):

        # Read SLA tide-gauge and map of SLA
        sla_tg, sla_alti, time_tg, lat_tg, lon_tg = read_tide_gauge(path.join(root, filename))

        # Compute correlation
        rcoeff = np.ma.corrcoef(sla_alti, sla_tg)[0, 1]

        if rcoeff > coef_corr_criterion:

            if np.min(np.diff(time_tg)) != np.max(np.diff(time_tg)):
                print "np.diff time"
            print(" ")
            print('Analyzing file %s' % filename)
            # make coherent timeseries between TG and alti (remove missing date)
            try:
                sla_tg = sla_tg.filled(-999.)
            except:
                print("No missing value SLA_TG in file %s" % filename)
            try:
                sla_alti = sla_alti.filled(-999.)
            except:
                print("No missing value SLA_alti in file %s" % filename)

            sla_alti = np.ma.masked_where(((sla_tg == -999.) | (sla_alti == -999.)), sla_alti)
            sla_tg = np.ma.masked_where(((sla_tg == -999.) | (sla_alti == -999.)), sla_tg)
            time_tg = np.ma.masked_where((sla_tg == -999) | (sla_alti == -999), time_tg)

            # plt.plot(timep, SLA_alti, color='r', label='MSLA')
            # plt.plot(timep, SLA_TG, color='b', label='TG')
            # plt.legend()
            # plt.show()

            sla_alti = sla_alti.compressed()
            sla_tg = sla_tg.compressed()
            time_tg = time_tg.compressed()

            # Prepare segment
            print("start segment computation", str(datetime.datetime.now()))

            computed_sla_segment, computed_msla_segment = \
                compute_segment_tide_gauge(sla_tg, time_tg, lenght_scale, delta_t, sla_alti)

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

            effective_resolution_out.append(output_effective_resolution)
            useful_resolution_out.append(output_useful_resolution)
            lat_out.append(lat_tg)
            lon_out.append(lon_tg)
            coherence_out.append(output_mean_coherence)
            power_spectrum_TG_out.append(output_mean_psd_sla_ref)
            power_spectrum_SLA_at_TG.append(output_mean_psd_sla_study)
            spectrum_TG_out.append(output_mean_psd_sla_ref)
            spectrum_SLA_at_TG.append(output_mean_ps_sla_study)
            frequency_out.append(output_mean_frequency)

print("start writing", str(datetime.datetime.now()))
write_netcdf_TG(output_nc_file, frequency_out, effective_resolution_out, useful_resolution_out,
                    lat_out, lon_out,
                    spectrum_TG_out, spectrum_SLA_at_TG,
                    power_spectrum_TG_out, power_spectrum_SLA_at_TG,
                    coherence_out)

print("end writing", str(datetime.datetime.now()))
