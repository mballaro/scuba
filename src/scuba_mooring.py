from netCDF4 import Dataset, date2num
import numpy as np
import matplotlib.pylab as plt
import datetime
import glob
from sys import argv, exit
from os import path, walk
import re
import datetime
import math
from scipy.fftpack import fft
import scipy.signal
import scipy.interpolate
from math import sqrt, cos, sin, asin, radians
from yaml import load, Loader
import fnmatch
import logging
import argparse

from mod_io import *
from mod_constant import *
from mod_geo import *
from mod_segmentation import *
from mod_spectral import *
from mod_interpolation import *

parser = argparse.ArgumentParser(description='Spectral computation tools for altimeter data')
parser.add_argument("config_file", help="configuration file for mooring analysis (.yaml format)")
parser.add_argument('-v', '--verbose', dest='verbose', default='error',
                    help='Verbosity : critical, error, warning, info, debug')
args = parser.parse_args()

config_file = args.config_file

FORMAT_LOG = "%(levelname)-10s %(asctime)s %(module)s.%(funcName)s : %(message)s"
logging.basicConfig(format=FORMAT_LOG, level=logging.ERROR, datefmt="%H:%M:%S")
logger = logging.getLogger()
logger.setLevel(getattr(logging, args.verbose.upper()))

YAML = load(open(str(config_file)), Loader=Loader)

debug = False

input_file_directory = YAML['inputs']['input_file_directory']
input_file_pattern = YAML['inputs']['input_file_pattern']
coef_corr_criterion = YAML['properties']['corr_coef']
lenght_scale = YAML['properties']['spectral_parameters']['lenght_scale']
segment_overlapping = YAML['properties']['spectral_parameters']['segment_overlapping']
delta_t = YAML['properties']['spectral_parameters']['delta_t']
output_nc_file = YAML['outputs']['output_filename']

lat_out = []
lon_out = []
coherence_out = []
psd_mooring_out = []
psd_study_out = []
wavenumber_out = []
psd_diff_mooring_study_out = []
cross_spectrum_out = []

for root, dirnames, filenames in walk(input_file_directory):

    number_of_timeseries = 1

    for filename in fnmatch.filter(filenames, input_file_pattern):

        # Read SLA tide-gauge and map of SLA
        ssh_mooring, ssh_alti, time_mooring, lat_mooring, lon_mooring = read_mooring(path.join(root, filename))

        # Compute correlation
        rcoeff = np.ma.corrcoef(ssh_alti, ssh_mooring)[0, 1]

        if rcoeff > coef_corr_criterion:

            if np.min(np.diff(time_mooring)) != np.max(np.diff(time_mooring)):
                print("np.diff time")

            logging.info('Analyzing file %s' % filename)
            # make coherent timeseries between mooring and alti (remove missing date)
            ssh_mooring = ssh_mooring.filled(-999.)
            ssh_alti = ssh_alti.filled(-999.)

            # Some cleaning
            ssh_alti = np.ma.masked_where(((ssh_mooring == -999.) | (ssh_alti == -999.)), ssh_alti)
            ssh_mooring = np.ma.masked_where(((ssh_mooring == -999.) | (ssh_alti == -999.)), ssh_mooring)
            time_mooring = np.ma.masked_where((ssh_mooring == -999) | (ssh_alti == -999), time_mooring)

            if debug:
                plt.plot(time_mooring, ssh_alti, color='r', label='SSH maps')
                plt.plot(time_mooring, ssh_mooring, color='b', label='SSH mooring')
                plt.legend()
                plt.show()

            ssh_alti = ssh_alti.compressed()
            ssh_mooring = ssh_mooring.compressed()
            time_mooring = time_mooring.compressed()

            # Prepare segment
            logging.info("start segment computation")

            mooring_segment, study_segment = compute_segment_tide_gauge_tao(ssh_mooring,
                                                                            time_mooring,
                                                                            lenght_scale,
                                                                            delta_t,
                                                                            segment_overlapping,
                                                                            ssh_alti)

            logging.info("end segment computation")

            # compute spectrum on grid
            logging.info("start spectral computation")

            wavenumber, psd_mooring, psd_study, psd_diff_mooring_study, coherence, cross_spectrum = \
                spectral_computation_tide_gauge_tao(np.asarray(mooring_segment).flatten(),
                                                    delta_t,
                                                    lenght_scale,
                                                    study_segments=np.asarray(study_segment).flatten())

            logging.info("end spectral computation")

            if wavenumber.size > 0:
                lat_out.append(lat_mooring)
                lon_out.append(lon_mooring)
                coherence_out.append(coherence)
                psd_mooring_out.append(psd_mooring)
                psd_study_out.append(psd_study)
                wavenumber_out.append(wavenumber)
                psd_diff_mooring_study_out.append(psd_diff_mooring_study)
                cross_spectrum_out.append(cross_spectrum)

logging.info("start writing")
write_netcdf_tide_tao(output_nc_file,
                      wavenumber_out,
                      lat_out,
                      lon_out,
                      psd_mooring_out,
                      psd_study_out,
                      psd_diff_mooring_study_out,
                      coherence_out,
                      cross_spectrum_out)

logging.info("end writing")
