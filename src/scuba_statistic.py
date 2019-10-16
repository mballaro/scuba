
import numpy as np
import matplotlib.pylab as plt
import datetime
from sys import argv, exit
from os import path
from yaml import load, Loader
import logging
import argparse

from mod_io import *
from mod_constant import *
from mod_geo import *
from mod_segmentation import *
from mod_spectral import *
from mod_interpolation import *
from mod_stat import *
from mod_filtering import *



parser = argparse.ArgumentParser(description='Spectral computation tools for altimeter data')
parser.add_argument("config_file", help="configuration file for alongtrack analysis (.yaml format)")
parser.add_argument('-v', '--verbose', dest='verbose', default='error',
                    help='Verbosity : critical, error, warning, info, debug')
args = parser.parse_args()

config_file = args.config_file

FORMAT_LOG = "%(levelname)-10s %(asctime)s %(module)s.%(funcName)s : %(message)s"
logging.basicConfig(format=FORMAT_LOG, level=logging.ERROR, datefmt="%H:%M:%S")
logger = logging.getLogger()
logger.setLevel(getattr(logging, args.verbose.upper()))

YAML = load(open(str(config_file)), Loader=Loader)

flag_reference_only = YAML['properties']['spectral_parameters']['flag_reference_only']
mission = YAML['inputs']['mission']
mission_management = YAML['properties']['mission_management']

# Read reference input
logging.info("start reading alongtrack")
if YAML['inputs']['ref_field_type_CLS'] == 'residus':
    ssh_alongtrack, time_alongtrack, lon_alongtrack, lat_alongtrack = read_residus_cls(YAML)
elif YAML['inputs']['ref_field_type_CLS'] == 'table':
    ssh_alongtrack, time_alongtrack, lon_alongtrack, lat_alongtrack = read_table_cls(YAML)
else:
    ssh_alongtrack, time_alongtrack, lon_alongtrack, lat_alongtrack = read_along_track(YAML)

logging.info("end reading alongtrack")

# Interpolate map of SLA onto along-track
logging.info("start map interpolation")

ssh_map_interpolated = interpolate_msla_on_alongtrack(time_alongtrack, lat_alongtrack, lon_alongtrack, YAML)

logging.info("end map interpolation")

# Remove bad values that appear on MSLA after interpolation
ssh_alongtrack = np.ma.masked_where(np.abs(ssh_map_interpolated) > 3., ssh_alongtrack, copy=False)
lon_alongtrack = np.ma.masked_where(np.abs(ssh_map_interpolated) > 3., lon_alongtrack, copy=False)
lat_alongtrack = np.ma.masked_where(np.abs(ssh_map_interpolated) > 3., lat_alongtrack, copy=False)
time_alongtrack = np.ma.masked_where(np.abs(ssh_map_interpolated) > 3., time_alongtrack, copy=False)
ssh_map_interpolated = np.ma.masked_where(np.abs(ssh_map_interpolated) > 3., ssh_map_interpolated, copy=False)

lon_alongtrack = np.ma.compressed(np.ma.masked_where(np.abs(ssh_alongtrack) > 3., lon_alongtrack, copy=False))
lat_alongtrack = np.ma.compressed(np.ma.masked_where(np.abs(ssh_alongtrack) > 3., lat_alongtrack, copy=False))
time_alongtrack = np.ma.compressed(np.ma.masked_where(np.abs(ssh_alongtrack) > 3., time_alongtrack, copy=False))
ssh_map_interpolated = np.ma.compressed(np.ma.masked_where(np.abs(ssh_alongtrack) > 3., ssh_map_interpolated, copy=False))
ssh_alongtrack = np.ma.compressed(np.ma.masked_where(np.abs(ssh_alongtrack) > 3., ssh_alongtrack, copy=False))


if  YAML['properties']['filtering']['filter_type'].lower() is not 'none':

    # Get number of point to consider for resolution = lenghtscale in km
    delta_t = get_deltat(mission.upper(), mission_management)
    delta_x = get_velocity(mission.upper(), mission_management) * delta_t

    cutoff_freq = []
    for ls in YAML['properties']['filtering']['length_scale']:
        npt = int(0.5*ls / delta_x)
        cutoff_freq.append(1./npt)

    cutoff_freq = np.sort(np.asarray(cutoff_freq))

    if len(cutoff_freq) == 1: 
        ssh_alongtrack_filtered = apply_highpass_filter(ssh_alongtrack, time_alongtrack, cutoff_freq)
        ssh_map_interpolated_filtered = apply_highpass_filter(ssh_map_interpolated, time_alongtrack, cutoff_freq)
    elif len(cutoff_freq) == 2:
        ssh_alongtrack_filtered = apply_bandpass_filter(ssh_alongtrack, time_alongtrack, cutoff_freq)
        ssh_map_interpolated_filtered = apply_bandpass_filter(ssh_map_interpolated, time_alongtrack, cutoff_freq)
    else:
        print("unknown filter definition")
        sys.exit()

    verif = False
    if verif:
        ref_segment, lon_segment, lat_segment, delta_x, npt = compute_segment_alongtrack(ssh_alongtrack,
                                                                                         lon_alongtrack,
                                                                                         lat_alongtrack,
                                                                                         time_alongtrack,
                                                                                         YAML)

        # compute global power spectrum density reference field
        global_wavenumber, global_psd_ref = scipy.signal.welch(np.asarray(ref_segment).flatten(),
                                                               fs=1.0 / delta_x,
                                                               nperseg=npt,
                                                               scaling='density',
                                                               noverlap=0)

        ref_segment, lon_segment, lat_segment, delta_x, npt = compute_segment_alongtrack(ssh_alongtrack_filtered,
                                                                                         lon_alongtrack,
                                                                                         lat_alongtrack,
                                                                                         time_alongtrack,
                                                                                         YAML)

        # compute global power spectrum density reference field
        global_wavenumber_filtered, global_psd_ref_filtered = scipy.signal.welch(np.asarray(ref_segment).flatten(),
                                                                 fs=1.0 / delta_x,
                                                                 nperseg=npt,
                                                                 scaling='density',
                                                                 noverlap=0)

        plt.plot(1./global_wavenumber_filtered, global_psd_ref_filtered, label='PSD filtered', color='r')
        plt.plot(1./global_wavenumber, global_psd_ref, label='PSD raw', color='b')
        for ls in YAML['properties']['filtering']['length_scale']:
            plt.axvline(x=ls, color='k')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

        ssh_alongtrack = ssh_alongtrack_filtered
        ssh_map_interpolated = ssh_map_interpolated_filtered

debug = False
if debug:
    plt.plot(time_alongtrack, ssh_alongtrack, color='k', label="Independent along-track", lw=2)
    plt.plot(time_alongtrack, ssh_map_interpolated, color='r', label="MSLA interpolated", lw=2)
    plt.ylabel('SLA (m)')
    plt.ylim(-2, 2)
    plt.xlabel('Timeline (Julian days since 1950-01-01)')
    plt.legend(loc='best')
    plt.show()


nobs, min, max, mean, variance, skewness, kurtosis, rmse, mae, correlation, pvalue , variance_ref, variance_study, mean_ref, mean_study = \
    statistic_computation(YAML, ssh_alongtrack, ssh_map_interpolated, lon_alongtrack, lat_alongtrack)

# # Write netCDF output
logging.info("start writing")
write_netcdf_stat_output(YAML, nobs, min, max, mean, variance, skewness, kurtosis, rmse, mae, correlation, pvalue,
                         variance_ref, variance_study, mean_ref, mean_study)
logging.info("end writing")
