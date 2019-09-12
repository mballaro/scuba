
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

# Read reference input
logging.info("start reading alongtrack")
if YAML['inputs']['ref_field_type_CLS'] == 'residus':
    ssh_alongtrack, time_alongtrack, lon_alongtrack, lat_alongtrack = read_residus_cls(YAML)
elif YAML['inputs']['ref_field_type_CLS'] == 'table':
    ssh_alongtrack, time_alongtrack, lon_alongtrack, lat_alongtrack = read_table_cls(YAML)
else:
    ssh_alongtrack, time_alongtrack, lon_alongtrack, lat_alongtrack = read_along_track(YAML)

logging.info("end reading alongtrack")

if flag_reference_only:

    # compute segment
    logging.info("start segment computation")

    ref_segment, lon_segment, lat_segment, delta_x, npt = compute_segment_alongtrack(ssh_alongtrack,
                                                                                     lon_alongtrack,
                                                                                     lat_alongtrack,
                                                                                     time_alongtrack,
                                                                                     YAML)
    # write segment
    write_segment(YAML, lat_segment, lon_segment, ref_segment, delta_x, 'km', npt)

    logging.info("end segment computation")

    # compute global power spectrum density reference field
    global_wavenumber, global_psd_ref = scipy.signal.welch(np.asarray(ref_segment).flatten(),
                                                           fs=1.0 / delta_x,
                                                           nperseg=npt,
                                                           scaling='density',
                                                           noverlap=0)

    # compute spectrum on grid
    logging.info("start gridding")

    wavenumber, psd_ref, nb_segment = spectral_computation(YAML,
                                                           np.asarray(ref_segment),
                                                           np.asarray(lon_segment),
                                                           np.asarray(lat_segment),
                                                           delta_x,
                                                           npt)

    logging.info("end gridding")

    # write netCDF output
    logging.info("start writing")

    write_netcdf_output(YAML, wavenumber, nb_segment, 'km', psd_ref, global_psd_ref)

    logging.info("end writing")

else:

    # Interpolate map of SLA onto along-track
    logging.info("start map interpolation")

    ssh_map_interpolated = interpolate_msla_on_alongtrack(time_alongtrack, lat_alongtrack, lon_alongtrack, YAML)

    logging.info("end map interpolation")

    # Remove bad values that appear on MSLA after interpolation
    ssh_alongtrack = np.ma.masked_where(np.abs(ssh_map_interpolated) > 10., ssh_alongtrack)
    lon_alongtrack = np.ma.masked_where(np.abs(ssh_map_interpolated) > 10., lon_alongtrack)
    lat_alongtrack = np.ma.masked_where(np.abs(ssh_map_interpolated) > 10., lat_alongtrack)
    time_alongtrack = np.ma.masked_where(np.abs(ssh_map_interpolated) > 10., time_alongtrack)
    ssh_map_interpolated = np.ma.masked_where(np.abs(ssh_map_interpolated) > 10., ssh_map_interpolated)

    lon_alongtrack = np.ma.compressed(np.ma.masked_where(np.abs(ssh_alongtrack) > 10., lon_alongtrack))
    lat_alongtrack = np.ma.compressed(np.ma.masked_where(np.abs(ssh_alongtrack) > 10., lat_alongtrack))
    time_alongtrack = np.ma.compressed(np.ma.masked_where(np.abs(ssh_alongtrack) > 10., time_alongtrack))
    ssh_map_interpolated = np.ma.compressed(np.ma.masked_where(np.abs(ssh_alongtrack) > 10., ssh_map_interpolated))
    ssh_alongtrack = np.ma.compressed(np.ma.masked_where(np.abs(ssh_alongtrack) > 10., ssh_alongtrack))
    
    debug = False
    if debug:
        plt.scatter(time_alongtrack, ssh_alongtrack, color='k', label="Independent along-track", lw=2)
        plt.plot(time_alongtrack, ssh_map_interpolated, color='r', label="MSLA interpolated", lw=2)
        # plt.plot(time_alongtrack_c, ssh_map_interpolated_c, color='c', label="MSLA cleaned interpolated", lw=2)
        plt.ylabel('SLA (m)')
        plt.ylim(-2, 2)
        plt.xlabel('Timeline (Julian days since 1950-01-01)')
        plt.legend(loc='best')
        plt.show()

    # Prepare segment
    logging.info("start segment computation")

    ref_segment, lon_segment, lat_segment, delta_x, npt, study_segment = \
        compute_segment_alongtrack(ssh_alongtrack,
                                   lon_alongtrack,
                                   lat_alongtrack,
                                   time_alongtrack,
                                   YAML,
                                   ssh_map_interpolated)

    write_segment(YAML, lat_segment, lon_segment, ref_segment, delta_x, 'km', npt, segment_study=study_segment)

    logging.info("end segment computation")

    # Power spectrum density reference field
    global_wavenumber, global_psd_ref = scipy.signal.welch(np.asarray(ref_segment).flatten(),
                                                           fs=1.0 / delta_x,
                                                           nperseg=npt,
                                                           scaling='density',
                                                           noverlap=0)

    # Power spectrum density study field
    _, global_psd_study = scipy.signal.welch(np.asarray(study_segment).flatten(),
                                             fs=1.0 / delta_x,
                                             nperseg=npt,
                                             scaling='density',
                                             noverlap=0)

    # compute spectrum on grid
    logging.info("start gridding")
    wavenumber, psd_ref, nb_segment, psd_study, psd_diff_study_ref, coherence, cross_spectrum = \
        spectral_computation(YAML,
                             np.asarray(ref_segment),
                             np.asarray(lon_segment),
                             np.asarray(lat_segment),
                             delta_x,
                             npt,
                             np.asarray(study_segment))

    logging.info("end gridding")

    # Write netCDF output
    logging.info("start writing")

    write_netcdf_output(YAML,
                        wavenumber,
                        nb_segment,
                        'km',
                        psd_ref,
                        global_psd_ref,
                        global_psd_study=global_psd_study,
                        psd_study=psd_study,
                        psd_diff_ref_study=psd_diff_study_ref,
                        coherence=coherence,
                        cross_spectrum=cross_spectrum)

    logging.info("end writing")
