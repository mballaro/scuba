from netCDF4 import Dataset, date2num
import numpy as np
import matplotlib.pylab as plt
import datetime
import glob
from sys import argv, exit
from os import path
import re
from mpl_toolkits.basemap import Basemap
import datetime
import math
from scipy.fftpack import fft
import scipy.signal
import scipy.interpolate
from math import sqrt, cos, sin, asin, radians
from yaml import load

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
input_file_reference = YAML['inputs']['input_file_reference']
mission = YAML['inputs']['mission']

input_map_directory = YAML['inputs']['input_map_directory']
map_file_pattern = YAML['inputs']['map_file_pattern']

study_lon_min = YAML['properties']['study_area']['llcrnrlon']
study_lon_max = YAML['properties']['study_area']['urcrnrlon']
study_lat_min = YAML['properties']['study_area']['llcrnrlat']
study_lat_max = YAML['properties']['study_area']['urcrnrlat']

flag_roll = YAML['properties']['study_area']['flag_roll']
flag_ewp = YAML['properties']['study_area']['flag_ewp']
flag_greenwich_start = YAML['properties']['study_area']['flag_greenwich_start']

start = datetime.datetime.strptime(str(YAML['properties']['time_window']['YYYYMMDD_min']), '%Y%m%d')
end = datetime.datetime.strptime(str(YAML['properties']['time_window']['YYYYMMDD_max']), '%Y%m%d')
study_time_min = int(date2num(start, units="days since 1950-01-01", calendar='standard'))
study_time_max = int(date2num(end, units="days since 1950-01-01", calendar='standard'))
time_ensemble = np.arange(study_time_min, study_time_max + 1, 1)

mission_management = YAML['properties']['mission_management']

flag_edit_coastal = YAML['properties']['flag_edit_coastal']
file_coastal_distance = YAML['properties']['file_coastal_distance']
coastal_criteria = YAML['properties']['coastal_criteria']

flag_edit_spatiotemporal_incoherence = YAML['properties']['flag_edit_spatiotemporal_incoherence']

flag_reference_only = YAML['properties']['spectral_parameters']['flag_reference_only']
lenght_scale = YAML['properties']['spectral_parameters']['lenght_scale']
delta_lat = YAML['properties']['spectral_parameters']['delta_lat']
delta_lon = YAML['properties']['spectral_parameters']['delta_lon']
equal_area = YAML['properties']['spectral_parameters']['equal_area']

grid_lat = np.arange(study_lat_min, study_lat_max, YAML['outputs']['output_lat_resolution'])
grid_lon = np.arange(study_lon_min, study_lon_max, YAML['outputs']['output_lon_resolution'])

nc_file = YAML['outputs']['output_filename']

slap, timep, lonp, latp = read_along_track(input_file_reference, study_time_min, study_time_max,
                                           study_lon_min, study_lon_max, study_lat_min, study_lat_max,
                                           flag_edit_spatiotemporal_incoherence,
                                           flag_edit_coastal, file_coastal_distance, coastal_criteria, flag_roll)

if flag_reference_only:

    # Prepare segment
    print("start segment computation", str(datetime.datetime.now()))

    computed_sla_segment, computed_lon_segment, computed_lat_segment, delta_x, npt = \
        compute_segment_alongtrack(slap, lonp, latp, timep, mission, mission_management, lenght_scale)

    print("end segment computation", str(datetime.datetime.now()))

    # compute spectrum on grid
    print("start gridding", str(datetime.datetime.now()))
    output_effective_lon, output_effective_lat, output_mean_frequency, \
    output_mean_PSD_sla, output_mean_PS_sla, output_nb_segment, \
    output_autocorrelation_ref, output_autocorrelation_ref_zero_crossing, output_autocorrelation_distance = \
        spectral_computation(grid_lon, grid_lat, delta_lon, delta_lat,
                         np.asarray(computed_sla_segment),
                         np.asarray(computed_lon_segment),
                         np.asarray(computed_lat_segment),
                         delta_x, npt, equal_area, flag_greenwich_start, None)

    print("end gridding", str(datetime.datetime.now()))

    # Write netCDF output
    print("start writing", str(datetime.datetime.now()))

    write_netcdf_output(nc_file,
                        grid_lon, grid_lat, output_effective_lon, output_effective_lat,
                        output_mean_frequency, output_nb_segment, 'km',
                        output_mean_PS_sla, output_mean_PSD_sla,
                        output_autocorrelation_distance,
                        output_autocorrelation_ref, output_autocorrelation_ref_zero_crossing,
                        output_mean_ps_sla_study=None, output_mean_psd_sla_study=None,
                        output_mean_ps_diff_sla_ref_sla_study=None, output_mean_psd_diff_sla_ref_sla_study=None,
                        output_mean_coherence=None, output_effective_resolution=None, output_useful_resolution=None)

    print("end writing", str(datetime.datetime.now()))

else:

    # Interpolate map of SLA onto along-track
    print("start MSLA interpolation", str(datetime.datetime.now()))

    MSLA_interpolated = interpolate_msla_on_alongtrack(timep, latp, lonp,
                                                       input_map_directory, map_file_pattern, time_ensemble, flag_roll)

    print("end MSLA interpolation", str(datetime.datetime.now()))

    # Remove bad values that appear on MSLA after interpolation
    slap = np.ma.compressed(np.ma.masked_where(np.abs(MSLA_interpolated) > 10., slap))
    lonp = np.ma.compressed(np.ma.masked_where(np.abs(MSLA_interpolated) > 10., lonp))
    latp = np.ma.compressed(np.ma.masked_where(np.abs(MSLA_interpolated) > 10., latp))
    timep = np.ma.compressed(np.ma.masked_where(np.abs(MSLA_interpolated) > 10., timep))
    MSLA_interpolated = np.ma.compressed(np.ma.masked_where(np.abs(MSLA_interpolated) > 10., MSLA_interpolated))

    # plt.plot(timep, slap, color='k', label="Independent along-track", lw=2)
    # plt.plot(timep, MSLA_interpolated, color='r', label="MSLA interpolated", lw=2)
    # plt.ylabel('SLA (m)')
    # plt.xlabel('Timeline (Julian days since 1950-01-01)')
    # plt.legend(loc='best')
    # plt.show()

    # Prepare segment
    print("start segment computation", str(datetime.datetime.now()))

    computed_sla_segment, computed_lon_segment, computed_lat_segment, delta_x, npt, computed_msla_segment, \
    computed_crosscorrelation_segment = \
        compute_segment_alongtrack(slap, lonp, latp, timep, mission, mission_management,
                                   lenght_scale, MSLA_interpolated)

    print("end segment computation", str(datetime.datetime.now()))

    # compute spectrum on grid
    print("start gridding", str(datetime.datetime.now()))
    output_effective_lon, output_effective_lat, output_mean_frequency, \
    output_mean_PSD_sla, output_mean_PS_sla, output_nb_segment, \
    output_autocorrelation_ref, output_autocorrelation_ref_zero_crossing, output_autocorrelation_distance, \
    output_mean_ps_sla_study, output_mean_ps_diff_sla_study_sla_ref, \
    output_mean_psd_sla_study, output_mean_psd_diff_sla_study_sla_ref, \
    output_mean_coherence, output_effective_resolution, output_useful_resolution, \
    output_autocorrelation_study, output_autocorrelation_study_zero_crossing, output_cross_correlation = \
        spectral_computation(grid_lon, grid_lat, delta_lon, delta_lat,
                             np.asarray(computed_sla_segment),
                             np.asarray(computed_lon_segment),
                             np.asarray(computed_lat_segment),
                             delta_x, npt, equal_area, flag_greenwich_start,
                             sla_study_segments=np.asarray(computed_msla_segment),
                             cross_correlation_segments=np.asarray(computed_crosscorrelation_segment))

    print("end gridding", str(datetime.datetime.now()))

    # Write netCDF output
    print("start writing", str(datetime.datetime.now()))

    write_netcdf_output(nc_file,
                        grid_lon, grid_lat, output_effective_lon, output_effective_lat,
                        output_mean_frequency, output_nb_segment, 'km',
                        output_mean_PS_sla, output_mean_PSD_sla,
                        output_autocorrelation_distance,
                        output_autocorrelation_ref, output_autocorrelation_ref_zero_crossing,
                        output_mean_ps_sla_study=output_mean_ps_sla_study,
                        output_mean_psd_sla_study=output_mean_psd_sla_study,
                        output_autocorrelation_study=output_autocorrelation_study,
                        output_autocorrelation_study_zero_crossing=output_autocorrelation_study_zero_crossing,
                        output_mean_ps_diff_sla_ref_sla_study=output_mean_ps_diff_sla_study_sla_ref,
                        output_mean_psd_diff_sla_ref_sla_study=output_mean_psd_diff_sla_study_sla_ref,
                        output_mean_coherence=output_mean_coherence,
                        output_effective_resolution=output_effective_resolution,
                        output_useful_resolution=output_useful_resolution,
                        output_cross_correlation=output_cross_correlation)

    print("end writing", str(datetime.datetime.now()))
