from netCDF4 import Dataset, date2num, num2date
import numpy as np
import matplotlib.pylab as plt
import datetime
from sys import argv, exit
from os import path, walk
import re
import datetime
from yaml import load
import fnmatch
# from pathlib import Path
import logging
import argparse

from mod_io import *
from mod_constant import *
from mod_geo import *
from mod_segmentation import *
from mod_spectral import *
from mod_interpolation import *


parser = argparse.ArgumentParser(description='Spectral computation tools for altimeter data')
parser.add_argument("config_file", help="configuration file for tao mooring analysis (.yaml format)")
parser.add_argument('-v', '--verbose', dest='verbose', default='error',
                    help='Verbosity : critical, error, warning, info, debug')
args = parser.parse_args()

config_file = args.config_file

FORMAT_LOG = "%(levelname)-10s %(asctime)s %(module)s.%(funcName)s : %(message)s"
logging.basicConfig(format=FORMAT_LOG, level=logging.ERROR, datefmt="%H:%M:%S")
logger = logging.getLogger()
logger.setLevel(getattr(logging, args.verbose.upper()))

YAML = load(open(str(config_file)), Loader=Loader)

# Load analysis information from YAML file
input_tao_file_directory = YAML['inputs']['input_tao_file_directory']
ref_field_scale_factor = YAML['inputs']['ref_field_scale_factor']
input_map_file = YAML['inputs']['input_map_file']
study_field_scale_factor = YAML['inputs']['study_field_scale_factor']

coef_corr_criterion = YAML['properties']['corr_coef']
lenght_scale = YAML['properties']['spectral_parameters']['lenght_scale']
delta_t = YAML['properties']['spectral_parameters']['delta_t']
segment_overlapping = YAML['properties']['spectral_parameters']['segment_overlapping']
output_nc_file = YAML['outputs']['output_filename']

debug = False

lat_out = []
lon_out = []
coherence_out = []
spectrum_TAO_out = []
spectrum_SSH_at_TAO = []
power_spectrum_TAO_out = []
power_spectrum_SSH_at_TAO = []
frequency_out = []

# Read map lon, lat, time vector
#nc = Dataset(input_map_file, 'r')
#lon_map = nc.variables['longitude'][:]
#lat_map = nc.variables['latitude'][:]
#time_map = nc.variables['time'][:]

logging.info("start reading ssh map file")
ncfile = YAML['inputs']['input_map_file']
list_of_file = []
for root, dirnames, filenames in walk(ncfile):
    for filename in fnmatch.filter(filenames, '*.nc'):
        list_of_file.append(path.join(root, filename))

if len(ncfile) > 0:
    ncfile = list_of_file
    if "*" in ncfile or isinstance(ncfile, list):
        ds = xr.open_mfdataset(ncfile, decode_times=False)
    else:
        ds = xr.open_dataset(ncfile, decode_times=False)

    lon_map = ds['longitude'].values
    lat_map = ds['latitude'].values
    time_map = ds['time'].values

else:
    ds = None
    lon_map = []
    lat_map = []
    time_map = []
    logging.info('Number of file = 0')
    exit(0)

logging.info("end reading ssh map file")

for root, dirnames, filenames in walk(input_tao_file_directory):

    number_of_timeseries = 1

    for filename in fnmatch.filter(filenames, '*.cdf*'):

        # Read SSH TAO
        logging.info("start reading TAO Mooring file : %s" % filename)
        print(filename)
        if path.exists(filename[:-7]+'.nc'):
            print('File %s is found' %(filename[:-7]+'.nc'))

        else:
            ssh_tao, time_tao, lat_tao, lon_tao = read_tao(path.join(root, filename))
            ssh_tao = ssh_tao * ref_field_scale_factor

            logging.info("end reading TAO Mooring file")

            # Cleaning
            ssh_tao = np.ma.masked_invalid(np.ma.masked_where(np.abs(ssh_tao) > 1.E10, ssh_tao))
            # convert date to julian day
            date_time_tao = num2date(time_tao, units="days since 1950-01-01", calendar='julian')
            date_JD_tao = date2num(date_time_tao, units="days since 1950-01-01", calendar='julian')
            # mask date > 20170101
            index_max_time_tao = find_nearest_index(time_tao, 24472)
            date_JD_tao = date_JD_tao[:index_max_time_tao]
            ssh_tao = ssh_tao[:index_max_time_tao]

            print(time_tao)
            if len(date_JD_tao) > 0:

                # select closest maps point to the TAO
                index_lon = find_nearest_index(lon_map, lon_tao)
                index_lat = find_nearest_index(lat_map, lat_tao)
                index_min_time = find_nearest_index(time_map, np.min(date_JD_tao))
                index_max_time = find_nearest_index(time_map, np.max(date_JD_tao) + 1)

                logging.info("start reading SSH map timeserie")
                # ssh_map = nc.variables['adt'][index_min_time:index_max_time, index_lat, index_lon]
                ds_tmp = ds.sel(longitude=lon_tao, latitude=lat_tao, method='nearest')
                ds_tmp = ds_tmp.sel(time=slice(np.min(date_JD_tao), np.max(date_JD_tao)))

                ssh_map = np.squeeze(ds_tmp[YAML['inputs']['input_map_varname']].values)
                # ssh_map = ds[YAML['inputs']['input_map_varname']].values[index_min_time:index_max_time, index_lat, index_lon]
                ssh_map = ssh_map * study_field_scale_factor
                logging.info("end reading SSH map timeserie")

                # save data
                nc_out = Dataset(path.basename(filename)[:-7]+'.nc', 'w', format='NETCDF4')
                nc_out.createDimension('time', date_JD_tao.size)
                nc_out.createDimension('x', 1)

                lon_out = nc_out.createVariable('lon', 'f8', 'x')
                lon_out[:] = lon_tao
                lon_out.longname = 'longitude sensor'

                lat_out = nc_out.createVariable('lat', 'f8', 'x')
                lat_out[:] = lat_tao
                lat_out.longname = 'latitude sensor'

                time_out = nc_out.createVariable('time', 'f8', 'time')
                time_out[:] = date_JD_tao
                time_out.longname = 'Time'
                time_out.units = 'days since 1950-01-01'

                ssh_sensor_out = nc_out.createVariable('ssh_sensor', 'f8', 'time')
                ssh_sensor_out[:] = ssh_tao
                ssh_sensor_out.units = 'm'

                ssh_alti_out = nc_out.createVariable('ssh_alti', 'f8', 'time')
                ssh_alti_out[:] = ssh_map
                ssh_alti_out.units = 'm'

                nc_out.close()

