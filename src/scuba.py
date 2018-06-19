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

from yaml import load, dump

if len(argv) != 2:
    print(' \n')
    print('USAGE   : %s <file.YAML> \n' % path.basename(argv[0]))
    print('Purpose : Spectral and spectral coherence computation tools for altimeter data \n')
    exit(0)

# Load analysis information from YAML file
YAML = load(open(str(argv[1])))
input_file_independent_alongtrack = YAML['inputs']['input_file_independent_alongtrack']
mission = YAML['inputs']['mission']

input_map_directory = YAML['inputs']['input_map_directory']
map_file_pattern = YAML['inputs']['map_file_pattern']

study_lon_min = YAML['properties']['study_area']['llcrnrlon']
study_lon_max = YAML['properties']['study_area']['urcrnrlon']
study_lat_min = YAML['properties']['study_area']['llcrnrlat']
study_lat_max = YAML['properties']['study_area']['urcrnrlat']

flag_roll = YAML['properties']['study_area']['flag_roll']

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

flag_alongtrack_only = YAML['properties']['spectral_parameters']['flag_alongtrack_only']
lenght_scale = YAML['properties']['spectral_parameters']['lenght_scale']
delta_lat = YAML['properties']['spectral_parameters']['delta_lat']
delta_lon = YAML['properties']['spectral_parameters']['delta_lon']
equal_area = YAML['properties']['spectral_parameters']['equal_area']

grid_lat = np.arange(study_lat_min, study_lat_max, YAML['outputs']['output_lat_resolution'])
grid_lon = np.arange(study_lon_min, study_lon_max, YAML['outputs']['output_lon_resolution'])

nc_file = YAML['outputs']['output_filename']

# print input_file_independent_alongtrack
# print mission
# print input_map_directory
# print map_file_pattern
# print study_lon_min
# print study_lon_max
# print study_lat_min
# print study_lat_max
# print start
# print end
# print study_time_min
# print study_time_max
# print time_ensemble
# print mission_management
# print flag_edit_coastal
# print file_coastal_distance
# print coastal_criteria
# print lenght_scale
# print delta_lat
# print delta_lon
# print equal_area
# print grid_lat
# print grid_lon
# print nc_file

# Constants
earth_radius = 6371.0  # in km
degrees_to_radians = math.pi/180.0
radians_to_degrees = 180.0/math.pi
buffer_zone = 10


def find_nearest_index(array, value):
    """
    Given an array and a value, return nearest value index in array
    :param array:
    :param value:
    :return:
    """
    index = (np.abs(array - value)).argmin()
    return index


def change_in_latitude(kilometers):
    """
    Given a distance north, return the change in latitude.
    :param kilometers:
    :return:
    """
    return (kilometers/earth_radius)*radians_to_degrees


def change_in_longitude(latitude, kilometers):
    """
    Given a latitude and a distance west, return the change in longitude.
    :param latitude:
    :param kilometers:
    :return:
    """
    # Find the radius of a circle around the earth at given latitude.
    r = earth_radius*math.cos(latitude*degrees_to_radians)
    return (kilometers/r)*radians_to_degrees


def get_vitesse(cmission):
    """
    Get velocity of a mission from MissionManagement.yaml
    """
    velocity = None
    YAML2 = load(open(str(mission_management)))
    velocity = YAML2[cmission]['VELOCITY']
    if velocity is not None:
        return velocity
    else:
        raise ValueError("velocity not found for mission %s in %s" % (cmission, mission_management))


def get_deltat(cmission):
    """
    Get deltaT of a mission from file MissionManagement.yaml
    """
    deltat = None
    YAML2 = load(open(str(mission_management)))
    deltat = YAML2[cmission]['DELTA_T']
    if deltat is not None:
        return deltat
    else:
        raise ValueError("deltat not found for mission %s in %s" % (cmission, mission_management))


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    :param lon1:
    :param lat1:
    :param lon2:
    :param lat2:
    :return:
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    ca1 = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    cc1 = 2 * asin(sqrt(ca1))
    return cc1 * earth_radius


def edit_bad_velocity(netcdf_file):
    """
    Edit aberant point from velocity criteria (cf. BUG REPORT OCEANNEXT)
    :param netcdf_file:
    :return:
    """

    ncfile = Dataset(netcdf_file, "r")
    time = ncfile.variables['time'][:]
    lon = ncfile.variables['longitude'][:]
    lat = ncfile.variables['latitude'][:]
    sla = ncfile.variables['SLA'][:]
    ncfile.close()

    # get mission velocity
    mission_velocity = get_vitesse(mission.upper())
    delta_velocity = 1.  # tolerance criteria km/s
    min_velocity_criteria = mission_velocity - delta_velocity
    max_velocity_criteria = mission_velocity + delta_velocity

    deltax = np.zeros(lon.size)
    deltat = np.zeros(lon.size)
    for ii in range(0, lon.size-1):
        deltax[ii] = haversine(lon[ii], lat[ii], lon[ii+1], lat[ii+1])
        deltat[ii] = (time[ii+1] - time[ii])*24.*3600.

    velocity = deltax/deltat
    lon_ok = np.ma.masked_where((velocity > max_velocity_criteria) & (velocity < min_velocity_criteria), lon[:])
    lat_ok = np.ma.masked_where((velocity > max_velocity_criteria) & (velocity < min_velocity_criteria), lat[:])
    time_ok = np.ma.masked_where((velocity > max_velocity_criteria) & (velocity < min_velocity_criteria), time[:])
    sla_ok = np.ma.masked_where((velocity > max_velocity_criteria) & (velocity < min_velocity_criteria), sla[:])

    return time_ok, lon_ok, lat_ok, sla_ok


def edit_coastal_data(sla, lon_along_track, lat_along_track, time_along_track):
    """
    Edit coastal along-track dta based on the coastal distance file
    :param sla:
    :param lon_along_track:
    :param lat_along_track:
    :param time_along_track:
    :return:
    """
    # Read coastal distance file
    nc = Dataset(file_coastal_distance, 'r')
    distance = nc.variables['distance'][:, :]
    lon_distance = nc.variables['lon'][:]
    lat_distance = nc.variables['lat'][:]
    coastal_flag = np.zeros((len(sla)))
    nc.close()

    # For Med Sea
    if flag_roll:
        lon_distance = np.where(lon_distance >= 180, lon_distance - 360, lon_distance)
    
    # Prepare mask for coastal region
    for time_index in range(sla.size):
        # nearest lon_distance from lon_along_track
        ii_nearest = find_nearest_index(lon_distance, lon_along_track[time_index])
        # nearest lat_distance from lat_along_track
        jj_nearest = find_nearest_index(lat_distance, lat_along_track[time_index])

        if distance[jj_nearest, ii_nearest] > coastal_criteria:
            coastal_flag[time_index] = 1.0

    edited_sla = np.ma.compressed(np.ma.masked_where(coastal_flag == 0., sla))
    edited_lon_along_track = np.ma.compressed(np.ma.masked_where(coastal_flag == 0., lon_along_track))
    edited_lat_along_track = np.ma.compressed(np.ma.masked_where(coastal_flag == 0., lat_along_track))
    edited_time_along_track = np.ma.compressed(np.ma.masked_where(coastal_flag == 0., time_along_track))

    return edited_sla, edited_time_along_track, edited_lon_along_track, edited_lat_along_track


def read_cls_format(fcid):
    """

    :param fcid:
    :return:
    """
    sla_map = np.array(fcid.variables['Grid_0001'][:, :]).transpose() / 100.  # in meters
    lat_map = np.array(fcid.variables['NbLatitudes'][:])
    lon_map = np.array(fcid.variables['NbLongitudes'][:])

    return sla_map, lat_map, lon_map


def read_cmems_format(fcid):
    """

    :param fcid:
    :return:
    """
    sla_map = np.array(fcid.variables['sla'][0, :, :])  # in meters
    lat_map = np.array(fcid.variables['latitude'][:])
    lon_map = np.array(fcid.variables['longitude'][:])

    return sla_map, lat_map, lon_map


def interpolate_msla_on_alongtrack(time_along_track, lat_along_track, lon_along_track):
    """
    Interpolate map of SLA onto a along-track dataset
    :param time_along_track:
    :param lat_along_track:
    :param lon_along_track:
    :return:
    """
    for tt in range(len(time_ensemble)):
        tref = time_ensemble[tt]
        date = datetime.datetime(1950, 1, 1) + datetime.timedelta(tref)
        char = date.strftime('%Y%m%d')
        file_type = input_map_directory + '/' + map_file_pattern + char + '_*.nc'
        file = glob.glob(file_type)[0]
        fcid = Dataset(file, 'r')

        # Achtung: CLS internal netCDF format, must be adapted by user
        # lat_map = np.array(fcid.variables['NbLatitudes'][:])
        # sla_map = np.array(fcid.variables['Grid_0001'][:, :]).transpose() / 100.  # in meters
        # lon_map = np.array(fcid.variables['NbLongitudes'][:])

        sla_map, lat_map, lon_map = read_cmems_format(fcid)

        # For Med Sea
        if flag_roll:
            lon_map = np.where(lon_map >= 180, lon_map - 360, lon_map)



        # Mask invalid data if necessary
        sla_map = np.ma.masked_invalid(sla_map)

        if tt == 0:
            saved_sla_map = np.zeros((len(time_ensemble), np.shape(sla_map)[0], np.shape(sla_map)[1]))
        saved_sla_map[tt, :, :] = sla_map

    finterp_map2alongtrack = scipy.interpolate.RegularGridInterpolator(
            [time_ensemble, lat_map, lon_map],
            saved_sla_map, bounds_error=False, fill_value=None)

    msla_interpolated = finterp_map2alongtrack(np.transpose([time_along_track, lat_along_track, lon_along_track]))

    return msla_interpolated


def fill_small_gap(sla, lon_along_track, lat_along_track, time_along_track):
    """
    Fill gab less than 3 points
    :param sla:
    :param lon_along_track:
    :param lat_along_track:
    :param time_along_track:
    :return:
    """
    # Get number of point to consider for resolution = lenghtscale in km
    delta_t = get_deltat(mission.upper())

    # Convert delta_t from second to Julian Day
    delta_t_jd = delta_t / (3600 * 24)

    indi_for_linear_interp = np.where((np.diff(time_along_track) > delta_t_jd)
                                      & (np.diff(time_along_track) < 4 * delta_t_jd))[0]

    # Initialize NEW DATA
    filled_sla = np.copy(sla)
    filled_lon_along_track = np.copy(lon_along_track)
    filled_lat_along_track = np.copy(lat_along_track)
    filled_time_along_track = np.copy(time_along_track)

    if indi_for_linear_interp.size > 0:
        for ii in range(indi_for_linear_interp.size):
            # select the two data points whre to interpolate
            t1 = time_along_track[indi_for_linear_interp[ii]]
            lon1 = lon_along_track[indi_for_linear_interp[ii]]
            lat1 = lat_along_track[indi_for_linear_interp[ii]]
            sla1 = sla[indi_for_linear_interp[ii]]

            t2 = time_along_track[indi_for_linear_interp[ii] + 1]
            lon2 = lon_along_track[indi_for_linear_interp[ii] + 1]
            lat2 = lat_along_track[indi_for_linear_interp[ii] + 1]
            sla2 = sla[indi_for_linear_interp[ii] + 1]

            flon_interp = scipy.interpolate.interp1d([t1, t2], [lon1, lon2])
            new_lon = flon_interp(np.linspace(t1, t2, int(round(abs(t2 - t1) / delta_t_jd)) + 1))

            flat_interp = scipy.interpolate.interp1d([t1, t2], [lat1, lat2])
            new_lat = flat_interp(np.linspace(t1, t2, int(round(abs(t2 - t1) / delta_t_jd)) + 1))

            fsla_interp = scipy.interpolate.interp1d([t1, t2], [sla1, sla2])
            new_sla = fsla_interp(np.linspace(t1, t2, int(round(abs(t2 - t1) / delta_t_jd)) + 1))

            new_time = np.linspace(t1, t2, int(round(abs(t2 - t1) / delta_t_jd)))

            filled_sla = np.insert(filled_sla, indi_for_linear_interp[ii] + 1,
                                   new_sla[1:new_sla.size - 1], axis=None)
            filled_lon_along_track = np.insert(filled_lon_along_track, indi_for_linear_interp[ii] + 1,
                                               new_lon[1:new_lon.size - 1], axis=None)
            filled_lat_along_track = np.insert(filled_lat_along_track, indi_for_linear_interp[ii] + 1,
                                               new_lat[1:new_lat.size - 1], axis=None)
            filled_time_along_track = np.insert(filled_time_along_track, indi_for_linear_interp[ii] + 1,
                                                new_time[1:new_time.size], axis=None)


def compute_segment(sla, msla, lon_along_track, lat_along_track, time_along_track):
    """

    :param sla:
    :param msla:
    :param lon_along_track:
    :param lat_along_track:
    :param time_along_track:
    :return:
    """

    list_lat_segment = []
    list_lon_segment = []
    list_sla_segment = []
    list_msla_segment = []

    # Get number of point to consider for resolution = lenghtscale in km
    delta_t = get_deltat(mission.upper())

    delta_x = get_vitesse(mission.upper()) * delta_t

    delta_t_jd = delta_t / (3600 * 24)
    npt = int(lenght_scale / delta_x)

    # cut track when diff time longer than 4*delta_t
    indi = np.where((np.diff(time_along_track) > 4 * delta_t_jd))[0]
    track_segment_lenght = np.insert(np.diff(indi), [0], indi[0])

    # Long track >= npt
    selected_track_segment = np.where(track_segment_lenght >= npt)[0]

    # nb_fig = 0

    if selected_track_segment.size > 0:

        nb_segment = 0
        nb_sub_segment = 0

        for track in selected_track_segment:

            nb_segment += 1
            # print "Processing segment %s / %s" %(str(nb_segment), str(selected_track_segment.size))

            if track-1 >= 0:
                index_start_selected_track = indi[track-1]
                index_end_selected_track = indi[track]
            else:
                index_start_selected_track = 0
                index_end_selected_track = indi[track]

            start_point = index_start_selected_track
            end_point = index_end_selected_track

            for sub_segment_point in range(start_point, end_point - npt, int(npt / 4)):

                nb_sub_segment += 1

                # Near Greenwhich case
                if ((lon_along_track[sub_segment_point + npt - 1] < 50.)
                    and (lon_along_track[sub_segment_point] > 320.)) \
                        or ((lon_along_track[sub_segment_point + npt - 1] > 320.)
                            and (lon_along_track[sub_segment_point] < 50.)):

                    tmp_lon = np.where(lon_along_track[sub_segment_point:sub_segment_point + npt] > 180,
                                       lon_along_track[sub_segment_point:sub_segment_point + npt] - 360,
                                       lon_along_track[sub_segment_point:sub_segment_point + npt])
                    mean_lon_sub_segment = np.median(tmp_lon)

                    if mean_lon_sub_segment < 0:
                        mean_lon_sub_segment = mean_lon_sub_segment + 360.
                else:

                    mean_lon_sub_segment = np.median(lon_along_track[sub_segment_point:sub_segment_point + npt])

                mean_lat_sub_segment = np.median(lat_along_track[sub_segment_point:sub_segment_point + npt])

                # plt.figure()
                # bmap = Basemap(
                #     projection='cyl', resolution='l', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360)
                # bmap.fillcontinents(color='grey', lake_color='white')
                # bmap.drawcoastlines()
                # bmap.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0], fontsize=15, dashes=[4, 4],
                #                    linewidth=0.25)
                # bmap.drawmeridians(np.arange(-180, 360, 30), labels=[0, 0, 0, 1], fontsize=15, dashes=[4, 4],
                #                    linewidth=0.25)
                # x, y = bmap(lon_along_track[start_point:end_point], lat_along_track[start_point:end_point])
                # bmap.scatter(x, y, s=0.1, marker="o", color='g')
                # x, y = bmap(lon_along_track[sub_segment_point:sub_segment_point + npt],
                #             lat_along_track[sub_segment_point:sub_segment_point + npt])
                # bmap.scatter(x, y, s=0.1, marker="o", color='r')
                # bmap.scatter(mean_lon_sub_segment, mean_lat_sub_segment, s=100, color='k', marker='x')
                # plt.title("Tracks and segments used in the computation")
                # # plt.show()
                # plt.savefig('fig_%04d.png' % nb_fig)
                # nb_fig += 1

                sla_segment = np.ma.masked_invalid(sla[sub_segment_point:sub_segment_point + npt])
                msla_segment = np.ma.masked_invalid(msla[sub_segment_point:sub_segment_point + npt])
                if np.ma.is_masked(msla_segment):
                    # print "MASK detected"
                    sla_segment = np.ma.compressed(np.ma.masked_where(np.ma.is_masked(msla_segment), sla_segment))
                    msla_segment = np.ma.compressed(msla_segment)

                # plt.plot(sla_segment, color='r', label="SLA along-track")
                # plt.plot(msla_segment, color='b', label='interpolated MSLA')
                # plt.legend(loc="best")
                # plt.show()

                if sla_segment.size > 0:
                    list_sla_segment.append(sla_segment)
                    list_msla_segment.append(msla_segment)
                    list_lon_segment.append(mean_lon_sub_segment)
                    list_lat_segment.append(mean_lat_sub_segment)

    return list_sla_segment, list_msla_segment, list_lon_segment, list_lat_segment


def compute_segment_alongtrack_only(sla, lon_along_track, lat_along_track, time_along_track):
    """

    :param sla:
    :param lon_along_track:
    :param lat_along_track:
    :param time_along_track:
    :return:
    """

    list_lat_segment = []
    list_lon_segment = []
    list_sla_segment = []

    # Get number of point to consider for resolution = lenghtscale in km
    delta_t = get_deltat(mission.upper())

    delta_x = get_vitesse(mission.upper()) * delta_t

    delta_t_jd = delta_t / (3600 * 24)
    npt = int(lenght_scale / delta_x)

    # cut track when diff time longer than 4*delta_t
    indi = np.where((np.diff(time_along_track) > 4 * delta_t_jd))[0]
    track_segment_lenght = np.insert(np.diff(indi), [0], indi[0])

    # Long track >= npt
    selected_track_segment = np.where(track_segment_lenght >= npt)[0]

    # nb_fig = 0

    if selected_track_segment.size > 0:

        nb_segment = 0
        nb_sub_segment = 0

        for track in selected_track_segment:

            nb_segment += 1
            # print "Processing segment %s / %s" %(str(nb_segment), str(selected_track_segment.size))

            if track-1 >= 0:
                index_start_selected_track = indi[track-1]
                index_end_selected_track = indi[track]
            else:
                index_start_selected_track = 0
                index_end_selected_track = indi[track]

            start_point = index_start_selected_track
            end_point = index_end_selected_track

            for sub_segment_point in range(start_point, end_point - npt, int(npt / 4)):

                nb_sub_segment += 1

                # Near Greenwhich case
                if ((lon_along_track[sub_segment_point + npt - 1] < 50.)
                    and (lon_along_track[sub_segment_point] > 320.)) \
                        or ((lon_along_track[sub_segment_point + npt - 1] > 320.)
                            and (lon_along_track[sub_segment_point] < 50.)):

                    tmp_lon = np.where(lon_along_track[sub_segment_point:sub_segment_point + npt] > 180,
                                       lon_along_track[sub_segment_point:sub_segment_point + npt] - 360,
                                       lon_along_track[sub_segment_point:sub_segment_point + npt])
                    mean_lon_sub_segment = np.median(tmp_lon)

                    if mean_lon_sub_segment < 0:
                        mean_lon_sub_segment = mean_lon_sub_segment + 360.
                else:

                    mean_lon_sub_segment = np.median(lon_along_track[sub_segment_point:sub_segment_point + npt])

                mean_lat_sub_segment = np.median(lat_along_track[sub_segment_point:sub_segment_point + npt])

                # plt.figure()
                # bmap = Basemap(
                #     projection='cyl', resolution='l', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360)
                # bmap.fillcontinents(color='grey', lake_color='white')
                # bmap.drawcoastlines()
                # bmap.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0], fontsize=15, dashes=[4, 4],
                #                    linewidth=0.25)
                # bmap.drawmeridians(np.arange(-180, 360, 30), labels=[0, 0, 0, 1], fontsize=15, dashes=[4, 4],
                #                    linewidth=0.25)
                # x, y = bmap(lon_along_track[start_point:end_point], lat_along_track[start_point:end_point])
                # bmap.scatter(x, y, s=0.1, marker="o", color='g')
                # x, y = bmap(lon_along_track[sub_segment_point:sub_segment_point + npt],
                #             lat_along_track[sub_segment_point:sub_segment_point + npt])
                # bmap.scatter(x, y, s=0.1, marker="o", color='r')
                # bmap.scatter(mean_lon_sub_segment, mean_lat_sub_segment, s=100, color='k', marker='x')
                # plt.title("Tracks and segments used in the computation")
                # # plt.show()
                # plt.savefig('fig_%04d.png' % nb_fig)
                # nb_fig += 1

                sla_segment = np.ma.masked_invalid(sla[sub_segment_point:sub_segment_point + npt])

                # plt.plot(sla_segment, color='r', label="SLA along-track")
                # plt.plot(msla_segment, color='b', label='interpolated MSLA')
                # plt.legend(loc="best")
                # plt.show()

                if sla_segment.size > 0:
                    list_sla_segment.append(sla_segment)
                    list_lon_segment.append(mean_lon_sub_segment)
                    list_lat_segment.append(mean_lat_sub_segment)

    return list_sla_segment, list_lon_segment, list_lat_segment


def selection_in_latlonbox(lon_array, lat_array, lon_min, lon_max, lat_min, lat_max):
    """

    :param lon_array:
    :param lat_array:
    :param lon_min:
    :param lon_max:
    :param lat_min:
    :param lat_max:
    :return:
    """

    selected_lat_index = np.where(np.logical_and(lat_array >= lat_min, lat_array <= lat_max))[0]

    if lon_min < 0.:
        selected_lon_index1 = np.where(np.logical_and(lon_array >= 0, lon_array <= lon_max))[0]
        selected_lon_index2 = np.where(np.logical_and(lon_array >= lon_min + 360., lon_array <= 360))[0]
        selected_lon_index = np.concatenate((selected_lon_index1, selected_lon_index2))
    elif lon_max > 360.:
        selected_lon_index1 = np.where(np.logical_and(lon_array >= lon_min, lon_array <= 360.))[0]
        selected_lon_index2 = np.where(np.logical_and(lon_array >= 0., lon_array <= lon_max - 360))[0]
        selected_lon_index = np.concatenate((selected_lon_index1, selected_lon_index2))
    else:
        selected_lon_index = np.where(np.logical_and(lon_array >= lon_min, lon_array <= lon_max))[0]

    selected_index = np.intersect1d(selected_lon_index, selected_lat_index)

    # bmap = Basemap(projection='cyl', resolution='l', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=0, urcrnrlon=360)
    # bmap.fillcontinents(color='grey', lake_color='white')
    # bmap.drawcoastlines()
    # bmap.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0], fontsize=20, dashes=[4, 4], linewidth=0.25)
    # bmap.drawmeridians(np.arange(-180, 360, 30), labels=[0, 0, 0, 1], fontsize=20, dashes=[4, 4], linewidth=0.25)
    # x, y = bmap(lon_array[selected_index], lat_array[selected_index])
    # bmap.scatter(x, y, s=0.1, marker="o", color='r')
    # plt.title("Point selected to estimate main value at lon=%s and lat =%s " %
    #           (str(0.5*(lon_min+lon_max)), str(0.5*(lat_min+lat_max))))
    # plt.show()

    return selected_index


def compute_resolution(array, frequency):
    """
    Given a coherence profile or spectral ratio, compute the resolution (i.e., where coherence = 0.5 or ratio = 0.5)
    :param array:
    :param frequency:
    :return:
    """
    # find index where coherence = 0.5 or ratio = 0.5
    try:
        ii05 = np.where(array[:] > 0.5)[0][-1]
    except:
        ii05 = -1

    if (ii05 + 1 < array[:].size) and (ii05 + 1 >= 0):
        d1 = array[ii05] - 0.5
        d2 = 0.5 - array[ii05 + 1]
        logres = np.log(frequency[ii05]) * (d2 / (d1 + d2)) + np.log(frequency[ii05 + 1]) * (
            d1 / (d1 + d2))
        resolution = 1. / np.exp(logres)

    else:
        # print "Error: ", ii, jj
        # plt.plot(1/frequency[:, jj, ii], coherence[:, jj, ii])
        # plt.show()
        resolution = 0.

    return resolution


def write_netcdf_scipy(output_netcdf_file, vlon, vlat, output_mean_frequency, output_mean_PSD_msla, output_mean_PSD_sla,
                       output_mean_PSD_diff_sla_msla, output_mean_PS_msla, output_mean_PS_sla,
                       output_mean_PS_diff_sla_msla,
                       output_mean_coherence, output_effective_lat, output_effective_lon, output_effective_resolution, output_useful_resolution,
                       output_nb_segment):

    array1 = np.swapaxes(
        np.asarray(output_mean_frequency).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_mean_frequency))[1])).T, 1, 2)

    array2 = np.swapaxes(
        np.asarray(output_mean_PSD_sla).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_mean_PSD_sla))[1])).T, 1, 2)

    array3 = np.swapaxes(
        np.asarray(output_mean_PSD_msla).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_mean_PSD_msla))[1])).T, 1, 2)

    array2b = np.swapaxes(
        np.asarray(output_mean_PS_sla).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_mean_PS_sla))[1])).T, 1, 2)

    array3b = np.swapaxes(
        np.asarray(output_mean_PS_msla).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_mean_PS_msla))[1])).T, 1, 2)

    array4 = np.swapaxes(
        np.asarray(output_mean_coherence).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_mean_coherence))[1])).T, 1, 2)

    array5 = np.swapaxes(np.asarray(output_effective_resolution).reshape((vlat.size, vlon.size)).T, 0, 1)

    array6 = np.swapaxes(np.asarray(output_effective_lat).reshape((vlat.size, vlon.size)).T, 0, 1)

    array7 = np.swapaxes(np.asarray(output_effective_lon).reshape((vlat.size, vlon.size)).T, 0, 1)

    array8 = np.swapaxes(np.asarray(output_nb_segment).reshape((vlat.size, vlon.size)).T, 0, 1)

    array9 = np.swapaxes(
        np.asarray(output_mean_PSD_diff_sla_msla).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_mean_PSD_diff_sla_msla))[1])).T, 1, 2)

    array10 = np.swapaxes(
        np.asarray(output_mean_PS_diff_sla_msla).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_mean_PS_diff_sla_msla))[1])).T, 1, 2)

    array11 = np.swapaxes(np.asarray(output_useful_resolution).reshape((vlat.size, vlon.size)).T, 0, 1)
    
    nc_out = Dataset(output_netcdf_file, 'w', format='NETCDF4')
    nc_out.createDimension('f', np.shape(np.asarray(output_mean_frequency))[1])
    nc_out.createDimension('y', vlat.size)
    nc_out.createDimension('x', vlon.size)
    frequence_out = nc_out.createVariable('frequency', 'f8', ('f', 'y', 'x'))
    spectre_out = nc_out.createVariable('spectrum_along_track', 'f8', ('f', 'y', 'x'))
    spectre_map_out = nc_out.createVariable('spectrum_map', 'f8', ('f', 'y', 'x'))
    psd_out = nc_out.createVariable('psd_along_track', 'f8', ('f', 'y', 'x'))
    psd_map_out = nc_out.createVariable('psd_map', 'f8', ('f', 'y', 'x'))
    spectre_diff_map_out = nc_out.createVariable('spectrum_diff_at_map', 'f8', ('f', 'y', 'x'))
    psd_diff_map_out = nc_out.createVariable('psd_diff_at_map', 'f8', ('f', 'y', 'x'))
    coherence_out = nc_out.createVariable('coherence', 'f8', ('f', 'y', 'x'))
    effective_resolution_out = nc_out.createVariable('effective_resolution', 'f8', ('y', 'x'))
    useful_resolution_out = nc_out.createVariable('useful_resolution', 'f8', ('y', 'x'))
    nb_segment_out = nc_out.createVariable('nb_segment', 'f8', ('y', 'x'))
    lat_out = nc_out.createVariable('lat2D', 'f8', ('y', 'x'))
    lon_out = nc_out.createVariable('lon2D', 'f8', ('y', 'x'))
    lat_out2 = nc_out.createVariable('lat', 'f8', ('y',))
    lon_out2 = nc_out.createVariable('lon', 'f8', ('x',))
    frequence_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array1 == 0, array1))
    psd_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array2 == 0, array2))
    psd_map_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array3 == 0, array3))
    spectre_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array2b == 0, array2b))
    spectre_map_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array3b == 0, array3b))
    coherence_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array4 == 0, array4))
    effective_resolution_out[:, :] = np.ma.masked_invalid(np.ma.masked_where(array5 == 0, array5))
    useful_resolution_out[:, :] = np.ma.masked_invalid(np.ma.masked_where(array11 == 0, array11))
    lat_out[:, :] = array6
    lon_out[:, :] = array7
    lat_out2[:] = array6[:, 0]
    lon_out2[:] = array7[0, :]
    nb_segment_out[:, :] = array8
    spectre_diff_map_out[:, :, :] = array9
    psd_diff_map_out[:, :, :] = array10
    nc_out.close()


def write_netcdf_alongtrack_spectrum(output_netcdf_file, vlon, vlat, output_mean_frequency, output_mean_PSD_sla,
                       output_mean_PS_sla, output_effective_lat, output_effective_lon, output_nb_segment):

    array1 = np.swapaxes(
        np.asarray(output_mean_frequency).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_mean_frequency))[1])).T, 1, 2)

    array2 = np.swapaxes(
        np.asarray(output_mean_PSD_sla).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_mean_PSD_sla))[1])).T, 1, 2)

    array2b = np.swapaxes(
        np.asarray(output_mean_PS_sla).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_mean_PS_sla))[1])).T, 1, 2)

    array6 = np.swapaxes(np.asarray(output_effective_lat).reshape((vlat.size, vlon.size)).T, 0, 1)

    array7 = np.swapaxes(np.asarray(output_effective_lon).reshape((vlat.size, vlon.size)).T, 0, 1)

    array8 = np.swapaxes(np.asarray(output_nb_segment).reshape((vlat.size, vlon.size)).T, 0, 1)

    nc_out = Dataset(output_netcdf_file, 'w', format='NETCDF4')
    nc_out.createDimension('f', np.shape(np.asarray(output_mean_frequency))[1])
    nc_out.createDimension('y', vlat.size)
    nc_out.createDimension('x', vlon.size)
    frequence_out = nc_out.createVariable('frequency', 'f8', ('f', 'y', 'x'))
    spectre_out = nc_out.createVariable('spectrum_along_track', 'f8', ('f', 'y', 'x'))
    psd_out = nc_out.createVariable('psd_along_track', 'f8', ('f', 'y', 'x'))
    nb_segment_out = nc_out.createVariable('nb_segment', 'f8', ('y', 'x'))
    lat_out = nc_out.createVariable('lat2D', 'f8', ('y', 'x'))
    lon_out = nc_out.createVariable('lon2D', 'f8', ('y', 'x'))
    lat_out2 = nc_out.createVariable('lat', 'f8', ('y',))
    lon_out2 = nc_out.createVariable('lon', 'f8', ('x',))
    frequence_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array1 == 0, array1))
    psd_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array2 == 0, array2))
    spectre_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array2b == 0, array2b))
    lat_out[:, :] = array6
    lon_out[:, :] = array7
    lat_out2[:] = array6[:, 0]
    lon_out2[:] = array7[0, :]
    nb_segment_out[:, :] = array8
    nc_out.close()


def compute_scipy_coherence_resolution_on_lonlat_mesh(vlon, vlat,
                                                      sla_segments, msla_segments,
                                                      lons, lats):

    list_effective_lon = []
    list_effective_lat = []
    list_mean_psd_msla = []
    list_mean_psd_sla = []
    list_mean_psd_diff_sla_msla = []
    list_mean_ps_msla = []
    list_mean_ps_sla = []
    list_mean_ps_diff_sla_msla = []
    list_mean_coherence = []
    list_mean_frequency = []
    list_effective_resolution = []
    list_useful_resolution = []
    list_nb_segment = []

    # Get number of point to consider for resolution = lenghtscale in km
    delta_t = get_deltat(mission.upper())

    delta_x = get_vitesse(mission.upper()) * delta_t

    npt = int(lenght_scale / delta_x)

    for ilat in vlat:

        if equal_area:
            lat_min = ilat - 0.5*change_in_latitude(lenght_scale)
            lat_max = ilat + 0.5*change_in_latitude(lenght_scale)
        else:
            lat_min = ilat - 0.5*delta_lat
            lat_max = ilat + 0.5*delta_lat

        effective_lat = 0.5 * (lat_max + lat_min)
        # print "Processing lat = ", effective_lat, str(datetime.datetime.now())

        for ilon in vlon:

            if equal_area:
                lon_min = ilon - 0.5*change_in_longitude(ilat, lenght_scale)
                lon_max = ilon + 0.5*change_in_longitude(ilat, lenght_scale)

            else:
                lon_min = ilon - 0.5*delta_lon
                lon_max = ilon + 0.5*delta_lon

            effective_lon = 0.5 * (lon_max + lon_min)
            if effective_lon > 360:
                effective_lon = effective_lon - 360

            # print "Processing lat = ", effective_lat, "lon = ,", effective_lon, str(datetime.datetime.now())

            selected_segment = selection_in_latlonbox(lons, lats, lon_min, lon_max, lat_min, lat_max)

            if selected_segment.size > 10:

                # plt.plot(sla_segments[selected_segment].flatten(), color='r', label='SLA')
                # plt.plot(msla_segments[selected_segment].flatten(), color='b', label='MSLA')
                # plt.legend(loc='best')
                # plt.show()

                diff_sla_msla = sla_segments[selected_segment].flatten() - msla_segments[selected_segment].flatten()

                wavenumber_scipy, ps_diff_sla_msla = scipy.signal.welch(
                    diff_sla_msla, fs=1.0 / delta_x, nperseg=npt, scaling='spectrum', noverlap=0)

                wavenumber_scipy, psd_diff_sla_msla = scipy.signal.welch(
                    diff_sla_msla, fs=1.0 / delta_x, nperseg=npt, scaling='density', noverlap=0)

                wavenumber_scipy, ps_msla = scipy.signal.welch(
                    msla_segments[selected_segment].flatten(), fs=1.0 / delta_x, nperseg=npt,
                    scaling='spectrum', noverlap=0)

                wavenumber_scipy, ps_sla = scipy.signal.welch(
                    sla_segments[selected_segment].flatten(), fs=1.0 / delta_x, nperseg=npt,
                    scaling='spectrum', noverlap=0)

                wavenumber_scipy, psd_msla = scipy.signal.welch(
                    msla_segments[selected_segment].flatten(), fs=1.0 / delta_x, nperseg=npt,
                    scaling='density', noverlap=0)

                wavenumber_scipy, psd_sla = scipy.signal.welch(
                    sla_segments[selected_segment].flatten(), fs=1.0 / delta_x, nperseg=npt,
                    scaling='density', noverlap=0)

                wavenumber_scipy, coherence_scipy = scipy.signal.coherence(
                    msla_segments[selected_segment].flatten(),
                    sla_segments[selected_segment].flatten(), fs=1.0 / delta_x, nperseg=npt, noverlap=0)

                effective_resolution = compute_resolution(coherence_scipy, wavenumber_scipy)

                useful_resolution = compute_resolution(psd_msla/psd_sla, wavenumber_scipy)

                list_effective_lon.append(effective_lon)
                list_effective_lat.append(effective_lat)
                list_mean_psd_msla.append(psd_msla)
                list_mean_psd_sla.append(psd_sla)
                list_mean_psd_diff_sla_msla.append(psd_diff_sla_msla)
                list_mean_ps_msla.append(ps_msla)
                list_mean_ps_sla.append(ps_sla)
                list_mean_ps_diff_sla_msla.append(ps_diff_sla_msla)
                list_mean_coherence.append(coherence_scipy)
                list_mean_frequency.append(wavenumber_scipy)
                list_effective_resolution.append(effective_resolution)
                list_useful_resolution.append(useful_resolution)
                list_nb_segment.append(selected_segment.size)
            else:
                list_effective_lon.append(effective_lon)
                list_effective_lat.append(effective_lat)
                list_mean_psd_msla.append(np.zeros((npt/2+1)))
                list_mean_psd_sla.append(np.zeros((npt/2+1)))
                list_mean_psd_diff_sla_msla.append(np.zeros((npt / 2 + 1)))
                list_mean_ps_msla.append(np.zeros((npt/2+1)))
                list_mean_ps_sla.append(np.zeros((npt/2+1)))
                list_mean_ps_diff_sla_msla.append(np.zeros((npt / 2 + 1)))
                list_mean_coherence.append(np.zeros((npt/2+1)))
                list_mean_frequency.append(np.zeros((npt/2+1)))
                list_effective_resolution.append(0.)
                list_useful_resolution.append(0.)
                list_nb_segment.append(0.)

    return list_effective_lon, list_effective_lat, list_mean_psd_msla, list_mean_psd_sla, list_mean_psd_diff_sla_msla, \
           list_mean_ps_msla, list_mean_ps_sla, list_mean_ps_diff_sla_msla, list_mean_frequency, list_mean_coherence, \
           list_effective_resolution, list_useful_resolution, list_nb_segment


def compute_scipy_alongtrack_spectrum_on_lonlat_mesh(vlon, vlat, sla_segments, lons, lats):

    list_effective_lon = []
    list_effective_lat = []
    list_mean_psd_sla = []
    list_mean_ps_sla = []
    list_mean_frequency = []
    list_nb_segment = []

    # Get number of point to consider for resolution = lenghtscale in km
    delta_t = get_deltat(mission.upper())

    delta_x = get_vitesse(mission.upper()) * delta_t

    npt = int(lenght_scale / delta_x)

    for ilat in vlat:

        if equal_area:
            lat_min = ilat - 0.5*change_in_latitude(lenght_scale)
            lat_max = ilat + 0.5*change_in_latitude(lenght_scale)
        else:
            lat_min = ilat - 0.5*delta_lat
            lat_max = ilat + 0.5*delta_lat

        effective_lat = 0.5 * (lat_max + lat_min)
        # print "Processing lat = ", effective_lat, str(datetime.datetime.now())

        for ilon in vlon:

            if equal_area:
                lon_min = ilon - 0.5*change_in_longitude(ilat, lenght_scale)
                lon_max = ilon + 0.5*change_in_longitude(ilat, lenght_scale)

            else:
                lon_min = ilon - 0.5*delta_lon
                lon_max = ilon + 0.5*delta_lon

            effective_lon = 0.5 * (lon_max + lon_min)
            if effective_lon > 360:
                effective_lon = effective_lon - 360

            # print "Processing lat = ", effective_lat, "lon = ,", effective_lon, str(datetime.datetime.now())

            selected_segment = selection_in_latlonbox(lons, lats, lon_min, lon_max, lat_min, lat_max)

            if selected_segment.size > 10:

                # plt.plot(sla_segments[selected_segment].flatten(), color='r', label='SLA')
                # plt.legend(loc='best')
                # plt.show()

                wavenumber_scipy, ps_sla = scipy.signal.welch(
                    sla_segments[selected_segment].flatten(), fs=1.0 / delta_x, nperseg=npt,
                    scaling='spectrum', noverlap=0)

                wavenumber_scipy, psd_sla = scipy.signal.welch(
                    sla_segments[selected_segment].flatten(), fs=1.0 / delta_x, nperseg=npt,
                    scaling='density', noverlap=0)

                list_effective_lon.append(effective_lon)
                list_effective_lat.append(effective_lat)
                list_mean_psd_sla.append(psd_sla)
                list_mean_ps_sla.append(ps_sla)
                list_mean_frequency.append(wavenumber_scipy)
                list_nb_segment.append(selected_segment.size)
            else:
                list_effective_lon.append(effective_lon)
                list_effective_lat.append(effective_lat)
                list_mean_psd_sla.append(np.zeros((npt/2+1)))
                list_mean_ps_sla.append(np.zeros((npt/2+1)))
                list_mean_frequency.append(np.zeros((npt/2+1)))
                list_nb_segment.append(0.)

    return list_effective_lon, list_effective_lat, list_mean_psd_sla, list_mean_ps_sla, \
           list_mean_frequency, list_nb_segment


# Read and select independent along-track data in the time windows
fid = Dataset(input_file_independent_alongtrack, 'r')
timep = np.array(fid.variables['time'][:])
lonp = np.array(fid.variables['longitude'][:])
# For Med Sea
if flag_roll:
    lonp = np.where(lonp >= 180, lonp-360, lonp)
latp = np.array(fid.variables['latitude'][:])
inds = np.where((timep >= study_time_min) & (timep <= study_time_max) &
                (lonp >= study_lon_min - buffer_zone) & (lonp <= study_lon_max + buffer_zone) &
                (latp >= study_lat_min - buffer_zone) & (latp <= study_lat_max + buffer_zone))[0]
lonp = np.array(fid.variables['longitude'][inds])
latp = np.array(fid.variables['latitude'][inds])
slap = np.array(fid.variables['SLA'][inds])
timep = timep[inds]
fid.close()

# Remove aberrant data that might be present in along-track (spatio-temporal incoherence BUG ConvertirTableResidus)
if flag_edit_spatiotemporal_incoherence:
    timep, lonp, latp, slap = edit_bad_velocity(input_file_independent_alongtrack)
    inds = np.where((timep >= study_time_min) & (timep <= study_time_max) &
                    (lonp >= study_lon_min - buffer_zone) & (lonp <= study_lon_max + buffer_zone) &
                    (latp >= study_lat_min - buffer_zone) & (latp <= study_lat_max + buffer_zone))[0]
    lonp = lonp[inds]
    latp = latp[inds]
    slap = slap[inds]
    timep = timep[inds]

# Edit coastal value
print("start coastal editing", str(datetime.datetime.now()))
if flag_edit_coastal:
    slap, timep, lonp, latp = edit_coastal_data(slap, lonp, latp, timep)
print("end coastal editing", str(datetime.datetime.now()))

# Interpolate when gap is smaller than XX
# TODO if necessary

if flag_alongtrack_only:
    print("start computation", str(datetime.datetime.now()))
    # Compute segment
    computed_sla_segment, computed_lon_segment, computed_lat_segment = \
        compute_segment_alongtrack_only(slap, lonp, latp, timep)
    print("end computation", str(datetime.datetime.now()))

    print("start gridding", str(datetime.datetime.now()))
    # Compute coherence and resolution
    output_effective_lon, output_effective_lat, output_mean_PSD_sla, \
    output_mean_PS_sla, output_mean_frequency, output_nb_segment = \
        compute_scipy_alongtrack_spectrum_on_lonlat_mesh(grid_lon, grid_lat,
                                                          np.asarray(computed_sla_segment),
                                                          np.asarray(computed_lon_segment),
                                                          np.asarray(computed_lat_segment))
    print("end gridding", str(datetime.datetime.now()))

    # Write netCDF output
    print("start writing", str(datetime.datetime.now()))
    write_netcdf_alongtrack_spectrum(nc_file, grid_lon, grid_lat, output_mean_frequency, output_mean_PSD_sla,
                       output_mean_PS_sla, output_effective_lat, output_effective_lon, output_nb_segment)
    print("end writing", str(datetime.datetime.now()))

else:

    # Interpolate map of SLA onto along-track
    print("start MSLA interpolation", str(datetime.datetime.now()))
    MSLA_interpolated = interpolate_msla_on_alongtrack(timep, latp, lonp)
    print("end MSLA interpolation", str(datetime.datetime.now()))

    # plt.plot(timep, np.ma.masked_where(np.abs(MSLA_interpolated) > 10., slap),
    # color='k', label="Independent along-track", lw=2)
    # plt.plot(timep, np.ma.masked_where(np.abs(MSLA_interpolated) > 10., MSLA_interpolated),
    # color='r', label="MSLA interpolated", lw=2)
    # plt.ylabel('SLA (m)')
    # plt.xlabel('Timeline (Julian days since 1950-01-01)')
    # plt.legend(loc='best')
    # plt.show()

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

    print("start computation", str(datetime.datetime.now()))
    # Compute segment
    computed_sla_segment, computed_msla_segment, computed_lon_segment, computed_lat_segment = \
        compute_segment(slap, MSLA_interpolated, lonp, latp, timep)

    # plt.plot(np.asarray(computed_sla_segment).flatten(), color='r', label="SLA along-track")
    # plt.plot(np.asarray(computed_msla_segment).flatten(), color='b', label='interpolated MSLA')
    # plt.legend(loc="best")
    # plt.show()

    print("end computation", str(datetime.datetime.now()))

    print("start gridding", str(datetime.datetime.now()))
    # Compute coherence and resolution
    output_effective_lon, output_effective_lat, output_mean_PSD_msla, output_mean_PSD_sla, \
    output_mean_PSD_diff_sla_msla, output_mean_PS_msla, output_mean_PS_sla, output_mean_PS_diff_sla_msla, \
    output_mean_frequency, output_mean_coherence, output_effective_resolution, output_useful_resolution, output_nb_segment = \
        compute_scipy_coherence_resolution_on_lonlat_mesh(grid_lon, grid_lat,
                                                          np.asarray(computed_sla_segment),
                                                          np.asarray(computed_msla_segment),
                                                          np.asarray(computed_lon_segment),
                                                          np.asarray(computed_lat_segment))

    print("end gridding", str(datetime.datetime.now()))
    
    # Write netCDF output
    print("start writing", str(datetime.datetime.now()))
    write_netcdf_scipy(nc_file, grid_lon, grid_lat, output_mean_frequency, output_mean_PSD_msla, output_mean_PSD_sla,
                       output_mean_PSD_diff_sla_msla, output_mean_PS_msla, output_mean_PS_sla,
                       output_mean_PS_diff_sla_msla,
                       output_mean_coherence,
                       output_effective_lat,
                       output_effective_lon,
                       output_effective_resolution,
                       output_useful_resolution,
                       output_nb_segment)
    print("end writing", str(datetime.datetime.now()))
