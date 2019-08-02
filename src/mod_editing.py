from netCDF4 import Dataset
import numpy as np

from mod_io import *
from mod_geo import *


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
        deltat[ii] = (time[ii+1] - time[ii])*24*3600

    velocity = deltax/deltat
    lon_ok = np.ma.masked_where((velocity > max_velocity_criteria) & (velocity < min_velocity_criteria), lon[:])
    lat_ok = np.ma.masked_where((velocity > max_velocity_criteria) & (velocity < min_velocity_criteria), lat[:])
    time_ok = np.ma.masked_where((velocity > max_velocity_criteria) & (velocity < min_velocity_criteria), time[:])
    sla_ok = np.ma.masked_where((velocity > max_velocity_criteria) & (velocity < min_velocity_criteria), sla[:])

    return time_ok, lon_ok, lat_ok, sla_ok


def edit_coastal_data(field, lon_along_track, lat_along_track, time_along_track,
                      file_coastal_distance, coastal_criteria, flag_roll):
    """
    Edit coastal along-track dta based on the coastal distance file
    :param field:
    :param lon_along_track:
    :param lat_along_track:
    :param time_along_track:
    :param file_coastal_distance:
    :param coastal_criteria:
    :param flag_roll:
    :return:
    """

    # Read coastal distance file
    ncfile = Dataset(file_coastal_distance, 'r')
    distance = ncfile.variables['distance'][:, :]
    lon_distance = ncfile.variables['lon'][:]
    lat_distance = ncfile.variables['lat'][:]
    coastal_flag = np.zeros((len(field)))
    ncfile.close()

    # For Med Sea
    if flag_roll:
        lon_distance = np.where(lon_distance >= 180, lon_distance - 360, lon_distance)

    # Prepare mask for coastal region
    for time_index in range(field.size):
        # nearest lon_distance from lon_along_track
        ii_nearest = find_nearest_index(lon_distance, lon_along_track[time_index])
        # nearest lat_distance from lat_along_track
        jj_nearest = find_nearest_index(lat_distance, lat_along_track[time_index])

        if distance[jj_nearest, ii_nearest] > coastal_criteria:
            coastal_flag[time_index] = 1.0

    edited_field = np.ma.compressed(np.ma.masked_where(coastal_flag == 0., field))
    edited_lon_along_track = np.ma.compressed(np.ma.masked_where(coastal_flag == 0., lon_along_track))
    edited_lat_along_track = np.ma.compressed(np.ma.masked_where(coastal_flag == 0., lat_along_track))
    edited_time_along_track = np.ma.compressed(np.ma.masked_where(coastal_flag == 0., time_along_track))

    return edited_field, edited_time_along_track, edited_lon_along_track, edited_lat_along_track
