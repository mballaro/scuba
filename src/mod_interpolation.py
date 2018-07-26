import datetime

from mod_io import *
import scipy.interpolate
import glob
import numpy as np


def interpolate_msla_on_alongtrack(time_along_track, lat_along_track, lon_along_track,
                                   input_map_directory, map_file_pattern, time_ensemble, flag_roll):
    """
    Interpolate map of SLA onto an along-track dataset
    :param time_along_track:
    :param lat_along_track:
    :param lon_along_track:
    :param input_map_directory:
    :param map_file_pattern:
    :param time_ensemble:
    :param flag_roll:
    :return:
    """

    for tt in range(len(time_ensemble)):

        tref = time_ensemble[tt]
        date = datetime.datetime(1950, 1, 1) + datetime.timedelta(tref)
        char = date.strftime('%Y%m%d')

        file_type = input_map_directory + '/' + map_file_pattern + char + '_*.nc'

        filename = glob.glob(file_type)[0]
        ncfile = Dataset(filename, 'r')

        sla_map, lat_map, lon_map = read_cmems_format(ncfile)
        # sla_map, lat_map, lon_map = read_cls_format(ncfile)

        # For Med Sea
        if flag_roll:
            lon_map = np.where(lon_map >= 180, lon_map - 360, lon_map)

        # Mask invalid data if necessary
        sla_map = np.ma.masked_invalid(sla_map)

        # Initialisation saved sla map array
        if tt == 0:
            saved_sla_map = np.zeros((len(time_ensemble), np.shape(sla_map)[0], np.shape(sla_map)[1]))

        saved_sla_map[tt, :, :] = sla_map

    finterp_map2alongtrack = scipy.interpolate.RegularGridInterpolator(
            [time_ensemble, lat_map, lon_map],
            saved_sla_map, bounds_error=False, fill_value=None)

    msla_interpolated = finterp_map2alongtrack(np.transpose([time_along_track, lat_along_track, lon_along_track]))

    return msla_interpolated


# def fill_small_gap(sla, lon_along_track, lat_along_track, time_along_track):
#     """
#     Fill gab less than 3 points
#     :param sla:
#     :param lon_along_track:
#     :param lat_along_track:
#     :param time_along_track:
#     :return:
#     """
#     # Get number of point to consider for resolution = lenghtscale in km
#     delta_t = get_deltat(mission.upper())
#
#     # Convert delta_t from second to Julian Day
#     delta_t_jd = delta_t / (3600 * 24)
#
#     indi_for_linear_interp = np.where((np.diff(time_along_track) > delta_t_jd)
#                                       & (np.diff(time_along_track) < 4 * delta_t_jd))[0]
#
#     # Initialize NEW DATA
#     filled_sla = np.copy(sla)
#     filled_lon_along_track = np.copy(lon_along_track)
#     filled_lat_along_track = np.copy(lat_along_track)
#     filled_time_along_track = np.copy(time_along_track)
#
#     if indi_for_linear_interp.size > 0:
#         for ii in range(indi_for_linear_interp.size):
#             # select the two data points whre to interpolate
#             t1 = time_along_track[indi_for_linear_interp[ii]]
#             lon1 = lon_along_track[indi_for_linear_interp[ii]]
#             lat1 = lat_along_track[indi_for_linear_interp[ii]]
#             sla1 = sla[indi_for_linear_interp[ii]]
#
#             t2 = time_along_track[indi_for_linear_interp[ii] + 1]
#             lon2 = lon_along_track[indi_for_linear_interp[ii] + 1]
#             lat2 = lat_along_track[indi_for_linear_interp[ii] + 1]
#             sla2 = sla[indi_for_linear_interp[ii] + 1]
#
#             flon_interp = scipy.interpolate.interp1d([t1, t2], [lon1, lon2])
#             new_lon = flon_interp(np.linspace(t1, t2, int(round(abs(t2 - t1) / delta_t_jd)) + 1))
#
#             flat_interp = scipy.interpolate.interp1d([t1, t2], [lat1, lat2])
#             new_lat = flat_interp(np.linspace(t1, t2, int(round(abs(t2 - t1) / delta_t_jd)) + 1))
#
#             fsla_interp = scipy.interpolate.interp1d([t1, t2], [sla1, sla2])
#             new_sla = fsla_interp(np.linspace(t1, t2, int(round(abs(t2 - t1) / delta_t_jd)) + 1))
#
#             new_time = np.linspace(t1, t2, int(round(abs(t2 - t1) / delta_t_jd)))
#
#             filled_sla = np.insert(filled_sla, indi_for_linear_interp[ii] + 1,
#                                    new_sla[1:new_sla.size - 1], axis=None)
#             filled_lon_along_track = np.insert(filled_lon_along_track, indi_for_linear_interp[ii] + 1,
#                                                new_lon[1:new_lon.size - 1], axis=None)
#             filled_lat_along_track = np.insert(filled_lat_along_track, indi_for_linear_interp[ii] + 1,
#                                                new_lat[1:new_lat.size - 1], axis=None)
#             filled_time_along_track = np.insert(filled_time_along_track, indi_for_linear_interp[ii] + 1,
#                                                 new_time[1:new_time.size], axis=None)
