import datetime
from mod_io import *
import scipy.interpolate
import glob
import numpy as np

try:
    import pyinterp.core
except ImportError:
    pass


def interpolate_msla_on_alongtrack(time_along_track, lat_along_track, lon_along_track, config):
    """
    Interpolate map of SLA onto an along-track dataset
    :param time_along_track:
    :param lat_along_track:
    :param lon_along_track:
    :param config:
    :return:
    """

    flag_roll = config['properties']['study_area']['flag_roll']
    start = datetime.datetime.strptime(str(config['properties']['time_window']['YYYYMMDD_min']), '%Y%m%d')
    end = datetime.datetime.strptime(str(config['properties']['time_window']['YYYYMMDD_max']), '%Y%m%d')
    study_time_min = int(date2num(start, units="days since 1950-01-01", calendar='standard'))
    study_time_max = int(date2num(end, units="days since 1950-01-01", calendar='standard'))
    time_ensemble = np.arange(study_time_min, study_time_max + 1, 1)

    input_map_directory = config['inputs']['input_map_directory']
    map_file_pattern = config['inputs']['map_file_pattern']
    study_field_scale_factor = config['inputs']['study_field_scale_factor']

    saved_sla_map = None
    lon_map = None
    lat_map = None

    for tt in range(len(time_ensemble)):

        tref = time_ensemble[tt]
        date = datetime.datetime(1950, 1, 1) + datetime.timedelta(np.float(tref))
        char = date.strftime('%Y%m%d')

        file_type = input_map_directory + '/' + map_file_pattern + char + '_*.nc'

        filename = glob.glob(file_type)[0]
        ncfile = Dataset(filename, 'r')

        if config['inputs']['study_field_type_CLS']:
            sla_map, lat_map, lon_map = read_cls_format(ncfile)
        else:
            sla_map, lat_map, lon_map = read_cmems_format(ncfile)

        # For Med Sea
        if flag_roll:
            lon_map = np.where(lon_map >= 180, lon_map - 360, lon_map)

        # Mask invalid data if necessary
        sla_map = np.ma.masked_invalid(sla_map)

        # Initialisation saved sla map array
        if tt == 0:
            saved_sla_map = np.zeros((len(time_ensemble), np.shape(sla_map)[0], np.shape(sla_map)[1]))

        saved_sla_map[tt, :, :] = sla_map

    # interpolator = pyinterp.core.interp3d.Trivariate(pyinterp.core.Axis(lon_map, is_circle=True),
    #                                                  pyinterp.core.Axis(lat_map),
    #                                                  pyinterp.core.Axis(time_ensemble),
    #                                                  saved_sla_map.T)

    # msla_interpolated = interpolator.evaluate(lon_along_track,
    #                            lat_along_track,
    #                            time_along_track,
    #                            pyinterp.core.interp3d.Trivariate.Type.kTrilinear)

    finterp_map2alongtrack = scipy.interpolate.RegularGridInterpolator([time_ensemble, lat_map, lon_map],
                                                                       saved_sla_map,
                                                                       bounds_error=False,
                                                                       fill_value=None)

    msla_interpolated = finterp_map2alongtrack(np.transpose([time_along_track, lat_along_track, lon_along_track]))

    return study_field_scale_factor*msla_interpolated
