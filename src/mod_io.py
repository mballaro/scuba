from netCDF4 import Dataset
import numpy as np
import datetime

from mod_constant import *
from mod_geo import *
from mod_editing import *
from yaml import load


def read_grid_field(input_file_reference, lon_name, lat_name, field_name,
                    study_lon_min, study_lon_max, study_lat_min, study_lat_max, flag_ewp):
    """
    Read input field from netcdf file
    :param input_file_reference:
    :param lon_name:
    :param lat_name:
    :param field_name:
    :param study_lon_min:
    :param study_lon_max:
    :param study_lat_min:
    :param study_lat_max:
    :param flag_ewp:
    :return:
    """

    # Read reference MDT map
    fid = Dataset(input_file_reference, 'r')
    lon_ref_map = fid.variables[lon_name][:]
    lat_ref_map = fid.variables[lat_name][:]
    ii_min = find_nearest_index(lon_ref_map, study_lon_min - buffer_zone)
    ii_max = find_nearest_index(lon_ref_map, study_lon_max + buffer_zone)
    jj_min = find_nearest_index(lat_ref_map, study_lat_min - buffer_zone)
    jj_max = find_nearest_index(lat_ref_map, study_lat_max + buffer_zone)
    sla_ref_map = fid.variables[field_name][:, jj_min:jj_max, ii_min:ii_max]
    lon_ref_map = fid.variables[lon_name][ii_min:ii_max]
    lat_ref_map = fid.variables[lat_name][jj_min:jj_max]
    fid.close()

    delta_lon_in = np.abs(lon_ref_map[0] - lon_ref_map[1])

    if flag_ewp:
        npt_x_extra = np.int(20 / delta_lon_in)  # add 20 degree extra point on left and right
        lx = sla_ref_map[0, 0, :].size
        sla_ref_map_east = sla_ref_map[:, :, 0:npt_x_extra]
        lon_ref_map_east = lon_ref_map[0:npt_x_extra]
        sla_ref_map_west = sla_ref_map[:, :, lx - npt_x_extra:]
        lon_ref_map_west = lon_ref_map[lx - npt_x_extra:]

        tmp = np.concatenate((lon_ref_map_west - 360., lon_ref_map), axis=-1)
        final_lon = np.concatenate((tmp, lon_ref_map_east + 360), axis=-1)
        tmp = np.concatenate((sla_ref_map_west, sla_ref_map), axis=-1)
        final_sla = np.concatenate((tmp, sla_ref_map_east), axis=-1)

        sla_ref_map = final_sla
        lon_ref_map = final_lon

    sla_ref_map = np.ma.masked_where(np.abs(sla_ref_map) > 1.E10, sla_ref_map)

    return lon_ref_map, lat_ref_map, sla_ref_map, delta_lon_in


def read_grid_fieldT(input_file_reference, lon_name, lat_name, field_name,
                    study_lon_min, study_lon_max, study_lat_min, study_lat_max, flag_ewp):
    """
    Read input field from netcdf file
    :param input_file_reference:
    :param lon_name:
    :param lat_name:
    :param field_name:
    :param study_lon_min:
    :param study_lon_max:
    :param study_lat_min:
    :param study_lat_max:
    :param flag_ewp:
    :return:
    """

    # Read reference MDT map
    fid = Dataset(input_file_reference, 'r')
    lon_ref_map = fid.variables[lon_name][:]
    lat_ref_map = fid.variables[lat_name][:]
    time_ref_map = fid.variables["time"][:]
    ii_min = find_nearest_index(lon_ref_map, study_lon_min - buffer_zone)
    ii_max = find_nearest_index(lon_ref_map, study_lon_max + buffer_zone)
    jj_min = find_nearest_index(lat_ref_map, study_lat_min - buffer_zone)
    jj_max = find_nearest_index(lat_ref_map, study_lat_max + buffer_zone)
    sla_ref_map = fid.variables[field_name][:, jj_min:jj_max, ii_min:ii_max]
    lon_ref_map = fid.variables[lon_name][ii_min:ii_max]
    lat_ref_map = fid.variables[lat_name][jj_min:jj_max]
    fid.close()

    delta_lon_in = np.abs(lon_ref_map[0] - lon_ref_map[1])

    if flag_ewp:
        npt_x_extra = np.int(20 / delta_lon_in)  # add 20 degree extra point on left and right
        lx = sla_ref_map[0, 0, :].size
        sla_ref_map_east = sla_ref_map[:, :, 0:npt_x_extra]
        lon_ref_map_east = lon_ref_map[0:npt_x_extra]
        sla_ref_map_west = sla_ref_map[:, :, lx - npt_x_extra:]
        lon_ref_map_west = lon_ref_map[lx - npt_x_extra:]

        tmp = np.concatenate((lon_ref_map_west - 360., lon_ref_map), axis=-1)
        final_lon = np.concatenate((tmp, lon_ref_map_east + 360), axis=-1)
        tmp = np.concatenate((sla_ref_map_west, sla_ref_map), axis=-1)
        final_sla = np.concatenate((tmp, sla_ref_map_east), axis=-1)

        sla_ref_map = final_sla
        lon_ref_map = final_lon

    sla_ref_map = np.ma.masked_where(np.abs(sla_ref_map) > 1.E10, sla_ref_map)

    return lon_ref_map, lat_ref_map, time_ref_map, sla_ref_map, delta_lon_in


def read_mdt(input_file_reference, study_lon_min, study_lon_max, study_lat_min, study_lat_max, flag_ewp):
    """

    :param input_file_reference:
    :param study_lon_min:
    :param study_lon_max:
    :param study_lat_min:
    :param study_lat_max:
    :param flag_ewp:
    :return:
    """

    # Read reference MDT map
    fid = Dataset(input_file_reference, 'r')
    lon_ref_map = fid.variables['lon'][:]
    lat_ref_map = fid.variables['lat'][:]
    ii_min = find_nearest_index(lon_ref_map, study_lon_min - buffer_zone)
    ii_max = find_nearest_index(lon_ref_map, study_lon_max + buffer_zone)
    jj_min = find_nearest_index(lat_ref_map, study_lat_min - buffer_zone)
    jj_max = find_nearest_index(lat_ref_map, study_lat_max + buffer_zone)
    sla_ref_map = fid.variables['mdt'][:, jj_min:jj_max, ii_min:ii_max]
    lon_ref_map = fid.variables['lon'][ii_min:ii_max]
    lat_ref_map = fid.variables['lat'][jj_min:jj_max]
    fid.close()

    delta_lon_in = np.abs(lon_ref_map[0] - lon_ref_map[1])

    if flag_ewp:
        npt_x_extra = np.int(20 / delta_lon_in)  # add 20 degree extra point on left and right
        lx = sla_ref_map[0, 0, :].size
        sla_ref_map_east = sla_ref_map[:, :, 0:npt_x_extra]
        lon_ref_map_east = lon_ref_map[0:npt_x_extra]
        sla_ref_map_west = sla_ref_map[:, :, lx - npt_x_extra:]
        lon_ref_map_west = lon_ref_map[lx - npt_x_extra:]

        tmp = np.concatenate((lon_ref_map_west - 360., lon_ref_map), axis=-1)
        final_lon = np.concatenate((tmp, lon_ref_map_east + 360), axis=-1)
        tmp = np.concatenate((sla_ref_map_west, sla_ref_map), axis=-1)
        final_sla = np.concatenate((tmp, sla_ref_map_east), axis=-1)

        sla_ref_map = final_sla
        lon_ref_map = final_lon

    sla_ref_map = np.ma.masked_where(np.abs(sla_ref_map) > 1.E10, sla_ref_map)

    return lon_ref_map, lat_ref_map, sla_ref_map, delta_lon_in


def read_natl60(input_file_reference, study_lon_min, study_lon_max, study_lat_min, study_lat_max, flag_ewp):
    """

    :param input_file_reference:
    :param study_lon_min:
    :param study_lon_max:
    :param study_lat_min:
    :param study_lat_max:
    :param flag_ewp:
    :return:
    """

    # Read reference MDT map
    fid = Dataset(input_file_reference, 'r')
    lon_ref_map = fid.variables['lon'][:]
    lat_ref_map = fid.variables['lat'][:]
    ii_min = find_nearest_index(lon_ref_map, study_lon_min - buffer_zone)
    ii_max = find_nearest_index(lon_ref_map, study_lon_max + buffer_zone)
    jj_min = find_nearest_index(lat_ref_map, study_lat_min - buffer_zone)
    jj_max = find_nearest_index(lat_ref_map, study_lat_max + buffer_zone)
    sla_ref_map = fid.variables['sossheig'][:, jj_min:jj_max, ii_min:ii_max]
    lon_ref_map = fid.variables['lon'][ii_min:ii_max]
    lat_ref_map = fid.variables['lat'][jj_min:jj_max]
    fid.close()

    delta_lon_in = np.abs(lon_ref_map[0] - lon_ref_map[1])

    if flag_ewp:
        npt_x_extra = np.int(20 / delta_lon_in)  # add 20 degree extra point on left and right
        lx = sla_ref_map[0, 0, :].size
        sla_ref_map_east = sla_ref_map[:, :, 0:npt_x_extra]
        lon_ref_map_east = lon_ref_map[0:npt_x_extra]
        sla_ref_map_west = sla_ref_map[:, :, lx - npt_x_extra:]
        lon_ref_map_west = lon_ref_map[lx - npt_x_extra:]

        tmp = np.concatenate((lon_ref_map_west - 360., lon_ref_map), axis=-1)
        final_lon = np.concatenate((tmp, lon_ref_map_east + 360), axis=-1)
        tmp = np.concatenate((sla_ref_map_west, sla_ref_map), axis=-1)
        final_sla = np.concatenate((tmp, sla_ref_map_east), axis=-1)

        sla_ref_map = final_sla
        lon_ref_map = final_lon

    sla_ref_map = np.ma.masked_where(np.abs(sla_ref_map) > 1.E10, sla_ref_map)

    return lon_ref_map, lat_ref_map, sla_ref_map, delta_lon_in


def read_along_track(input_file_alongtrack,
                     study_time_min, study_time_max,
                     study_lon_min, study_lon_max,
                     study_lat_min, study_lat_max,
                     flag_edit_spatiotemporal_incoherence,
                     flag_edit_coastal, file_coastal_distance, coastal_criteria, flag_roll):

    # Read and select independent along-track data in the time windows
    fid = Dataset(input_file_alongtrack, 'r')
    timep = np.array(fid.variables['time'][:])
    lonp = np.array(fid.variables['longitude'][:])

    # For Med Sea (a verifier)
    if flag_roll:
        lonp = np.where(lonp >= 180, lonp - 360, lonp)

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
        print("start suspicious velocity editing", str(datetime.datetime.now()))
        timep, lonp, latp, slap = edit_bad_velocity(input_file_independent_alongtrack)
        print("end suspicious velocity editing", str(datetime.datetime.now()))
        inds = np.where((timep >= study_time_min) & (timep <= study_time_max) &
                        (lonp >= study_lon_min - buffer_zone) & (lonp <= study_lon_max + buffer_zone) &
                        (latp >= study_lat_min - buffer_zone) & (latp <= study_lat_max + buffer_zone))[0]
        lonp = lonp[inds]
        latp = latp[inds]
        slap = slap[inds]
        timep = timep[inds]

    # Edit coastal value
    if flag_edit_coastal:
        print("start coastal editing", str(datetime.datetime.now()))
        slap, timep, lonp, latp = edit_coastal_data(slap, lonp, latp, timep,
                                                    file_coastal_distance, coastal_criteria, flag_roll)
        print("end coastal editing", str(datetime.datetime.now()))

    # Interpolate when gap is smaller than XXcoastal_criteria
    # TODO if necessary

    return slap, timep, lonp, latp


def read_tide_gauge(filename):
    """

    :param filename:
    :return:
    """

    nc = Dataset(filename, 'r')
    time_tg = nc.variables['time'][:]
    sla_alti = nc.variables['SLA_Alti'][:]
    sla_tg = nc.variables['SLA_TG'][:]
    lat_tg = nc.Latitude_TG
    lon_tg = nc.Longitude_TG
    nc.close()

    return sla_tg, sla_alti, time_tg, lat_tg, lon_tg


def read_cls_format(fcid):
    """
    Read sea level anomaly, lon and lat from CLS maps
    :param fcid:
    :return:
    """
    sla_map = np.array(fcid.variables['Grid_0001'][:, :]).transpose() / 100.  # convert in meters
    lat_map = np.array(fcid.variables['NbLatitudes'][:])
    lon_map = np.array(fcid.variables['NbLongitudes'][:])

    return sla_map, lat_map, lon_map


def read_cmems_format(fcid):
    """
    Read sea level anomaly, lon and lat from CMEMS maps
    :param fcid:
    :return:
    """
    sla_map = np.array(fcid.variables['sla'][0, :, :])  # in meters
    lat_map = np.array(fcid.variables['latitude'][:])
    lon_map = np.array(fcid.variables['longitude'][:])

    return sla_map, lat_map, lon_map


def get_velocity(cmission, mission_management):
    """
    Get velocity of a mission from MissionManagement.yaml
    """
    yaml = load(open(str(mission_management)))
    velocity = yaml[cmission]['VELOCITY']
    if velocity is not None:
        return velocity
    else:
        raise ValueError("velocity not found for mission %s in %s" % (cmission, mission_management))


def get_deltat(cmission, mission_management):
    """
    Get deltaT of a mission from file MissionManagement.yaml
    """
    yaml = load(open(str(mission_management)))
    deltat = yaml[cmission]['DELTA_T']
    if deltat is not None:
        return deltat
    else:
        raise ValueError("deltat not found for mission %s in %s" % (cmission, mission_management))


def write_netcdf_output(output_netcdf_file,
                        vlon, vlat, output_effective_lon, output_effective_lat,
                        output_mean_frequency, output_nb_segment, freq_unit,
                        output_mean_ps_sla_ref, output_mean_psd_sla_ref,
                        output_autocorrelation_distance,
                        output_autocorrelation_ref,
                        output_autocorrelation_ref_zero_crossing,
                        output_mean_ps_sla_study=None, output_mean_psd_sla_study=None,
                        output_autocorrelation_study=None,
                        output_autocorrelation_study_zero_crossing=None,
                        output_mean_ps_diff_sla_ref_sla_study=None, output_mean_psd_diff_sla_ref_sla_study=None,
                        output_mean_coherence=None, output_effective_resolution=None, output_useful_resolution=None,
                        output_cross_correlation=None):
    """

    :param output_netcdf_file:
    :param vlon:
    :param vlat:
    :param output_effective_lon:
    :param output_effective_lat:
    :param output_mean_frequency:
    :param output_nb_segment:
    :param freq_unit:
    :param output_mean_ps_sla_ref:
    :param output_mean_psd_sla_ref:
    :param output_autocorrelation_distance:
    :param output_autocorrelation_ref:
    :param output_autocorrelation_ref_zero_crossing:
    :param output_mean_ps_sla_study:
    :param output_mean_psd_sla_study:
    :param output_autocorrelation_study:
    :param output_autocorrelation_study_zero_crossing:
    :param output_mean_ps_diff_sla_ref_sla_study:
    :param output_mean_psd_diff_sla_ref_sla_study:
    :param output_mean_coherence:
    :param output_effective_resolution:
    :param output_useful_resolution:
    :return:
    """

    nc_out = Dataset(output_netcdf_file, 'w', format='NETCDF4')
    nc_out.createDimension('f', np.shape(np.asarray(output_mean_frequency))[1])
    nc_out.createDimension('d', np.shape(np.asarray(output_autocorrelation_distance))[1])
    nc_out.createDimension('y', vlat.size)
    nc_out.createDimension('x', vlon.size)

    array1 = np.swapaxes(
        np.asarray(output_mean_frequency).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_mean_frequency))[1])).T, 1, 2)
    frequence_out = nc_out.createVariable('frequency', 'f8', ('f', 'y', 'x'))
    frequence_out.units = "1/%s" % freq_unit
    frequence_out.axis = 'T'
    frequence_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array1 == 0, array1))

    array2 = np.swapaxes(np.asarray(output_nb_segment).reshape((vlat.size, vlon.size)).T, 0, 1)
    nb_segment_out = nc_out.createVariable('nb_segment', 'f8', ('y', 'x'))
    nb_segment_out.long_name = "number of segment used in spectral computation"
    nb_segment_out[:, :] = array2

    array3 = np.swapaxes(np.asarray(output_effective_lat).reshape((vlat.size, vlon.size)).T, 0, 1)
    lat_out = nc_out.createVariable('lat2D', 'f8', ('y', 'x'))
    lat_out[:, :] = array3

    array4 = np.swapaxes(np.asarray(output_effective_lon).reshape((vlat.size, vlon.size)).T, 0, 1)
    lon_out = nc_out.createVariable('lon2D', 'f8', ('y', 'x'))
    lon_out[:, :] = array4

    lat_out2 = nc_out.createVariable('lat', 'f8', ('y',))
    lat_out2[:] = array3[:, 0]

    lon_out2 = nc_out.createVariable('lon', 'f8', ('x',))
    lon_out2[:] = array4[0, :]

    array5 = np.swapaxes(
        np.asarray(output_mean_ps_sla_ref).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_mean_ps_sla_ref))[1])).T, 1, 2)
    spectrum_ref = nc_out.createVariable('spectrum_ref', 'f8', ('f', 'y', 'x'))
    spectrum_ref.units = 'm2'
    spectrum_ref.coordinates = "frequency lat lon"
    spectrum_ref.long_name = "power spectrum reference field"
    spectrum_ref[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array5 == 0, array5))

    array6 = np.swapaxes(
        np.asarray(output_mean_psd_sla_ref).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_mean_psd_sla_ref))[1])).T, 1, 2)
    psd_ref = nc_out.createVariable('psd_ref', 'f8', ('f', 'y', 'x'))
    psd_ref.units = 'm2/%s' % freq_unit
    psd_ref.coordinates = "frequency lat lon"
    psd_ref.long_name = "power spectrum density reference field"
    psd_ref[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array6 == 0, array6))

    array7 = np.swapaxes(
        np.asarray(output_autocorrelation_distance).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_autocorrelation_distance))[1])).T, 1, 2)
    autocorrelation_distance = nc_out.createVariable('distance', 'f8', ('d', 'y', 'x'))
    autocorrelation_distance.units = freq_unit
    autocorrelation_distance.long_name = 'distance from origin autocorrelation function reference field'
    autocorrelation_distance[:, :, :] = array7

    array8 = np.swapaxes(
        np.asarray(output_autocorrelation_ref).reshape(
            (vlat.size, vlon.size, np.shape(np.asarray(output_autocorrelation_distance))[1])).T, 1, 2)
    autocorrelation_ref = nc_out.createVariable('autocorrelation_ref', 'f8', ('d', 'y', 'x'))
    autocorrelation_ref.coordinates = "distance lat lon"
    autocorrelation_ref.long_name = "autocorrelation function reference field computed from power spectrum density"
    autocorrelation_ref[:, :, :] = array8

    array9 = np.swapaxes(np.asarray(output_autocorrelation_ref_zero_crossing).reshape((vlat.size, vlon.size)).T, 0, 1)
    zero_crossing_ref = nc_out.createVariable('zero_crossing_ref', 'f8', ('y', 'x'))
    zero_crossing_ref.coordinates = "lat lon"
    zero_crossing_ref.units = freq_unit
    zero_crossing_ref.long_name = "zero crossing autocorrelation function reference field"
    zero_crossing_ref[:, :] = array9

    if output_mean_ps_sla_study is not None:
        array10 = np.swapaxes(
            np.asarray(output_mean_ps_sla_study).reshape(
                (vlat.size, vlon.size, np.shape(np.asarray(output_mean_ps_sla_study))[1])).T, 1, 2)
        spectrum_study = nc_out.createVariable('spectrum_study', 'f8', ('f', 'y', 'x'))
        spectrum_study.units = 'm2'
        spectrum_study.coordinates = "frequency lat lon"
        spectrum_study.long_name = "power spectrum study field"
        spectrum_study[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array10 == 0, array10))

    if output_mean_psd_sla_study is not None:
        array11 = np.swapaxes(
            np.asarray(output_mean_psd_sla_study).reshape(
                (vlat.size, vlon.size, np.shape(np.asarray(output_mean_psd_sla_study))[1])).T, 1, 2)
        psd_study = nc_out.createVariable('psd_study', 'f8', ('f', 'y', 'x'))
        psd_study.units = 'm2/%s' % freq_unit
        psd_study.coordinates = "frequency lat lon"
        psd_study.long_name = "power spectrum density study field"
        psd_study[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array11 == 0, array11))

    if output_autocorrelation_study:
        array12 = np.swapaxes(
            np.asarray(output_autocorrelation_study).reshape(
                (vlat.size, vlon.size, np.shape(np.asarray(output_autocorrelation_study))[1])).T, 1, 2)
        autocorrelation_study = nc_out.createVariable('autocorrelation_study', 'f8', ('d', 'y', 'x'))
        autocorrelation_study.coordinates = "distance lat lon"
        autocorrelation_study.long_name = "autocorrelation function study field computed from power spectrum density"
        autocorrelation_study[:, :, :] = array12

    if output_autocorrelation_study_zero_crossing:
        array13 = np.swapaxes(np.asarray(output_autocorrelation_study_zero_crossing).reshape((vlat.size, vlon.size)).T,
                              0, 1)
        zero_crossing_study = nc_out.createVariable('zero_crossing_study', 'f8', ('y', 'x'))
        zero_crossing_study.coordinates = "lat lon"
        zero_crossing_study.units = freq_unit
        zero_crossing_study.long_name = "zero crossing autocorrelation function study field"
        zero_crossing_study[:, :] = array13

    if output_mean_ps_diff_sla_ref_sla_study is not None:
        array14 = np.swapaxes(
            np.asarray(output_mean_ps_diff_sla_ref_sla_study).reshape(
                (vlat.size, vlon.size, np.shape(np.asarray(output_mean_ps_diff_sla_ref_sla_study))[1])).T, 1, 2)
        spectrum_diff = nc_out.createVariable('spectrum_diff', 'f8', ('f', 'y', 'x'))
        spectrum_diff.units = 'm2'
        spectrum_diff.coordinates = "frequency lat lon"
        spectrum_diff.long_name = "power spectrum of difference study minus reference field"
        spectrum_diff[:, :, :] = array14

    if output_mean_psd_diff_sla_ref_sla_study is not None:
        array15 = np.swapaxes(
            np.asarray(output_mean_psd_diff_sla_ref_sla_study).reshape(
                (vlat.size, vlon.size, np.shape(np.asarray(output_mean_psd_diff_sla_ref_sla_study))[1])).T, 1, 2)
        psd_diff = nc_out.createVariable('psd_diff', 'f8', ('f', 'y', 'x'))
        psd_diff.units = 'm2/%s' % freq_unit
        psd_diff.coordinates = "frequency lat lon"
        psd_diff.long_name = "power spectrum density of difference study minus reference field"
        psd_diff[:, :, :] = array15

    if output_mean_coherence is not None:
        array16 = np.swapaxes(
            np.asarray(output_mean_coherence).reshape(
                (vlat.size, vlon.size, np.shape(np.asarray(output_mean_coherence))[1])).T, 1, 2)
        coherence_out = nc_out.createVariable('coherence', 'f8', ('f', 'y', 'x'))
        coherence_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array16 == 0, array16))
        coherence_out.coordinates = "frequency lat lon"
        coherence_out.long_name = "magnitude squared coherence between reference and study fields"

    if output_effective_resolution is not None:
        array17 = np.swapaxes(np.asarray(output_effective_resolution).reshape((vlat.size, vlon.size)).T, 0, 1)
        effective_resolution_out = nc_out.createVariable('effective_resolution', 'f8', ('y', 'x'))
        effective_resolution_out.units = freq_unit
        effective_resolution_out.coordinates = "lat lon"
        effective_resolution_out.long_name = "effective resolution study field computed from " \
                                             "magnitude squared coherence between reference and study fields"
        effective_resolution_out.comment = 'effective resolution is the wavenumber where the ' \
                                           'magnitude squared coherence = 0.5'
        effective_resolution_out[:, :] = np.ma.masked_invalid(np.ma.masked_where(array17 == 0, array17))

    if output_useful_resolution is not None:
        array18 = np.swapaxes(np.asarray(output_useful_resolution).reshape((vlat.size, vlon.size)).T, 0, 1)
        useful_resolution_out = nc_out.createVariable('useful_resolution', 'f8', ('y', 'x'))
        useful_resolution_out.units = freq_unit
        useful_resolution_out[:, :] = np.ma.masked_invalid(np.ma.masked_where(array18 == 0, array18))
        useful_resolution_out.coordinates = "lat lon"
        useful_resolution_out.long_name = "useful/available resolution study field computed from ratio " \
                                          "between study and reference power spectrum"
        useful_resolution_out.comment = "useful/available resolution is the wavenumber where the ratio = 0.5"

    if output_cross_correlation is not None:
        nc_out.createDimension('dc', np.shape(np.asarray(output_cross_correlation))[1])
        array19 = np.swapaxes(
            np.asarray(output_cross_correlation).reshape(
                (vlat.size, vlon.size, np.shape(np.asarray(output_cross_correlation))[1])).T, 1, 2)
        cross_correlation = nc_out.createVariable('cross_correlation', 'f8', ('dc', 'y', 'x'))
        cross_correlation.units = ''
        cross_correlation.coordinates = "segment_lenght lat lon"
        cross_correlation.long_name = "squared cross_correlation"
        cross_correlation[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(array19 == 0, array19))

    nc_out.close()


def write_netcdf_TG(filename, frequency_out, effective_resolution_out, useful_resolution_out,
                    lat_out, lon_out,
                    spectrum_TG_out, spectrum_SLA_at_TG,
                    power_spectrum_TG_out, power_spectrum_SLA_at_TG,
                    coherence_out):


    nc_out = Dataset(filename, 'w', format='NETCDF4')
    x = nc_out.createDimension('x', np.asarray(effective_resolution_out).size)
    t = nc_out.createDimension('t', np.asarray(frequency_out)[0, :].size)

    data_effective_resolution_out = nc_out.createVariable('effective_resolution', 'f8', 'x')
    data_effective_resolution_out[:] = np.asarray(effective_resolution_out)
    data_effective_resolution_out.units = 'days'
    data_effective_resolution_out.longname = 'lenghtscale where spectral coherence = 0.5'

    data_useful_resolution_out = nc_out.createVariable('useful_resolution', 'f8', 'x')
    data_useful_resolution_out[:] = np.asarray(useful_resolution_out)
    data_useful_resolution_out.units = 'days'
    data_useful_resolution_out.longname = 'lenghtscale where spectral ratio = 0.5'

    data_lat_out = nc_out.createVariable('lat', 'f8', 'x')
    data_lat_out[:] = np.asarray(lat_out)
    data_lat_out.longname = 'latitude tide gauge'

    data_lon_out = nc_out.createVariable('lon', 'f8', 'x')
    data_lon_out[:] = np.asarray(lon_out)
    data_lon_out.longname = 'longitude tide gauge'

    data_spectrum_TG_out = nc_out.createVariable('spectrum_TG', 'f8', ('x', 't'))
    data_spectrum_TG_out[:, :] = np.asarray(spectrum_TG_out)

    data_spectrum_SLA_at_TG = nc_out.createVariable('spectrum_alti', 'f8', ('x', 't'))
    data_spectrum_SLA_at_TG[:, :] = np.asarray(spectrum_SLA_at_TG)

    data_power_spectrum_TG_out = nc_out.createVariable('power_spectrum_TG', 'f8', ('x', 't'))
    data_power_spectrum_TG_out[:, :] = np.asarray(power_spectrum_TG_out)

    data_power_spectrum_SLA_at_TG = nc_out.createVariable('power_spectrum_alti', 'f8', ('x', 't'))
    data_power_spectrum_SLA_at_TG[:, :] = np.asarray(power_spectrum_SLA_at_TG)

    data_coherency_out = nc_out.createVariable('coherence', 'f8', ('x', 't'))
    data_coherency_out[:, :] = np.asarray(coherence_out)

    data_frequency_out = nc_out.createVariable('frequency', 'f8', ('x', 't'))
    data_frequency_out[:, :] = np.asarray(frequency_out)

    nc_out.close()