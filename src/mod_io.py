
from netCDF4 import Dataset, date2num
import numpy as np
import datetime
from sys import exit
import xarray as xr
import glob

from mod_constant import *
from mod_geo import *
from mod_editing import *
from yaml import load, Loader


def read_grid(config, case):

    buffer_zone = np.int(0.01 * config['properties']['spectral_parameters']['lenght_scale'])
    study_lon_min = config['properties']['study_area']['llcrnrlon'] - buffer_zone
    study_lon_max = config['properties']['study_area']['urcrnrlon'] + buffer_zone
    study_lat_min = config['properties']['study_area']['llcrnrlat'] - buffer_zone
    study_lat_max = config['properties']['study_area']['urcrnrlat'] + buffer_zone
    start = datetime.datetime.strptime(str(config['properties']['time_window']['YYYYMMDD_min']), '%Y%m%d')
    end = datetime.datetime.strptime(str(config['properties']['time_window']['YYYYMMDD_max']), '%Y%m%d')
    study_time_min = int(date2num(start, units="days since 1950-01-01", calendar='standard'))
    study_time_max = int(date2num(end, units="days since 1950-01-01", calendar='standard'))
    flag_ewp = config['properties']['study_area']['flag_ewp']

    # Open either single or multi-file data set depending if list of wildcard
    if case == 'ref':
        ncfile = config['inputs']['input_file_reference']
        # Get time, lon, lat name from well known name
        lon_varname = config['inputs']['ref_field_lon_name']
        lat_varname = config['inputs']['ref_field_lat_name']
        time_varname = config['inputs']['ref_field_time_name']
        field_varname = config['inputs']['ref_field_name']

    elif case == 'study':
        ncfile = config['inputs']['input_file_study']
        # Get time, lon, lat name from well known name
        lon_varname = config['inputs']['study_field_lon_name']
        lat_varname = config['inputs']['study_field_lat_name']
        time_varname = config['inputs']['study_field_time_name']
        field_varname = config['inputs']['study_field_name']
    else:
        print("Unknown case in read_grid")
        ncfile = None
        lon_varname = None
        lat_varname = None
        time_varname = None
        field_varname = None
        exit(0)

    if "*" in ncfile or isinstance(ncfile, list):
        ds = xr.open_mfdataset(ncfile, decode_times=False)
    else:
        ds = xr.open_dataset(ncfile, decode_times=False)

    ds = ds.where((ds[lon_varname] >= study_lon_min) & (ds[lon_varname] <= study_lon_max), drop=True)
    ds = ds.where((ds[lat_varname] >= study_lat_min) & (ds[lat_varname] <= study_lat_max), drop=True)
    ds = ds.where((ds[time_varname] >= study_time_min) & (ds[time_varname] <= study_time_max), drop=True)

    time = ds[time_varname].values
    lon = ds[lon_varname].values
    lat = ds[lat_varname].values
    ssh = ds[field_varname].values

    delta_lon = np.abs(lon[0] - lon[1])

    # check dimension order
    if lon.size == np.shape(ssh)[1] and lat.size == np.shape(ssh)[2]:
        # swith time order (time, lat, lon) order
        ssh = np.swapaxes(ssh, 1, 2)

    elif lon.size == np.shape(ssh)[2] and lat.size == np.shape(ssh)[1]:
        pass

    else:
        print("Check dimension order of input file")
        exit(0)

    if flag_ewp:
        npt_x_extra = np.int(buffer_zone / delta_lon)  # add 20 degree extra point on left and right
        lx = ssh[0, 0, :].size
        ssh_east = ssh[:, :, 0:npt_x_extra]
        lon_east = lon[0:npt_x_extra]
        ssh_west = ssh[:, :, lx - npt_x_extra:]
        lon_west = lon[lx - npt_x_extra:]

        tmp = np.concatenate((lon_west - 360., lon), axis=-1)
        final_lon = np.concatenate((tmp, lon_east + 360), axis=-1)
        tmp = np.concatenate((ssh_west, ssh), axis=-1)
        final_sla = np.concatenate((tmp, ssh_east), axis=-1)

        ssh = final_sla
        lon = final_lon

    ssh = np.ma.masked_invalid(ssh)
    ssh = np.ma.masked_outside(ssh, -10, 10)

    return ssh, time, lon, lat, delta_lon


def read_along_track(config):
    """

    :param config:
    :return:
    """

    buffer_zone = np.int(0.01 * config['properties']['spectral_parameters']['lenght_scale'])
    study_lon_min = config['properties']['study_area']['llcrnrlon'] - buffer_zone
    study_lon_max = config['properties']['study_area']['urcrnrlon'] + buffer_zone
    study_lat_min = config['properties']['study_area']['llcrnrlat'] - buffer_zone
    study_lat_max = config['properties']['study_area']['urcrnrlat'] + buffer_zone
    start = datetime.datetime.strptime(str(config['properties']['time_window']['YYYYMMDD_min']), '%Y%m%d')
    end = datetime.datetime.strptime(str(config['properties']['time_window']['YYYYMMDD_max']), '%Y%m%d')
    study_time_min = int(date2num(start, units="days since 1950-01-01", calendar='standard'))
    study_time_max = int(date2num(end, units="days since 1950-01-01", calendar='standard'))

    # Open either single or multi-file data set depending if list of wildcard
    ncfile = config['inputs']['input_file_reference']
    if "*" in ncfile or isinstance(ncfile, list):
        ds = xr.open_mfdataset(ncfile, decode_times=False)
    else:
        ds = xr.open_dataset(ncfile, decode_times=False)

    # Get time, lon, lat name from well known name
    lon_varname = config['inputs']['ref_field_lon_name']
    lat_varname = config['inputs']['ref_field_lat_name']
    time_varname = config['inputs']['ref_field_time_name']
    field_varname = config['inputs']['ref_field_name']

    time = ds[time_varname].values
    lon = ds[lon_varname].values
    lat = ds[lat_varname].values

    # For Med Sea (a verifier)
    if config['properties']['study_area']['flag_roll']:
        lon = np.where(lon >= 180, lon - 360, lon)

    inds = np.where((time >= study_time_min) & (time <= study_time_max) &
                    (lon >= study_lon_min) & (lon <= study_lon_max) &
                    (lat >= study_lat_min) & (lat <= study_lat_max))[0]

    lon = lon[inds]
    lat = lat[inds]
    time = time[inds]
    field = ds[field_varname][inds]
    ref_field_scale_factor = config['inputs']['ref_field_scale_factor']
    field = ref_field_scale_factor * field

    # Edit coastal value
    if config['properties']['flag_edit_coastal']:
        print("start coastal editing", str(datetime.datetime.now()))
        field, time, lon, lat = edit_coastal_data(field, lon, lat, time,
                                                  config['properties']['file_coastal_distance'],
                                                  config['properties']['coastal_criteria'],
                                                  config['properties']['study_area']['flag_roll'])
        print("end coastal editing", str(datetime.datetime.now()))

    # Delete Dataset
    del ds

    return field, time, lon, lat


def read_residus_cls(config):
    """

    :param config:
    :return:
    """

    buffer_zone = np.int(0.01 * config['properties']['spectral_parameters']['lenght_scale'])
    study_lon_min = config['properties']['study_area']['llcrnrlon'] - buffer_zone
    study_lon_max = config['properties']['study_area']['urcrnrlon'] + buffer_zone
    study_lat_min = config['properties']['study_area']['llcrnrlat'] - buffer_zone
    study_lat_max = config['properties']['study_area']['urcrnrlat'] + buffer_zone
    start = datetime.datetime.strptime(str(config['properties']['time_window']['YYYYMMDD_min']), '%Y%m%d')
    end = datetime.datetime.strptime(str(config['properties']['time_window']['YYYYMMDD_max']), '%Y%m%d')
    study_time_min = int(date2num(start, units="days since 1950-01-01", calendar='standard'))
    study_time_max = int(date2num(end, units="days since 1950-01-01", calendar='standard'))

    lon_tmp = []
    lat_tmp = []
    time_tmp = []
    sla_tmp = []

    list_of_file = glob.glob(config['inputs']['input_file_reference'])
    for filename in sorted(list_of_file):

        ncfile = Dataset(filename, "r")
        deltat = ncfile.variables['DeltaT'][:]
        dataindexes = ncfile.variables['DataIndexes'][:]
        begindates = ncfile.variables['BeginDates'][:, 0]
        nbpoints = ncfile.variables['NbPoints'][:]
        lon_alongtrack = ncfile.variables['Longitudes'][:]
        lat_alongtrack = ncfile.variables['Latitudes'][:]
        sla_alongtrack = ncfile.variables['SLA'][:]
        ncfile.close()

        # Convert CLS time in julian days
        time_alongtrack = np.repeat(begindates, nbpoints, axis=0) + dataindexes * deltat / 86400.0

        lon_tmp = np.append(lon_tmp, lon_alongtrack)
        lat_tmp = np.append(lat_tmp, lat_alongtrack)
        time_tmp = np.append(time_tmp, time_alongtrack)
        sla_tmp = np.append(sla_tmp, sla_alongtrack)

    lon = np.asarray(lon_tmp).flatten()
    lat = np.asarray(lat_tmp).flatten()
    time = np.asarray(time_tmp).flatten()
    field = np.ma.masked_outside(np.asarray(sla_tmp).flatten(), -10., 10.)

    # For Med Sea (a verifier)
    if config['properties']['study_area']['flag_roll']:
        lon = np.where(lon >= 180, lon - 360, lon)

    inds = np.where((time >= study_time_min) & (time <= study_time_max) &
                    (lon >= study_lon_min) & (lon <= study_lon_max) &
                    (lat >= study_lat_min) & (lat <= study_lat_max))[0]

    lon = lon[inds]
    lat = lat[inds]
    time = time[inds]
    field = field[inds]
    ref_field_scale_factor = config['inputs']['ref_field_scale_factor']
    field = ref_field_scale_factor * field

    # Edit coastal value
    if config['properties']['flag_edit_coastal']:
        print("start coastal editing", str(datetime.datetime.now()))
        field, time, lon, lat = edit_coastal_data(field, lon, lat, time,
                                                  config['properties']['file_coastal_distance'],
                                                  config['properties']['coastal_criteria'],
                                                  config['properties']['study_area']['flag_roll'])
        print("end coastal editing", str(datetime.datetime.now()))

    return field, time, lon, lat


def read_table_cls(config):
    """

    :param config:
    :return:
    """

    buffer_zone = np.int(0.01 * config['properties']['spectral_parameters']['lenght_scale'])
    study_lon_min = config['properties']['study_area']['llcrnrlon'] - buffer_zone
    study_lon_max = config['properties']['study_area']['urcrnrlon'] + buffer_zone
    study_lat_min = config['properties']['study_area']['llcrnrlat'] - buffer_zone
    study_lat_max = config['properties']['study_area']['urcrnrlat'] + buffer_zone
    start = datetime.datetime.strptime(str(config['properties']['time_window']['YYYYMMDD_min']), '%Y%m%d')
    end = datetime.datetime.strptime(str(config['properties']['time_window']['YYYYMMDD_max']), '%Y%m%d')
    study_time_min = int(date2num(start, units="days since 1950-01-01", calendar='standard'))
    study_time_max = int(date2num(end, units="days since 1950-01-01", calendar='standard'))

    import octant.data.table
    from octant.date import Datetime

    table_name = config['inputs']['input_file_reference']    # "TABLE_L2P_DT_OCEAN_AL"

    table = octant.data.table.TableMeasure(table_name, mode='r', mask_default=True)

    # for field in table.fields:
    #     print("Field(s): %s" % (field))

    start = Datetime.fromjulianday(study_time_min)
    end = Datetime.fromjulianday(study_time_max)

    table_lon = table.read_values(["LONGITUDE"], start, end)
    table_lat = table.read_values(["LATITUDE"], start, end)
    table_sla = table.read_values(["SEA_LEVEL_ANOMALY"], start, end)
    table.close()

    lon = table_lon["LONGITUDE"]
    lat = table_lat["LATITUDE"]
    sla = table_sla["SEA_LEVEL_ANOMALY"]

    sla = np.ma.masked_outside(sla, -10., 10.).filled(np.nan)

    lon = np.ma.masked_array(lon, mask=np.isnan(sla)).compressed()
    lat = np.ma.masked_array(lat, mask=np.isnan(sla)).compressed()
    time = np.ma.masked_array(table_sla["time"], mask=np.isnan(sla)).compressed()
    field = np.ma.masked_array(sla, mask=np.isnan(sla)).compressed()

    for ii in range(time.size):
        time[ii] = Datetime.fromjulianday(time[ii]).julianday(dtype=float)

    # For Med Sea (a verifier)
    if config['properties']['study_area']['flag_roll']:
        lon = np.where(lon >= 180, lon - 360, lon)

    inds = np.where((lon >= study_lon_min) & (lon <= study_lon_max) &
                    (lat >= study_lat_min) & (lat <= study_lat_max))[0]

    lon = lon[inds]
    lat = lat[inds]
    time = time[inds]
    field = field[inds]
    ref_field_scale_factor = config['inputs']['ref_field_scale_factor']
    field = ref_field_scale_factor * field
    print(np.shape(lon), np.shape(lat), np.shape(time), np.shape(field))
    print(np.min(field), np.max(field))
    # Edit coastal value
    if config['properties']['flag_edit_coastal']:
        print("start coastal editing", str(datetime.datetime.now()))
        field, time, lon, lat = edit_coastal_data(field, lon, lat, time,
                                                  config['properties']['file_coastal_distance'],
                                                  config['properties']['coastal_criteria'],
                                                  config['properties']['study_area']['flag_roll'])
        print("end coastal editing", str(datetime.datetime.now()))

    print(np.shape(lon), np.shape(lat), np.shape(time), np.shape(field))

    return field, time, lon, lat


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


def read_mooring(filename):
    """

    :param filename:
    :return:
    """

    nc = Dataset(filename, 'r')
    time = nc.variables['time'][:]
    sla_alti = nc.variables['ssh_alti'][:]
    sla_mooring = nc.variables['ssh_sensor'][:]
    lat = nc.variables['lat'][0]
    lon = nc.variables['lon'][0]
    nc.close()

    return sla_mooring, sla_alti, time, lat, lon


def read_tao(filename):
    """

    :param filename:
    :return:
    """

    nc = Dataset(filename, 'r')
    ssh_tao = nc.variables['DYN_13'][:, 0, 0, 0] / 100
    lon_tao = nc.variables['lon'][:]
    lat_tao = nc.variables['lat'][:]
    time_tao = nc.variables['time'][:] - 2433660 + 365 + 12
    nc.close()

    return ssh_tao, time_tao, lat_tao, lon_tao


def read_cls_format(fcid):
    """
    Read sea level anomaly, lon and lat from CLS maps
    :param fcid:
    :return:
    """
    sla_map = np.array(fcid.variables['Grid_0001'][:, :]).transpose()  # / 100.  # convert in meters
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
    yaml = load(open(str(mission_management)), Loader=Loader)
    velocity = yaml[cmission]['VELOCITY']
    if velocity is not None:
        return velocity
    else:
        raise ValueError("velocity not found for mission %s in %s" % (cmission, mission_management))


def get_deltat(cmission, mission_management):
    """
    Get deltaT of a mission from file MissionManagement.yaml
    """
    yaml = load(open(str(mission_management)), Loader=Loader)
    deltat = yaml[cmission]['DELTA_T']
    if deltat is not None:
        return deltat
    else:
        raise ValueError("deltat not found for mission %s in %s" % (cmission, mission_management))


def write_segment(config, lat_segment, lon_segment, sla_segment, resolution, resolution_units, npt,
                  segment_study=None, direction=None):
    """

    :param config:
    :param lat_segment:
    :param lon_segment:
    :param sla_segment:
    :param resolution:
    :param resolution_units:
    :param npt:
    :param segment_study:
    :param direction:
    :return:
    """

    if direction == 'zonal':
        output = config['outputs']['output_segment_filename_x_direction']
    elif direction == 'meridional':
        output = config['outputs']['output_segment_filename_y_direction']
    else:
        output = config['outputs']['output_segment_filename']

    nc_out = Dataset(output, 'w', format='NETCDF4')
    nb_segment = np.shape(np.asarray(sla_segment))[0]
    
    nc_out.createDimension('segment_size', npt)
    nc_out.createDimension('nb_segment', nb_segment)
    nc_out.createDimension('resolution', 1)

    lon_out = nc_out.createVariable('lon', 'f8', 'nb_segment', zlib=True)
    lon_out[:] = np.asarray(lon_segment)
    lon_out.longname = 'longitude segment center'

    lat_out = nc_out.createVariable('lat', 'f8', 'nb_segment', zlib=True)
    lat_out[:] = np.asarray(lat_segment)
    lat_out.longname = 'latitude segment center'

    segment_out = nc_out.createVariable('sla_segment', 'f8', ('nb_segment', 'segment_size'), zlib=True)
    
    segment_out[:, :] = np.asarray(sla_segment)
    segment_out.longname = 'array of segments'

    if segment_study is not None:
        segment_study_out = nc_out.createVariable('sla_study_segment', 'f8', ('nb_segment', 'segment_size'), zlib=True)
        segment_study_out[:, :] = np.asarray(segment_study)
        segment_study_out.longname = 'array of study segments'

    segment_resolution = nc_out.createVariable('resolution', 'f8', 'resolution', zlib=True)
    segment_resolution[:] = resolution
    segment_resolution.longname = 'resolution of segments'
    segment_resolution.units = resolution_units

    nc_out.close()


def write_netcdf_output(config, wavenumber, nb_segment, freq_unit, psd_ref, global_psd_ref,
                        global_psd_study=None,
                        psd_study=None,
                        psd_diff_ref_study=None,
                        coherence=None,
                        cross_spectrum=None,
                        direction=None):
    """

    :param config:
    :param wavenumber:
    :param nb_segment:
    :param freq_unit:
    :param psd_ref:
    :param global_psd_ref:
    :param global_psd_study:
    :param psd_study:
    :param psd_diff_ref_study:
    :param coherence:
    :param cross_spectrum:
    :param direction:
    :return:
    """

    study_lon_min = config['properties']['study_area']['llcrnrlon']
    study_lon_max = config['properties']['study_area']['urcrnrlon']
    study_lat_min = config['properties']['study_area']['llcrnrlat']
    study_lat_max = config['properties']['study_area']['urcrnrlat']
    lat = np.arange(study_lat_min, study_lat_max, config['outputs']['output_lat_resolution'])
    lon = np.arange(study_lon_min, study_lon_max, config['outputs']['output_lon_resolution'])
    if direction == 'zonal':
        output_netcdf_file = config['outputs']['output_filename_x_direction']
    elif direction == 'meridional':
        output_netcdf_file = config['outputs']['output_filename_y_direction']
    else:
        output_netcdf_file = config['outputs']['output_filename']

    nc_out = Dataset(output_netcdf_file, 'w', format='NETCDF4')
    fsize = np.shape(np.asarray(wavenumber))[1]
    nc_out.createDimension('wavenumber', fsize)
    nc_out.createDimension('lat', lat.size)
    nc_out.createDimension('lon', lon.size)

    frequence_out = nc_out.createVariable('wavenumber', 'f8', 'wavenumber', zlib=True)
    frequence_out.units = "1/%s" % freq_unit
    frequence_out.axis = 'T'
    freq = np.ma.mean(np.ma.masked_invalid(np.ma.masked_where(np.asarray(wavenumber) == 0,
                                                              np.asarray(wavenumber))), axis=0).filled(0.)

    frequence_out[:] = freq

    data = np.asarray(nb_segment).reshape((lat.size, lon.size))
    nb_segment_out = nc_out.createVariable('nb_segment', 'f8', ('lat', 'lon'), zlib=True)
    nb_segment_out.long_name = "number of segment used in spectral computation"
    nb_segment_out[:, :] = np.ma.masked_where(data == 0., data)

    lat_out = nc_out.createVariable('lat', 'f8', 'lat')
    lat_out[:] = lat
    lon_out = nc_out.createVariable('lon', 'f8', 'lon')
    lon_out[:] = lon

    data = np.transpose(np.asarray(psd_ref)).reshape((fsize, lat.size, lon.size))
    psd_ref = nc_out.createVariable('psd_ref', 'f8', ('wavenumber', 'lat', 'lon'), zlib=True)
    psd_ref.units = 'm2/%s' % freq_unit
    psd_ref.coordinates = "freq lat lon"
    psd_ref.long_name = "power spectrum density reference field"
    psd_ref[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(data == 0, data))

    global_psd_ref_out = nc_out.createVariable('global_mean_psd_ref', 'f8', 'wavenumber', zlib=True)
    global_psd_ref_out.units = 'm2/%s' % freq_unit
    global_psd_ref_out[:] = np.ma.masked_invalid(global_psd_ref)
    global_psd_ref_out.long_name = "global power spectrum density reference field"

    if global_psd_study is not None:
        global_psd_study_out = nc_out.createVariable('global_mean_psd_study', 'f8', 'wavenumber', zlib=True)
        global_psd_study_out.units = 'm2/%s' % freq_unit
        global_psd_study_out[:] = np.ma.masked_invalid(global_psd_study)
        global_psd_study_out.long_name = "global power spectrum density study field"

    if psd_study is not None:
        data = np.transpose(np.asarray(psd_study)).reshape((fsize, lat.size, lon.size))
        psd_study = nc_out.createVariable('psd_study', 'f8', ('wavenumber', 'lat', 'lon'), zlib=True)
        psd_study.units = 'm2/%s' % freq_unit
        psd_study.coordinates = "freq lat lon"
        psd_study.long_name = "power spectrum density study field"
        psd_study[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(data == 0, data))

    if psd_diff_ref_study is not None:
        data = np.transpose(np.asarray(psd_diff_ref_study)).reshape((fsize, lat.size, lon.size))
        psd_diff = nc_out.createVariable('psd_diff', 'f8', ('wavenumber', 'lat', 'lon'), zlib=True)
        psd_diff.units = 'm2/%s' % freq_unit
        psd_diff.coordinates = "freq lat lon"
        psd_diff.long_name = "power spectrum density of difference study minus reference field"
        psd_diff[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(data == 0, data))

    if coherence is not None:
        data = np.transpose(np.asarray(coherence)).reshape((fsize, lat.size, lon.size))
        coherence_out = nc_out.createVariable('coherence', 'f8', ('wavenumber', 'lat', 'lon'), zlib=True)
        coherence_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(data == 0, data))
        coherence_out.coordinates = "freq lat lon"
        coherence_out.long_name = "magnitude squared coherence between reference and study fields"

    if cross_spectrum is not None:
        data = np.transpose(np.asarray(cross_spectrum)).reshape((fsize, lat.size, lon.size))
        cross_spectrum_real_out = nc_out.createVariable('cross_spectrum_real', 'f8', ('wavenumber', 'lat', 'lon'),
                                                        zlib=True)
        cross_spectrum_real_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(np.real(data) == 0., np.real(data)))
        cross_spectrum_real_out.coordinates = "freq lat lon"
        cross_spectrum_real_out.long_name = "real part of cross_spectrum between reference and study fields"
        cross_spectrum_imag_out = nc_out.createVariable('cross_spectrum_imag', 'f8', ('wavenumber', 'lat', 'lon'),
                                                        zlib=True)
        cross_spectrum_imag_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(np.imag(data) == 0., np.imag(data)))
        cross_spectrum_imag_out.coordinates = "freq lat lon"
        cross_spectrum_imag_out.long_name = "imaginary part of cross_spectrum between reference and study fields"

    nc_out.close()


def write_netcdf_tide_tao(filename, wavenumber, lat, lon, psd_tg, psd_study, psd_diff_tg_study, coherence,
                          cross_spectrum):
    """

    :param filename:
    :param wavenumber:
    :param lat:
    :param lon:
    :param psd_tg:
    :param psd_study:
    :param psd_diff_tg_study:
    :param coherence:
    :param cross_spectrum:
    :return:
    """

    # sort sensor by latitude
    lat_sorted_index = np.argsort(lat)

    nc_out = Dataset(filename, 'w', format='NETCDF4')
    nc_out.createDimension('wavenumber', np.asarray(wavenumber)[1, :].size)
    nc_out.createDimension('sensor', np.asarray(lat).size)

    wavenumber_out = nc_out.createVariable('wavenumber', 'f8', 'wavenumber', zlib=True)
    wavenumber_out[:] = np.asarray(wavenumber)[1, :]

    lat_out = nc_out.createVariable('lat', 'f8', 'sensor', zlib=True)
    lat_out[:] = np.asarray(lat)[lat_sorted_index]
    lat_out.longname = 'latitude sensor'

    lon_out = nc_out.createVariable('lon', 'f8', 'sensor', zlib=True)
    lon_out[:] = np.asarray(lon)[lat_sorted_index]
    lon_out.longname = 'longitude sensor'

    psd_tg_out = nc_out.createVariable('psd_ref', 'f8', ('sensor', 'wavenumber'), zlib=True)
    psd_tg_out[:, :] = np.asarray(psd_tg)[lat_sorted_index, :]
    psd_tg_out.coordinates = "sensor freq"
    psd_tg_out.long_name = "power spectrum density ref field"

    psd_study_out = nc_out.createVariable('psd_study', 'f8', ('sensor', 'wavenumber'), zlib=True)
    psd_study_out[:, :] = np.asarray(psd_study)[lat_sorted_index, :]
    psd_study_out.coordinates = "sensor freq"
    psd_study_out.long_name = "power spectrum density study field"
    
    psd_diff_study_tg_out = nc_out.createVariable('psd_diff', 'f8', ('sensor', 'wavenumber'), zlib=True)
    psd_diff_study_tg_out[:, :] = np.asarray(psd_diff_tg_study)[lat_sorted_index, :]
    psd_diff_study_tg_out.coordinates = "sensor freq"
    psd_diff_study_tg_out.long_name = "power spectrum density of difference study minus reference field"

    coherence_out = nc_out.createVariable('coherence', 'f8', ('sensor', 'wavenumber'), zlib=True)
    coherence_out[:, :] = np.asarray(coherence)[lat_sorted_index, :]
    coherence_out.coordinates = "sensor freq"
    coherence_out.long_name = "magnitude squared coherence between reference and study fields"

    cross_spectrum_real_out = nc_out.createVariable('cross_spectrum_real', 'f8', ('sensor', 'wavenumber'), zlib=True)
    cross_spectrum_real_out[:, :] = np.real(np.asarray(cross_spectrum))[lat_sorted_index, :]
    cross_spectrum_real_out.coordinates = "freq lat lon"
    cross_spectrum_real_out.long_name = "real part cross_spectrum between reference and study fields"

    cross_spectrum_imag_out = nc_out.createVariable('cross_spectrum_imag', 'f8', ('sensor', 'wavenumber'), zlib=True)
    cross_spectrum_imag_out[:, :] = np.imag(np.asarray(cross_spectrum))[lat_sorted_index, :]
    cross_spectrum_imag_out.coordinates = "freq lat lon"
    cross_spectrum_imag_out.long_name = "imaginary part cross_spectrum between reference and study fields"

    nc_out.close()


def write_netcdf_temporal_output(config, wavenumber, lat, lon, psd_ref, psd_study=None, psd_diff_ref_study=None,
                                 coherence=None, cross_spectrum=None):
    """

    :param config:
    :param wavenumber:
    :param lat:
    :param lon:
    :param psd_ref:
    :param psd_study:
    :param psd_diff_ref_study:
    :param coherence:
    :param cross_spectrum:
    :return:
    """

    output_netcdf_file = config['outputs']['output_filename_t_direction']
    freq_unit = 'days'

    nc_out = Dataset(output_netcdf_file, 'w', format='NETCDF4')
    fsize = np.shape(np.asarray(wavenumber))[0]
    nc_out.createDimension('wavenumber', fsize)
    nc_out.createDimension('lat', lat.size)
    nc_out.createDimension('lon', lon.size)

    frequence_out = nc_out.createVariable('wavenumber', 'f8', 'wavenumber', zlib=True)
    frequence_out.units = "1/%s" % freq_unit
    frequence_out.axis = 'T'
    freq = np.ma.masked_invalid(wavenumber)

    frequence_out[:] = freq

    lat_out = nc_out.createVariable('lat', 'f8', 'lat', zlib=True)
    lat_out[:] = lat
    lon_out = nc_out.createVariable('lon', 'f8', 'lon', zlib=True)
    lon_out[:] = lon

    data = np.transpose(np.asarray(psd_ref)).reshape((fsize, lat.size, lon.size))
    psd_ref = nc_out.createVariable('psd_ref', 'f8', ('wavenumber', 'lat', 'lon'), zlib=True)
    psd_ref.units = 'm2/%s' % freq_unit
    psd_ref.coordinates = "freq lat lon"
    psd_ref.long_name = "power spectrum density reference field"
    psd_ref[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(data == 0, data))

    if psd_study is not None:
        data = np.transpose(np.asarray(psd_study)).reshape((fsize, lat.size, lon.size))
        psd_study = nc_out.createVariable('psd_study', 'f8', ('wavenumber', 'lat', 'lon'), zlib=True)
        psd_study.units = 'm2/%s' % freq_unit
        psd_study.coordinates = "freq lat lon"
        psd_study.long_name = "power spectrum density study field"
        psd_study[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(data == 0, data))

    if psd_diff_ref_study is not None:
        data = np.transpose(np.asarray(psd_diff_ref_study)).reshape((fsize, lat.size, lon.size))
        psd_diff = nc_out.createVariable('psd_diff', 'f8', ('wavenumber', 'lat', 'lon'), zlib=True)
        psd_diff.units = 'm2/%s' % freq_unit
        psd_diff.coordinates = "freq lat lon"
        psd_diff.long_name = "power spectrum density of difference study minus reference field"
        psd_diff[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(data == 0, data))

    if coherence is not None:
        data = np.transpose(np.asarray(coherence)).reshape((fsize, lat.size, lon.size))
        coherence_out = nc_out.createVariable('coherence', 'f8', ('wavenumber', 'lat', 'lon'), zlib=True)
        coherence_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(data == 0, data))
        coherence_out.coordinates = "freq lat lon"
        coherence_out.long_name = "magnitude squared coherence between reference and study fields"

    if cross_spectrum is not None:
        data = np.transpose(np.asarray(cross_spectrum)).reshape((fsize, lat.size, lon.size))
        cross_spectrum_real_out = nc_out.createVariable('cross_spectrum_real', 'f8', ('wavenumber', 'lat', 'lon'),
                                                        zlib=True)
        cross_spectrum_real_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(np.real(data) == 0., np.real(data)))
        cross_spectrum_real_out.coordinates = "freq lat lon"
        cross_spectrum_real_out.long_name = "real part of cross_spectrum between reference and study fields"
        cross_spectrum_imag_out = nc_out.createVariable('cross_spectrum_imag', 'f8', ('wavenumber', 'lat', 'lon'),
                                                        zlib=True)
        cross_spectrum_imag_out[:, :, :] = np.ma.masked_invalid(np.ma.masked_where(np.imag(data) == 0., np.imag(data)))
        cross_spectrum_imag_out.coordinates = "freq lat lon"
        cross_spectrum_imag_out.long_name = "imaginary part of cross_spectrum between reference and study fields"

    nc_out.close()


def write_netcdf_stat_output(config, nobs, min, max, mean, variance, skewness, kurtosis, rmse, mae, correlation, pvalue,
                             variance_ref, variance_study, mean_ref, mean_study):
    """

    """

    study_lon_min = config['properties']['study_area']['llcrnrlon']
    study_lon_max = config['properties']['study_area']['urcrnrlon']
    study_lat_min = config['properties']['study_area']['llcrnrlat']
    study_lat_max = config['properties']['study_area']['urcrnrlat']
    lat = np.arange(study_lat_min, study_lat_max, config['outputs']['output_lat_resolution'])
    lon = np.arange(study_lon_min, study_lon_max, config['outputs']['output_lon_resolution'])
    output_netcdf_file = config['outputs']['output_filename']

    nc_out = Dataset(output_netcdf_file, 'w', format='NETCDF4')
    nc_out.createDimension('lat', lat.size)
    nc_out.createDimension('lon', lon.size)

    lat_out = nc_out.createVariable('lat', 'f8', 'lat', zlib=True)
    lat_out[:] = lat
    lon_out = nc_out.createVariable('lon', 'f8', 'lon', zlib=True)
    lon_out[:] = lon

    nobs_out = nc_out.createVariable('nobs', 'i4', ('lat', 'lon'), zlib=True)
    nobs_out.long_name = "number of observation"
    nobs_out[:, :] = np.ma.masked_where(nobs == 0, nobs)

    min_out = nc_out.createVariable('min', 'f8', ('lat', 'lon'), zlib=True)
    min_out.long_name = "minimum value"
    min_out[:, :] = np.ma.masked_where(min == 0., min)

    max_out = nc_out.createVariable('max', 'f8', ('lat', 'lon'), zlib=True)
    max_out.long_name = "maximum value"
    max_out[:, :] = np.ma.masked_where(max == 0., max)

    mean_out = nc_out.createVariable('mean', 'f8', ('lat', 'lon'), zlib=True)
    mean_out.long_name = "mean value"
    mean_out[:, :] = np.ma.masked_where(mean == 0., mean)

    variance_out = nc_out.createVariable('variance', 'f8', ('lat', 'lon'), zlib=True)
    variance_out.long_name = "variance value"
    variance_out[:, :] = np.ma.masked_where(variance == 0., variance)

    skewness_out = nc_out.createVariable('skewness', 'f8', ('lat', 'lon'), zlib=True)
    skewness_out.long_name = "skewness value"
    skewness_out[:, :] = np.ma.masked_where(skewness == 0., skewness)

    kurtosis_out = nc_out.createVariable('kurtosis', 'f8', ('lat', 'lon'), zlib=True)
    kurtosis_out.long_name = "kurtosis value"
    kurtosis_out[:, :] = np.ma.masked_where(kurtosis == 0., kurtosis)

    rmse_out = nc_out.createVariable('rmse', 'f8', ('lat', 'lon'), zlib=True)
    rmse_out.long_name = "mean square error value"
    rmse_out[:, :] = np.ma.masked_where(rmse == 0., rmse)

    mae_out = nc_out.createVariable('mae', 'f8', ('lat', 'lon'), zlib=True)
    mae_out.long_name = "mean absolute error value"
    mae_out[:, :] = np.ma.masked_where(mae == 0., mae)

    correlation_out = nc_out.createVariable('correlation', 'f8', ('lat', 'lon'), zlib=True)
    correlation_out.long_name = "correlation value"
    correlation_out[:, :] = np.ma.masked_where(correlation == 0., correlation)

    pvalue_out = nc_out.createVariable('pvalue', 'f8', ('lat', 'lon'), zlib=True)
    pvalue_out.long_name = "pvalue correlation"
    pvalue_out[:, :] = np.ma.masked_where(pvalue == 0., pvalue)

    variance_ref_out = nc_out.createVariable('variance_ref', 'f8', ('lat', 'lon'), zlib=True)
    variance_ref_out.long_name = "variance reference field"
    variance_ref_out[:, :] = np.ma.masked_where(variance_ref == 0., variance_ref)

    variance_study_out = nc_out.createVariable('variance_study', 'f8', ('lat', 'lon'), zlib=True)
    variance_study_out.long_name = "variance value"
    variance_study_out[:, :] = np.ma.masked_where(variance_study == 0., variance_study)

    mean_ref_out = nc_out.createVariable('mean_ref', 'f8', ('lat', 'lon'), zlib=True)
    mean_ref_out.long_name = "mean reference field"
    mean_ref_out[:, :] = np.ma.masked_where(mean_ref == 0., mean_ref)

    mean_study_out = nc_out.createVariable('mean_study', 'f8', ('lat', 'lon'), zlib=True)
    mean_study_out.long_name = "mean study field"
    mean_study_out[:, :] = np.ma.masked_where(mean_study == 0., mean_study)
    nc_out.close()


def write_netcdf_timesries_stat_output(config, nobs, min, max, mean, variance, skewness, kurtosis, rmse, mae,
                                       correlation, pvalue, timeline, variance_ref, variance_study,
                                       mean_ref, mean_study):
    """

    """

    output_netcdf_file = config['outputs']['output_timeseries_filename']

    nc_out = Dataset(output_netcdf_file, 'w', format='NETCDF4')
    nc_out.createDimension('time', timeline.size)

    time_out = nc_out.createVariable('time', 'f8', 'time')
    time_out[:] = timeline
    time_out.units= 'days since 1950-01-01'

    nobs_out = nc_out.createVariable('nobs', 'i4', 'time')
    nobs_out.long_name = "number of observation"
    nobs_out[:] = np.ma.masked_where(nobs == 0, nobs)

    min_out = nc_out.createVariable('min', 'f8', 'time')
    min_out.long_name = "minimum value"
    min_out[:] = np.ma.masked_where(min == 0., min)

    max_out = nc_out.createVariable('max', 'f8', 'time')
    max_out.long_name = "maximum value"
    max_out[:] = np.ma.masked_where(max == 0., max)

    mean_out = nc_out.createVariable('mean', 'f8', 'time')
    mean_out.long_name = "mean value"
    mean_out[:] = np.ma.masked_where(mean == 0., mean)

    variance_out = nc_out.createVariable('variance', 'f8', 'time')
    variance_out.long_name = "variance value"
    variance_out[:] = np.ma.masked_where(variance == 0., variance)

    skewness_out = nc_out.createVariable('skewness', 'f8', 'time')
    skewness_out.long_name = "skewness value"
    skewness_out[:] = np.ma.masked_where(skewness == 0., skewness)

    kurtosis_out = nc_out.createVariable('kurtosis', 'f8', 'time')
    kurtosis_out.long_name = "kurtosis value"
    kurtosis_out[:] = np.ma.masked_where(kurtosis == 0., kurtosis)

    rmse_out = nc_out.createVariable('rmse', 'f8', 'time')
    rmse_out.long_name = "mean square error value"
    rmse_out[:] = np.ma.masked_where(rmse == 0., rmse)

    mae_out = nc_out.createVariable('mae', 'f8', 'time')
    mae_out.long_name = "mean absolute error value"
    mae_out[:] = np.ma.masked_where(mae == 0., mae)

    correlation_out = nc_out.createVariable('correlation', 'f8', 'time')
    correlation_out.long_name = "correlation value"
    correlation_out[:] = np.ma.masked_where(correlation == 0., correlation)

    pvalue_out = nc_out.createVariable('pvalue', 'f8', 'time')
    pvalue_out.long_name = "pvalue correlation"
    pvalue_out[:] = np.ma.masked_where(pvalue == 0., pvalue)

    mean_ref_out = nc_out.createVariable('mean_ref', 'f8', 'time')
    mean_ref_out.long_name = "mean ref value"
    mean_ref_out[:] = np.ma.masked_where(mean_ref == 0., mean_ref)

    variance_ref_out = nc_out.createVariable('variance_ref', 'f8', 'time')
    variance_ref_out.long_name = "variance ref value"
    variance_ref_out[:] = np.ma.masked_where(variance_ref == 0., variance_ref)

    mean_study_out = nc_out.createVariable('mean_study', 'f8', 'time')
    mean_study_out.long_name = "mean study value"
    mean_study_out[:] = np.ma.masked_where(mean_study == 0., mean_study)

    variance_study_out = nc_out.createVariable('variance_study', 'f8', 'time')
    variance_study_out.long_name = "variance study value"
    variance_study_out[:] = np.ma.masked_where(variance_study == 0., variance_study)

    nc_out.close()
