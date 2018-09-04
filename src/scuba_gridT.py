from netCDF4 import date2num
from sys import argv, exit
from os import path
import datetime

from mod_io import *
from mod_constant import *
from mod_geo import *
from mod_segmentation import *
from mod_spectral import *
from yaml import load

if len(argv) != 2:
    print(' \n')
    print('USAGE   : %s <file.yaml> \n' % path.basename(argv[0]))
    print('Purpose : Spectral and spectral coherence computation tools for altimeter data \n')
    exit(0)

# Load analysis information from YAML file
YAML = load(open(str(argv[1])))
input_file_reference = YAML['inputs']['input_file_reference']
ref_lon_name = YAML['inputs']['ref_lon_name']
ref_lat_name = YAML['inputs']['ref_lat_name']
ref_field_name = YAML['inputs']['ref_field_name']
input_file_study = YAML['inputs']['input_file_study']
study_lon_name = YAML['inputs']['study_lon_name']
study_lat_name = YAML['inputs']['study_lat_name']
study_field_name = YAML['inputs']['study_field_name']

# input_map_directory = YAML['inputs']['input_map_directory']
# map_file_pattern = YAML['inputs']['map_file_pattern']

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

nc_file_x = YAML['outputs']['output_filename_x_direction']
nc_file_y = YAML['outputs']['output_filename_y_direction']

# lon_ref_map_study, lat_ref_map_study, sla_ref_map_study, delta_lon_in = read_mdt(input_file_reference,
#                                                                                   study_lon_min, study_lon_max,
#                                                                                   study_lat_min, study_lat_max,
#                                                                                   flag_ewp)

# lon_ref_map_study, lat_ref_map_study, sla_ref_map_study, delta_lon_in = read_natl60(input_file_reference,
#                                                                                   study_lon_min, study_lon_max,
#                                                                                   study_lat_min, study_lat_max,
#                                                                                   flag_ewp)


lon_ref_map_study, lat_ref_map_study, time_ref_map, sla_ref_map_study, delta_lon_in = read_grid_fieldT(input_file_reference,
                                                                                        ref_lon_name, ref_lat_name,
                                                                                        ref_field_name,
                                                                                        study_lon_min, study_lon_max,
                                                                                        study_lat_min, study_lat_max,
                                                                                        flag_ewp)

if flag_reference_only:

    time_sample = (time_ref_map - time_ref_map[0])/(3600*24)
    delta_time = np.diff(time_sample)[0]
    freq_min = 1./time_sample.size
    freq_max = 1./delta_time
    output_freq_sample = np.linspace(freq_min, freq_max, time_sample.size)

    sla_sample = sla_ref_map_study
    # compute temporal spectrum on grid
    print("start gridding", str(datetime.datetime.now()))
    lx = sla_sample[0, 0, :].size
    ly = sla_sample[0, :, 0].size

    # Initialisation
    pgram = np.zeros((output_freq_sample.size, ly, lx))

    for jj in range(ly):
        for ii in range(lx):
            pgram[:, jj, ii] = compute_lombscargle_periodogram(np.float64(time_sample),
                                                    np.float64(sla_sample[:, jj, ii]),
                                                    np.float64(output_freq_sample))


    print("end gridding", str(datetime.datetime.now()))

    # Write netCDF output
    print("start writing", str(datetime.datetime.now()))

    from netCDF4 import Dataset
    nc_out = Dataset("test.nc", 'w', format='NETCDF4')
    nc_out.createDimension('f', output_freq_sample.size)
    nc_out.createDimension('y', lat_ref_map_study.size)
    nc_out.createDimension('x', lon_ref_map_study.size)

    frequence_out = nc_out.createVariable('frequency', 'f8', 'f')
    frequence_out.axis = 'T'
    frequence_out[:] = 1/output_freq_sample

    lat_out = nc_out.createVariable('lat', 'f8', 'y')
    lat_out[:] = lat_ref_map_study

    lon_out = nc_out.createVariable('lon', 'f8', 'x')
    lon_out[:] = lon_ref_map_study

    pgram_out = nc_out.createVariable('periodogram', 'f8', ('f', 'y', 'x'))
    pgram_out[:, :, :] = pgram
    pgram_out.coordinates = 'frequency lat lon'


    nc_out.close()


    print("end writing", str(datetime.datetime.now()))

else:

    lon_study_map_study, lat_study_map_study, sla_study_map_study, delta_lon_in = read_grid_field(input_file_reference,
                                                                                                  study_lon_name,
                                                                                                  study_lat_name,
                                                                                                  study_field_name,
                                                                                                  study_lon_min,
                                                                                                  study_lon_max,
                                                                                                  study_lat_min,
                                                                                                  study_lat_max,
                                                                                                  flag_ewp)

    # TEST TO BE REMOVED
    # sla_study_map_study = np.copy(sla_ref_map_study)
    sla_study_map_study = np.roll(sla_study_map_study, 5, axis=0)

    print("start segment computation", str(datetime.datetime.now()))

    computed_lat_segment_x, computed_lon_segment_x, computed_sla_ref_segment_x, delta_x, npt_x, \
    computed_lat_segment_y, computed_lon_segment_y, computed_sla_ref_segment_y, delta_y, npt_y, \
    computed_sla_study_segment_x, computed_sla_study_segment_y = \
        compute_segment_xy_maps(
            sla_ref_map_study, lon_ref_map_study, lat_ref_map_study, lenght_scale, sla_study_map_study)

    print("end segment computation", str(datetime.datetime.now()))

    # compute zonal spectrum on grid
    print("start gridding x", str(datetime.datetime.now()))

    output_effective_lon, output_effective_lat, output_mean_frequency, \
    output_mean_PSD_sla, output_mean_PS_sla, output_nb_segment, \
    output_autocorrelation_ref, output_autocorrelation_ref_zero_crossing, output_autocorrelation_distance, \
    output_mean_PS_sla2, output_mean_PS_diff_sla2_sla, \
    output_mean_PSD_sla2, output_mean_PSD_diff_sla2_sla,\
    output_mean_coherence, output_effective_resolution, output_useful_resolution, \
    output_autocorrelation_study, output_autocorrelation_study_zero_crossing = \
        spectral_computation(grid_lon, grid_lat, delta_lon, delta_lat,
                             np.asarray(computed_sla_ref_segment_x),
                             np.asarray(computed_lon_segment_x),
                             np.asarray(computed_lat_segment_x),
                             delta_lon_in, npt_x, equal_area, flag_greenwich_start, np.asarray(computed_sla_study_segment_x))

    print("end gridding x", str(datetime.datetime.now()))

    # Write netCDF output
    print("start writing x", str(datetime.datetime.now()))
    write_netcdf_output(nc_file_x,
                        grid_lon, grid_lat, output_effective_lon, output_effective_lat,
                        output_mean_frequency, output_nb_segment, 'degree',
                        output_mean_PS_sla, output_mean_PSD_sla,
                        output_autocorrelation_distance,
                        output_autocorrelation_ref, output_autocorrelation_ref_zero_crossing,
                        output_mean_ps_sla_study=output_mean_PS_sla2, output_mean_psd_sla_study=output_mean_PSD_sla2,
                        output_autocorrelation_study=output_autocorrelation_study,
                        output_autocorrelation_study_zero_crossing=output_autocorrelation_study_zero_crossing,
                        output_mean_ps_diff_sla_ref_sla_study=output_mean_PS_diff_sla2_sla,
                        output_mean_psd_diff_sla_ref_sla_study=output_mean_PSD_diff_sla2_sla,
                        output_mean_coherence=output_mean_coherence,
                        output_effective_resolution=output_effective_resolution,
                        output_useful_resolution=output_useful_resolution)

    print("end writing x", str(datetime.datetime.now()))

    # compute meridional spectrum on grid
    print("start gridding y", str(datetime.datetime.now()))

    output_effective_lon, output_effective_lat, output_mean_frequency, \
    output_mean_PSD_sla, output_mean_PS_sla, output_nb_segment, \
    output_autocorrelation_ref, output_autocorrelation_ref_zero_crossing, output_autocorrelation_distance, \
    output_mean_PS_sla2, output_mean_PS_diff_sla2_sla, \
    output_mean_PSD_sla2, output_mean_PSD_diff_sla2_sla,\
    output_mean_coherence, output_effective_resolution, output_useful_resolution, \
    output_autocorrelation_study, output_autocorrelation_study_zero_crossing = \
        spectral_computation(grid_lon, grid_lat, delta_lon, delta_lat,
                             np.asarray(computed_sla_ref_segment_y),
                             np.asarray(computed_lon_segment_y),
                             np.asarray(computed_lat_segment_y),
                             delta_y, npt_y, equal_area, flag_greenwich_start, np.asarray(computed_sla_study_segment_y))

    print("end gridding y", str(datetime.datetime.now()))

    # Write netCDF output
    print("start writing y", str(datetime.datetime.now()))

    write_netcdf_output(nc_file_y,
                        grid_lon, grid_lat, output_effective_lon, output_effective_lat,
                        output_mean_frequency, output_nb_segment, 'km',
                        output_mean_PS_sla, output_mean_PSD_sla,
                        output_autocorrelation_distance,
                        output_autocorrelation_ref, output_autocorrelation_ref_zero_crossing,
                        output_mean_ps_sla_study=output_mean_PS_sla2, output_mean_psd_sla_study=output_mean_PSD_sla2,
                        output_autocorrelation_study=output_autocorrelation_study,
                        output_autocorrelation_study_zero_crossing=output_autocorrelation_study_zero_crossing,
                        output_mean_ps_diff_sla_ref_sla_study=output_mean_PS_diff_sla2_sla,
                        output_mean_psd_diff_sla_ref_sla_study=output_mean_PSD_diff_sla2_sla,
                        output_mean_coherence=output_mean_coherence,
                        output_effective_resolution=output_effective_resolution,
                        output_useful_resolution=output_useful_resolution)

    print("end writing y", str(datetime.datetime.now()))