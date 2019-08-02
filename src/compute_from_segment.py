from netCDF4 import Dataset
from sys import argv, exit
from mod_spectral import *
from glob import glob
import matplotlib.pylab as plt
from os import path
from yaml import load
import datetime

if len(argv) != 3:
    print(' \n')
    print('USAGE   : %s <segment_files> \n' % path.basename(argv[0]))
    print('Purpose :  \n')
    exit(0)

input_file = glob(argv[1])
YAML = load(open(str(argv[2])))

equal_area = False

study_lon_min = YAML['properties']['study_area']['llcrnrlon']
study_lon_max = YAML['properties']['study_area']['urcrnrlon']
study_lat_min = YAML['properties']['study_area']['llcrnrlat']
study_lat_max = YAML['properties']['study_area']['urcrnrlat']
grid_lat = np.arange(study_lat_min, study_lat_max, YAML['outputs']['output_lat_resolution'])
grid_lon = np.arange(study_lon_min, study_lon_max, YAML['outputs']['output_lon_resolution'])
delta_lat = YAML['properties']['spectral_parameters']['delta_lat']
delta_lon = YAML['properties']['spectral_parameters']['delta_lon']

nc_file='test_psd.nc'

for filename in input_file[:2]:

    nc = Dataset(filename, 'r')
    sla_ref_segment = nc.variables['sla_segment'][:, :]
    sla_study_segment = nc.variables['sla_study_segment'][:, :]
    lon_segment = nc.variables['lon'][:]
    lat_segment = nc.variables['lat'][:]
    delta_x = nc.variables['resolution'][:]
    npt = sla_ref_segment[0, :].size

    # plt.plot(sla_ref_segment[0, :], color='b', lw=2)
    # plt.plot(sla_ref_segment.flatten()[:npt], color='r')
    # plt.show()

    #sla_ref_segment = sla_ref_segment.flatten()
    #sla_study_segment = sla_study_segment.flatten()

    print("start gridding", str(datetime.datetime.now()))
    output_effective_lon, output_effective_lat, output_mean_frequency, output_mean_PSD_sla, output_nb_segment, \
    output_mean_PSD_sla_study, output_mean_PSD_diff_sla2_sla, output_mean_coherence = \
        spectral_computation(grid_lon, grid_lat, delta_lon, delta_lat,
                             sla_ref_segment,
                             lon_segment,
                             lat_segment,
                             delta_x, npt, equal_area,
                             sla_study_segments=sla_study_segment)

    print("end gridding", str(datetime.datetime.now()))


# Write netCDF output
print("start writing x", str(datetime.datetime.now()))

write_netcdf_output(nc_file,
                        grid_lon, grid_lat, output_effective_lon, output_effective_lat,
                        output_mean_frequency, output_nb_segment, 'km',
                        output_mean_PSD_sla,
                        output_mean_frequency,
                        output_global_mean_psd_sla_study=None,
                        output_mean_psd_sla_study=output_mean_PSD_sla_study,
                        output_mean_psd_diff_sla_ref_sla_study=output_mean_PSD_diff_sla2_sla,
                        output_mean_coherence=output_mean_coherence)

print("end writing x", str(datetime.datetime.now()))


