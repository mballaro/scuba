from netCDF4 import date2num
from sys import argv, exit
from os import path
import datetime
import scipy.signal
import matplotlib.pylab as plt
import logging
import argparse

from mod_io import *
from mod_constant import *
from mod_geo import *
from mod_segmentation import *
from mod_spectral import *
from yaml import load, Loader


parser = argparse.ArgumentParser(description='Spectral computation tools for altimeter data')
parser.add_argument("config_file", help="configuration file for alongtrack analysis (.yaml format)")
parser.add_argument('-v', '--verbose', dest='verbose', default='error',
                    help='Verbosity : critical, error, warning, info, debug')
args = parser.parse_args()

config_file = args.config_file

FORMAT_LOG = "%(levelname)-10s %(asctime)s %(module)s.%(funcName)s : %(message)s"
logging.basicConfig(format=FORMAT_LOG, level=logging.ERROR, datefmt="%H:%M:%S")
logger = logging.getLogger()
logger.setLevel(getattr(logging, args.verbose.upper()))

YAML = load(open(str(config_file)), Loader=Loader)

direction = YAML['properties']['spectral_parameters']['direction']
flag_reference_only = YAML['properties']['spectral_parameters']['flag_reference_only']


# Read reference input
logging.info("start reading reference grid")
ssh_ref, time, lon, lat, delta_lon_in = read_grid(YAML, 'ref')
logging.info("end reading reference grid")

ssh_ref = ssh_ref * YAML['inputs']['ref_field_scale_factor']


if flag_reference_only:

    if direction == "zonal" or direction == "both":

        # compute segment
        logging.info("start segment computation x-direction")

        ref_segment, lon_segment, lat_segment, delta_x, npt = compute_segment_maps(ssh_ref, lon, lat, "zonal", YAML)

        # write segment
        # write_segment(YAML, lat_segment, lon_segment, ref_segment, delta_x, 'km', npt, direction="zonal")

        logging.info("end segment computation x-direction")

        # Compute globally averaged zonal spectrum
        # Power spectrum density reference field
        global_wavenumber, global_psd_ref = scipy.signal.welch(np.asarray(ref_segment).flatten(),
                                                               fs=1.0 / delta_x,
                                                               nperseg=npt,
                                                               scaling='density',
                                                               noverlap=0)

        # compute zonal spectrum on grid
        # compute spectrum on grid
        logging.info("start gridding x-direction")

        wavenumber, psd_ref, nb_segment = spectral_computation(YAML,
                                                               np.asarray(ref_segment),
                                                               np.asarray(lon_segment),
                                                               np.asarray(lat_segment),
                                                               delta_x,
                                                               npt)

        logging.info("end gridding x-direction")

        # write netCDF output
        logging.info("start writing x-direction")

        write_netcdf_output(YAML, wavenumber, nb_segment, 'km', psd_ref, global_psd_ref, direction="zonal")

        logging.info("end writing x-direction")

    if direction == "meridional" or direction == "both":

        # compute segment
        logging.info("start segment computation y-direction")

        ref_segment, lon_segment, lat_segment, delta_y, npt = compute_segment_maps(ssh_ref,
                                                                                   lon,
                                                                                   lat,
                                                                                   "meridional",
                                                                                   YAML)

        # write segment
        # write_segment(YAML, lat_segment, lon_segment, ref_segment, delta_x, 'km', npt, direction="meridional")

        logging.info("end segment computation y-direction")

        # Compute globally averaged zonal spectrum
        # Power spectrum density reference field
        global_wavenumber, global_psd_ref = scipy.signal.welch(np.asarray(ref_segment).flatten(),
                                                               fs=1.0 / delta_y,
                                                               nperseg=npt,
                                                               scaling='density',
                                                               noverlap=0)

        # compute zonal spectrum on grid
        # compute spectrum on grid
        logging.info("start gridding y-direction")

        wavenumber, psd_ref, nb_segment = spectral_computation(YAML,
                                                               np.asarray(ref_segment),
                                                               np.asarray(lon_segment),
                                                               np.asarray(lat_segment),
                                                               delta_y,
                                                               npt)

        logging.info("end gridding y-direction")

        # write netCDF output
        logging.info("start writing y-direction")

        write_netcdf_output(YAML, wavenumber, nb_segment, 'km', psd_ref, global_psd_ref, direction="meridional")

        logging.info("end writing y-direction")

    if direction == "temporal":

        lat_out = []
        lon_out = []
        psd_ref_out = []
        wavenumber_out = []

        delta_t = YAML['properties']['spectral_parameters']['delta_t']
        lenght_scale = YAML['properties']['spectral_parameters']['lenght_scale']
        segment_overlapping = YAML['properties']['spectral_parameters']['segment_overlapping']
        output_nc_file = YAML['outputs']['output_filename_t_direction']

        index_lon_min = find_nearest_index(lon, YAML['properties']['study_area']['llcrnrlon'])
        index_lon_max = find_nearest_index(lon, YAML['properties']['study_area']['urcrnrlon'])
        index_lat_min = find_nearest_index(lat, YAML['properties']['study_area']['llcrnrlat'])
        index_lat_max = find_nearest_index(lat, YAML['properties']['study_area']['urcrnrlat'])

        # compute segment
        # logging.info("start segment computation t-direction")

        for jj in range(index_lat_min, index_lat_max + 1):  # ssh_ref[0, :, 0].size):
            for ii in range(index_lon_min, index_lon_max + 1):   # ssh_ref[0, 0, :].size):

                if ssh_ref[:, jj, ii].compressed().size > 0:
                    ref_segment = compute_segment_tide_gauge_tao(ssh_ref[:, jj, ii], time, lenght_scale, delta_t,
                                                                 segment_overlapping)

                    wavenumber, psd_ref = spectral_computation_tide_gauge_tao(np.asarray(ref_segment).flatten(),
                                                                              delta_t,
                                                                              lenght_scale)

                    if wavenumber.size > 0:
                        lat_out.append(lat[jj])
                        lon_out.append(lon[ii])
                        psd_ref_out.append(psd_ref)
                        wavenumber_out.append(wavenumber)

                    else:
                        psd_ref_out.append(np.empty(int(lenght_scale / delta_t) + 1))
                        wavenumber_out.append(np.empty(int(lenght_scale / delta_t) + 1))

                else:
                    psd_ref_out.append(np.empty(int(lenght_scale / delta_t) + 1))
                    wavenumber_out.append(np.empty(int(lenght_scale / delta_t) + 1))

        logging.info("start writing")
        write_netcdf_temporal_output(YAML, np.asarray(wavenumber_out)[1], lat[index_lat_min: index_lat_max + 1],
                                     lon[index_lon_min: index_lon_max + 1], np.asarray(psd_ref_out))
        logging.info("end writing")

else:

    # Read reference input
    logging.info("start reading study grid")
    ssh_study, time, lon, lat, delta_lon_in = read_grid(YAML, 'study')
    logging.info("end reading reference grid")
    ssh_study = ssh_study * YAML['inputs']['study_field_scale_factor']

    ssh_ref_tmp = np.ma.masked_where((ssh_study.filled(-999.) == -999.), ssh_ref)
    ssh_study = np.ma.masked_where((ssh_study.filled(-999.) == -999.), ssh_study)
    ssh_study = np.ma.masked_where((ssh_ref.filled(-999.) == -999.), ssh_study)
    ssh_ref = ssh_ref_tmp

    if direction == "zonal" or direction == "both":

        # compute segment
        logging.info("start segment computation x-direction")

        ref_segment, lon_segment, lat_segment, delta_x, npt = compute_segment_maps(ssh_ref, lon, lat, "zonal", YAML)

        study_segment, _, _, _, _ = compute_segment_maps(ssh_ref, lon, lat, "zonal", YAML)

        logging.info("end segment computation x-direction")

        # Compute globally averaged zonal spectrum
        # Power spectrum density reference field
        global_wavenumber, global_psd_ref = scipy.signal.welch(np.asarray(ref_segment).flatten(),
                                                               fs=1.0 / delta_x,
                                                               nperseg=npt,
                                                               scaling='density',
                                                               noverlap=0)

        _, global_psd_study = scipy.signal.welch(np.asarray(study_segment).flatten(), fs=1.0 / delta_x, nperseg=npt,
                                                 scaling='density', noverlap=0)

        # compute zonal spectrum on grid
        # compute spectrum on grid
        logging.info("start gridding x-direction")

        wavenumber, psd_ref, nb_segment, psd_study, psd_diff_study_ref, coherence, cross_spectrum = \
            spectral_computation(
            YAML, np.asarray(ref_segment), np.asarray(lon_segment), np.asarray(lat_segment), delta_x, npt,
            study_segments=np.asarray(study_segment))

        logging.info("end gridding x-direction")

        # write netCDF output
        logging.info("start writing x-direction")

        write_netcdf_output(YAML, wavenumber, nb_segment, 'km', psd_ref, global_psd_ref,
                            global_psd_study=global_psd_study,
                            psd_study=psd_study,
                            psd_diff_ref_study=psd_diff_study_ref,
                            coherence=coherence,
                            cross_spectrum=cross_spectrum,
                            direction="zonal")

        logging.info("end writing x-direction")

    if direction == "meridional" or direction == "both":

        # compute segment
        logging.info("start segment computation y-direction")

        ref_segment, lon_segment, lat_segment, delta_y, npt = compute_segment_maps(ssh_ref,
                                                                                   lon,
                                                                                   lat,
                                                                                   "meridional",
                                                                                   YAML)

        study_segment, _, _, _, _ = compute_segment_maps(ssh_study, lon, lat, "meridional", YAML)

        logging.info("end segment computation y-direction")

        # Compute globally averaged zonal spectrum
        # Power spectrum density reference field
        global_wavenumber, global_psd_ref = scipy.signal.welch(np.asarray(ref_segment).flatten(),
                                                               fs=1.0 / delta_y,
                                                               nperseg=npt,
                                                               scaling='density',
                                                               noverlap=0)

        _, global_psd_study = scipy.signal.welch(np.asarray(study_segment).flatten(), fs=1.0 / delta_y, nperseg=npt,
                                                 scaling='density', noverlap=0)

        # compute zonal spectrum on grid
        # compute spectrum on grid
        logging.info("start gridding y-direction")

        wavenumber, psd_ref, nb_segment, psd_study, psd_diff_study_ref, coherence, cross_spectrum = \
                spectral_computation(YAML,
                                     np.asarray(ref_segment),
                                     np.asarray(lon_segment),
                                     np.asarray(lat_segment),
                                     delta_y,
                                     npt,
                                     study_segments=np.asarray(study_segment))

        logging.info("end gridding y-direction")

        # write netCDF output
        logging.info("start writing y-direction")

        write_netcdf_output(YAML, wavenumber, nb_segment, 'km', psd_ref, global_psd_ref,
                            global_psd_study=global_psd_study,
                            psd_study=psd_study,
                            psd_diff_ref_study=psd_diff_study_ref,
                            coherence=coherence,
                            cross_spectrum=cross_spectrum,
                            direction="meridional")

        logging.info("end writing y-direction")

    if direction == "temporal":

        lat_out = []
        lon_out = []
        coherence_out = []
        psd_ref_out = []
        psd_study_out = []
        wavenumber_out = []
        psd_diff_ref_study_out = []
        cross_spectrum_out = []

        delta_t = YAML['properties']['spectral_parameters']['delta_t']
        lenght_scale = YAML['properties']['spectral_parameters']['lenght_scale']
        segment_overlapping = YAML['properties']['spectral_parameters']['segment_overlapping']
        output_nc_file = YAML['outputs']['output_filename_t_direction']

        index_lon_min = find_nearest_index(lon, YAML['properties']['study_area']['llcrnrlon'])
        index_lon_max = find_nearest_index(lon, YAML['properties']['study_area']['urcrnrlon'])
        index_lat_min = find_nearest_index(lat, YAML['properties']['study_area']['llcrnrlat'])
        index_lat_max = find_nearest_index(lat, YAML['properties']['study_area']['urcrnrlat'])

        # compute segment
        # logging.info("start segment computation t-direction")

        for jj in range(index_lat_min, index_lat_max + 1):  # ssh_ref[0, :, 0].size):
            for ii in range(index_lon_min, index_lon_max + 1):   # ssh_ref[0, 0, :].size):

                if ssh_ref[:, jj, ii].compressed().size > 0:
                    ref_segment, study_segment = compute_segment_tide_gauge_tao(ssh_ref[:, jj, ii], time,
                                                                                lenght_scale, delta_t,
                                                                                segment_overlapping,
                                                                                msla=ssh_study[:, jj, ii])

                    wavenumber, psd_ref, psd_study, psd_diff_ref_study, coherence, cross_spectrum = \
                        spectral_computation_tide_gauge_tao(np.asarray(ref_segment).flatten(),
                                                            delta_t,
                                                            lenght_scale,
                                                            study_segments=np.asarray(study_segment).flatten())

                    if wavenumber.size > 0:
                        lat_out.append(lat[jj])
                        lon_out.append(lon[ii])
                        coherence_out.append(coherence)
                        psd_ref_out.append(psd_ref)
                        psd_study_out.append(psd_study)
                        wavenumber_out.append(wavenumber)
                        psd_diff_ref_study_out.append(psd_diff_ref_study)
                        cross_spectrum_out.append(cross_spectrum)

                    else:
                        psd_ref_out.append(np.empty(int(lenght_scale / delta_t) + 1))
                        wavenumber_out.append(np.empty(int(lenght_scale / delta_t) + 1))
                        coherence_out.append(np.empty(int(lenght_scale / delta_t) + 1))
                        psd_study_out.append(np.empty(int(lenght_scale / delta_t) + 1))
                        psd_diff_ref_study_out.append(np.empty(int(lenght_scale / delta_t) + 1))
                        cross_spectrum_out.append(np.empty(int(lenght_scale / delta_t) + 1))

                else:
                    psd_ref_out.append(np.empty(int(lenght_scale / delta_t) + 1))
                    wavenumber_out.append(np.empty(int(lenght_scale / delta_t) + 1))
                    psd_ref_out.append(np.empty(int(lenght_scale / delta_t) + 1))
                    wavenumber_out.append(np.empty(int(lenght_scale / delta_t) + 1))
                    coherence_out.append(np.empty(int(lenght_scale / delta_t) + 1))
                    psd_study_out.append(np.empty(int(lenght_scale / delta_t) + 1))
                    psd_diff_ref_study_out.append(np.empty(int(lenght_scale / delta_t) + 1))
                    cross_spectrum_out.append(np.empty(int(lenght_scale / delta_t) + 1))

        logging.info("start writing")
        write_netcdf_temporal_output(YAML, np.asarray(wavenumber_out)[1], lat[index_lat_min: index_lat_max + 1],
                                     lon[index_lon_min: index_lon_max + 1], np.asarray(psd_ref_out),
                                     psd_study=np.asarray(psd_study_out),
                                     psd_diff_ref_study=np.asarray(psd_diff_ref_study_out),
                                     coherence=np.asarray(coherence_out),
                                     cross_spectrum=np.asarray(cross_spectrum_out))
        logging.info("end writing")
