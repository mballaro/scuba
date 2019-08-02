from mod_geo import *
import numpy as np
import scipy.signal
import matplotlib.pylab as plt
from mod_constant import *


def spectral_computation(config, ref_segments, lon_segment, lat_segment, delta_x, npt, study_segments=None):
    """

    :param config:
    :param ref_segments:
    :param lon_segment:
    :param lat_segment:
    :param delta_x:
    :param npt:
    :param study_segments:
    :return:
    """

    study_lon_min = config['properties']['study_area']['llcrnrlon']
    study_lon_max = config['properties']['study_area']['urcrnrlon']
    study_lat_min = config['properties']['study_area']['llcrnrlat']
    study_lat_max = config['properties']['study_area']['urcrnrlat']
    grid_lat = np.arange(study_lat_min, study_lat_max, config['outputs']['output_lat_resolution'])
    grid_lon = np.arange(study_lon_min, study_lon_max, config['outputs']['output_lon_resolution'])
    delta_lat = config['properties']['spectral_parameters']['delta_lat']
    delta_lon = config['properties']['spectral_parameters']['delta_lon']

    list_mean_psd_ref = []
    list_nb_segment = []
    list_mean_frequency = []

    list_mean_psd_study = []
    list_mean_psd_diff_study_ref = []
    list_mean_coherence = []
    list_mean_cross_spectrum = []
    # list_cross_correlation = []

    fs = 1.0 / delta_x

    # Loop over output lon/lat boxes and selection of the segment within the box plus/minus delta_lon/lat
    for ilat in grid_lat:

        lat_min = ilat - 0.5*delta_lat
        lat_max = ilat + 0.5*delta_lat

        selected_lat_index = np.where(np.logical_and(lat_segment >= lat_min, lat_segment <= lat_max))[0]
        ref_segments_tmp = ref_segments[selected_lat_index]
        if study_segments is not None:
            study_segments_tmp = study_segments[selected_lat_index]
        else:
            study_segments_tmp = None

        for ilon in grid_lon % 360.:

            lon_min = ilon - 0.5*delta_lon
            lon_max = ilon + 0.5*delta_lon

            if (lon_min < 0.) and (lon_max > 0.):
                selected_segment = np.where(np.logical_or(lon_segment[selected_lat_index] % 360. >= lon_min + 360.,
                                                          lon_segment[selected_lat_index] % 360. <= lon_max))[0]
            elif (lon_min > 0.) and (lon_max > 360.):
                selected_segment = np.where(np.logical_or(lon_segment[selected_lat_index] % 360. >= lon_min,
                                                          lon_segment[selected_lat_index] % 360. <= lon_max - 360.))[0]
            else:
                selected_segment = np.where(np.logical_and(lon_segment[selected_lat_index] % 360. >= lon_min,
                                                           lon_segment[selected_lat_index] % 360. <= lon_max))[0]

            if len(selected_segment) > nb_min_segment:
                selected_ref_segments = np.ma.masked_where(ref_segments_tmp[selected_segment].flatten() > 1.E10,
                                                           ref_segments_tmp[selected_segment].flatten())

                # Power spectrum density reference field
                wavenumber, psd_ref = scipy.signal.welch(selected_ref_segments,
                                                         fs=fs,
                                                         nperseg=npt,
                                                         scaling='density',
                                                         noverlap=0)

                list_mean_frequency.append(wavenumber)
                list_mean_psd_ref.append(psd_ref)
                list_nb_segment.append(selected_segment.size)

                if study_segments is not None:

                    selected_study_segments = np.ma.masked_where(
                        study_segments_tmp[selected_segment].flatten() > 1.E10,
                        study_segments_tmp[selected_segment].flatten())

                    # Compute diff study minus ref
                    diff_study_ref = selected_study_segments - selected_ref_segments

                    # Power spectrum density of the error between to field
                    wavenumber, psd_diff_study_ref = scipy.signal.welch(diff_study_ref,
                                                                        fs=fs,
                                                                        nperseg=npt,
                                                                        scaling='density',
                                                                        noverlap=0)

                    # Power spectrum density study field
                    wavenumber, psd_study = scipy.signal.welch(selected_study_segments,
                                                               fs=fs,
                                                               nperseg=npt,
                                                               scaling='density',
                                                               noverlap=0)

                    # Magnitude square coherence between the ref and study field
                    wavenumber, coherence = scipy.signal.coherence(selected_study_segments,
                                                                   selected_ref_segments,
                                                                   fs=fs,
                                                                   nperseg=npt,
                                                                   noverlap=0)

                    # Cross spectrum
                    wavenumber, cross_spectrum = scipy.signal.csd(selected_study_segments,
                                                                  selected_ref_segments,
                                                                  fs=fs,
                                                                  nperseg=npt,
                                                                  noverlap=0)

                    list_mean_psd_study.append(psd_study)
                    list_mean_psd_diff_study_ref.append(psd_diff_study_ref)
                    list_mean_coherence.append(coherence)
                    list_mean_cross_spectrum.append(cross_spectrum)

            else:

                list_mean_frequency.append(np.zeros((int(npt / 2) + 1)))
                list_mean_psd_ref.append(np.zeros((int(npt / 2) + 1)))
                list_nb_segment.append(0.)

                if study_segments is not None:
                    list_mean_psd_study.append(np.zeros((int(npt / 2) + 1)))
                    list_mean_psd_diff_study_ref.append(np.zeros((int(npt / 2) + 1)))
                    list_mean_coherence.append(np.zeros((int(npt / 2) + 1)))
                    list_mean_cross_spectrum.append(np.zeros((int(npt / 2) + 1)))

    if study_segments is not None:

        return list_mean_frequency, \
               list_mean_psd_ref, list_nb_segment, \
               list_mean_psd_study, list_mean_psd_diff_study_ref, \
               list_mean_coherence, list_mean_cross_spectrum

    else:

        return list_mean_frequency, list_mean_psd_ref, list_nb_segment


def spectral_computation_tide_gauge_tao(ref_segments, delta_t, npt, study_segments=None):
    """

    :param ref_segments:
    :param delta_t:
    :param npt:
    :param study_segments:
    :return:
    """

    fs = 1.0 / delta_t
    # Power spectrum density reference field
    wavenumber, psd_ref = scipy.signal.welch(ref_segments, fs=fs, nperseg=npt, scaling='density', noverlap=0)

    if study_segments is not None:

        diff_study_ref = study_segments - ref_segments

        # Power spectrum density of the error between study and reference field
        wavenumber, psd_diff_study_ref = scipy.signal.welch(diff_study_ref,
                                                            fs=fs,
                                                            nperseg=npt,
                                                            scaling='density',
                                                            noverlap=0)

        # Power spectrum density study field
        wavenumber, psd_study = scipy.signal.welch(study_segments,
                                                   fs=fs,
                                                   nperseg=npt,
                                                   scaling='density',
                                                   noverlap=0)

        # Magnitude square coherence between the ref and study field
        wavenumber, coherence = scipy.signal.coherence(study_segments, ref_segments, fs=fs, nperseg=npt, noverlap=0)

        # Cross spectrum
        wavenumber, cross_spectrum = scipy.signal.csd(study_segments, ref_segments, fs=fs, nperseg=npt, noverlap=0)

        return wavenumber, psd_ref, psd_study, psd_diff_study_ref, coherence, cross_spectrum

    else:

        return wavenumber, psd_ref


def compute_lombscargle_periodogram(time_sample, sample, output_freq_sample):

    pgram = scipy.signal.lombscargle(time_sample, sample, output_freq_sample)

    return pgram
