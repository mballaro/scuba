from mod_geo import *
import numpy as np
import scipy.signal
import matplotlib.pylab as plt
from mod_constant import *


def spectral_computation(grid_lon, grid_lat, delta_lon, delta_lat,
                         sla_ref_segments, lon_segment, lat_segment,
                         delta_x, npt, equal_area, greenwhich_start, sla_study_segments=None,
                         cross_correlation_segments=None):
    """
    Spectral computation
    :param grid_lon:
    :param grid_lat:
    :param delta_lon:
    :param delta_lat:
    :param sla_ref_segments:
    :param lon_segment:
    :param lat_segment:
    :param delta_x:
    :param npt:
    :param equal_area:
    :param greenwhich_start:
    :param sla_study_segments:
    :return:
    """

    list_effective_lon = []
    list_effective_lat = []
    list_mean_psd_sla_ref = []
    list_mean_ps_sla_ref = []
    list_nb_segment = []
    list_mean_frequency = []
    list_autocorrelation_ref = []
    list_autocorrelation_ref_zero_crossing = []
    list_autocorrelation_distance = []

    if sla_study_segments is not None:
        list_mean_psd_sla_study = []
        list_mean_ps_sla_study = []
        list_mean_psd_diff_sla_study_sla_ref = []
        list_mean_ps_diff_sla_study_sla_ref = []
        list_mean_coherence = []
        list_effective_resolution = []
        list_useful_resolution = []
        list_autocorrelation_study = []
        list_autocorrelation_study_zero_crossing = []
        list_autocorrelation_distance = []
        list_cross_correlation = []

    for ilat in grid_lat:

        if equal_area:
            lat_min = ilat - 0.5*change_in_latitude(lenght_scale)
            lat_max = ilat + 0.5*change_in_latitude(lenght_scale)
        else:
            lat_min = ilat - 0.5*delta_lat
            lat_max = ilat + 0.5*delta_lat

        effective_lat = 0.5 * (lat_max + lat_min)

        for ilon in grid_lon:

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
            selected_segment = selection_in_latlonbox(lon_segment, lat_segment,
                                                      lon_min, lon_max, lat_min, lat_max, greenwhich_start)

            if selected_segment.size > nb_min_segment:

                selected_sla_ref_segments = np.ma.masked_where(sla_ref_segments[selected_segment].flatten() > 1.E10,
                                   sla_ref_segments[selected_segment].flatten())

                # Power spectrum reference field
                wavenumber, ps_sla_ref = scipy.signal.welch(
                    selected_sla_ref_segments, fs=1.0 / delta_x, nperseg=npt,
                    scaling='spectrum', noverlap=0)

                # Power spectrum density reference field
                wavenumber, psd_sla_ref = scipy.signal.welch(
                    selected_sla_ref_segments, fs=1.0 / delta_x, nperseg=npt,
                    scaling='density', noverlap=0)

                # Autocorrelation function of study field
                autocorrelation, distance = compute_autocorrelation(psd_sla_ref, wavenumber)
                autocorrelation_zero_crossing = compute_resolution(
                    autocorrelation, distance, threshold=0.0)

                if autocorrelation_zero_crossing > 0.:
                    autocorrelation_zero_crossing = 1.0 / autocorrelation_zero_crossing
                else:
                    autocorrelation_zero_crossing = 0.

                list_effective_lon.append(effective_lon)
                list_effective_lat.append(effective_lat)
                list_mean_frequency.append(wavenumber)
                list_mean_psd_sla_ref.append(psd_sla_ref)
                list_mean_ps_sla_ref.append(ps_sla_ref)
                list_nb_segment.append(selected_segment.size)
                list_autocorrelation_ref.append(autocorrelation)
                list_autocorrelation_ref_zero_crossing.append(autocorrelation_zero_crossing)
                list_autocorrelation_distance.append(distance)

                if sla_study_segments is not None:

                    selected_sla_study_segments = np.ma.masked_where(
                        sla_study_segments[selected_segment].flatten() > 1.E10,
                        sla_study_segments[selected_segment].flatten())

                    diff_sla_study_sla_ref = selected_sla_study_segments - selected_sla_ref_segments

                    # Power spectrum of the error between to field
                    wavenumber, ps_diff_sla_study_sla_ref = scipy.signal.welch(
                        diff_sla_study_sla_ref, fs=1.0 / delta_x, nperseg=npt, scaling='spectrum', noverlap=0)

                    # Power spectrum density of the error between to field
                    wavenumber, psd_diff_sla_study_sla_ref = scipy.signal.welch(
                        diff_sla_study_sla_ref, fs=1.0 / delta_x, nperseg=npt, scaling='density', noverlap=0)

                    # Power spectrum study field
                    wavenumber, ps_sla_study = scipy.signal.welch(
                        selected_sla_study_segments, fs=1.0 / delta_x, nperseg=npt,
                        scaling='spectrum', noverlap=0)

                    # Power spectrum density study field
                    wavenumber, psd_sla_study = scipy.signal.welch(
                        selected_sla_study_segments, fs=1.0 / delta_x, nperseg=npt,
                        scaling='density', noverlap=0)

                    # Magnitude square coherence between the ref and study field
                    wavenumber, coherence = scipy.signal.coherence(
                        selected_sla_study_segments,
                        selected_sla_ref_segments, fs=1.0 / delta_x, nperseg=npt, noverlap=0)

                    # Effective resolution
                    effective_resolution = compute_resolution(coherence, wavenumber)

                    # Useful resolution
                    useful_resolution = compute_resolution(psd_sla_study / psd_sla_ref, wavenumber)

                    # Autocorrelation function of study field
                    autocorrelation, distance = compute_autocorrelation(psd_sla_study, wavenumber)
                    autocorrelation_zero_crossing = compute_resolution(
                        autocorrelation, distance, threshold=0.0)

                    if autocorrelation_zero_crossing > 0.:
                        autocorrelation_zero_crossing = 1.0 / autocorrelation_zero_crossing
                    else:
                        autocorrelation_zero_crossing = 0.

                    #print cross_correlation, cross_correlation.size, selected_sla_study_segments.size

                    list_mean_ps_sla_study.append(ps_sla_study)
                    list_mean_ps_diff_sla_study_sla_ref.append(ps_diff_sla_study_sla_ref)
                    list_mean_psd_sla_study.append(psd_sla_study)
                    list_mean_psd_diff_sla_study_sla_ref.append(psd_diff_sla_study_sla_ref)
                    list_mean_coherence.append(coherence)
                    list_effective_resolution.append(effective_resolution)
                    list_useful_resolution.append(useful_resolution)
                    list_autocorrelation_study.append(autocorrelation)
                    list_autocorrelation_study_zero_crossing.append(autocorrelation_zero_crossing)

                    # mean_cross_correlation = np.mean(cross_correlation_segments[selected_segment], axis=0)
                    # list_cross_correlation.append(mean_cross_correlation)

            else:
                list_effective_lon.append(effective_lon)
                list_effective_lat.append(effective_lat)
                list_mean_frequency.append(np.zeros((npt / 2 + 1)))
                list_mean_psd_sla_ref.append(np.zeros((npt / 2 + 1)))
                list_mean_ps_sla_ref.append(np.zeros((npt / 2 + 1)))
                list_nb_segment.append(0.)
                list_autocorrelation_ref.append(np.zeros(int(round(npt/4.0))))
                list_autocorrelation_ref_zero_crossing.append(0.)
                list_autocorrelation_distance.append(np.zeros(int(round(npt/4.0))))

                if sla_study_segments is not None:
                    list_mean_ps_sla_study.append(np.zeros((npt / 2 + 1)))
                    list_mean_ps_diff_sla_study_sla_ref.append(np.zeros((npt / 2 + 1)))
                    list_mean_psd_sla_study.append(np.zeros((npt / 2 + 1)))
                    list_mean_psd_diff_sla_study_sla_ref.append(np.zeros((npt / 2 + 1)))
                    list_mean_coherence.append(np.zeros((npt / 2 + 1)))
                    list_effective_resolution.append(0.)
                    list_useful_resolution.append(0.)
                    list_autocorrelation_study.append(np.zeros(int(round(npt/4.0))))
                    list_autocorrelation_study_zero_crossing.append(0.)
                    # list_cross_correlation.append(np.zeros((npt)))

    if sla_study_segments is not None:

        return list_effective_lon, list_effective_lat, list_mean_frequency, \
               list_mean_psd_sla_ref, list_mean_ps_sla_ref, list_nb_segment, \
               list_autocorrelation_ref, list_autocorrelation_ref_zero_crossing, list_autocorrelation_distance, \
               list_mean_ps_sla_study, list_mean_ps_diff_sla_study_sla_ref, \
               list_mean_psd_sla_study, list_mean_psd_diff_sla_study_sla_ref, \
               list_mean_coherence, list_effective_resolution, list_useful_resolution, \
               list_autocorrelation_study, list_autocorrelation_study_zero_crossing
        #, list_cross_correlation

    else:

        return list_effective_lon, list_effective_lat, list_mean_frequency, \
               list_mean_psd_sla_ref, list_mean_ps_sla_ref, list_nb_segment, \
               list_autocorrelation_ref, list_autocorrelation_ref_zero_crossing, list_autocorrelation_distance


def spectral_computation_tide_gauge(sla_ref_segments, delta_t, npt, sla_study_segments=None):
    """

    :param sla_ref_segments:
    :param delta_t:
    :param npt:
    :param sla_study_segments:
    :return:
    """

    # Power spectrum reference field
    wavenumber, ps_sla_ref = scipy.signal.welch(
        sla_ref_segments, fs=1.0 / delta_t, nperseg=npt,
        scaling='spectrum', noverlap=0)

    # Power spectrum density reference field
    wavenumber, psd_sla_ref = scipy.signal.welch(
        sla_ref_segments, fs=1.0 / delta_t, nperseg=npt,
        scaling='density', noverlap=0)

    # Autocorrelation function of study field
    try:
        autocorrelation_ref, distance = compute_autocorrelation(psd_sla_ref, wavenumber)
    except ValueError:
        autocorrelation_ref, distance = None, None

    try:
        autocorrelation_ref_zero_crossing = compute_resolution(autocorrelation_ref, distance, threshold=0.0)
    except:
        autocorrelation_ref_zero_crossing = 0.

    if autocorrelation_ref_zero_crossing > 0.:
        autocorrelation_ref_zero_crossing = 1.0 / autocorrelation_ref_zero_crossing
    else:
        autocorrelation_ref_zero_crossing = 0.

    if sla_study_segments is not None:

        diff_sla_study_sla_ref = sla_study_segments - sla_ref_segments

        # Power spectrum of the error between to field
        wavenumber, ps_diff_sla_study_sla_ref = scipy.signal.welch(
            diff_sla_study_sla_ref, fs=1.0 / delta_t, nperseg=npt, scaling='spectrum', noverlap=0)

        # Power spectrum density of the error between to field
        wavenumber, psd_diff_sla_study_sla_ref = scipy.signal.welch(
            diff_sla_study_sla_ref, fs=1.0 / delta_t, nperseg=npt, scaling='density', noverlap=0)

        # Power spectrum study field
        wavenumber, ps_sla_study = scipy.signal.welch(
            sla_study_segments, fs=1.0 / delta_t, nperseg=npt,
            scaling='spectrum', noverlap=0)

        # Power spectrum density study field
        wavenumber, psd_sla_study = scipy.signal.welch(
            sla_study_segments, fs=1.0 / delta_t, nperseg=npt,
            scaling='density', noverlap=0)

        # Magnitude square coherence between the ref and study field
        wavenumber, coherence = scipy.signal.coherence(
            sla_study_segments,
            sla_ref_segments, fs=1.0 / delta_t, nperseg=npt, noverlap=0)

        #plt.plot(sla_study_segments, color='r')
        #plt.plot(sla_ref_segments, color='b')
        #plt.show()
        # Effective resolution
        effective_resolution = compute_resolution(coherence, wavenumber)

        # Useful resolution
        useful_resolution = compute_resolution(psd_sla_study / psd_sla_ref, wavenumber)

        # Autocorrelation function of study field
        try:
            autocorrelation_study, distance = compute_autocorrelation(psd_sla_study, wavenumber)
        except ValueError:
            autocorrelation_study, distance = None, None

        try:
            autocorrelation_study_zero_crossing = compute_resolution(
                autocorrelation_study, distance, threshold=0.0)
        except:
            autocorrelation_study_zero_crossing = 0.

        if autocorrelation_study_zero_crossing > 0.:
            autocorrelation_study_zero_crossing = 1.0 / autocorrelation_study_zero_crossing
        else:
            autocorrelation_study_zero_crossing = 0.

        return wavenumber, psd_sla_ref, ps_sla_ref, autocorrelation_ref, autocorrelation_ref_zero_crossing,\
               distance, ps_sla_study, ps_diff_sla_study_sla_ref, \
               psd_sla_study, psd_diff_sla_study_sla_ref, coherence, effective_resolution, useful_resolution, \
               autocorrelation_study, autocorrelation_study_zero_crossing

    else:

        return wavenumber, psd_sla_ref, ps_sla_ref, autocorrelation_ref, autocorrelation_ref_zero_crossing,\
               distance


def compute_lombscargle_periodogram(time_sample, sla_sample, output_freq_sample):

    pgram = scipy.signal.lombscargle(time_sample, sla_sample, output_freq_sample)

    return pgram


def compute_resolution(array, frequency, threshold=0.5):
    """
    Given a coherence profile or spectral ratio, compute the resolution
    (i.e., where coherence = threshold or ratio = threshold or autocorrelation = threshold)
    :param array:
    :param frequency:
    :param threshold:
    :return:
    """

    ii05 = np.where(np.diff(np.where(array[:] > threshold)[0]) > 1)[0]
    if not ii05.size:
        if np.where(array[:] > threshold)[0].size:
            ii05 = np.where(array[:] > threshold)[0][-1]
        else:
            ii05 = -1
    elif ii05.size > 1:
        ii05 = ii05[-1]

    if (ii05 + 1 < array[:].size) and (ii05 + 1 >= 0):
        d1 = array[ii05] - threshold
        d2 = threshold - array[ii05 + 1]

        if (d1 + d2 != 0) and (frequency[ii05] != 0) and (frequency[ii05+1] != 0):
            logres = np.log(frequency[ii05]) * (d2 / (d1 + d2)) + np.log(frequency[ii05 + 1]) * (d1 / (d1 + d2))
            resolution = 1. / np.exp(logres)

        else:

            resolution = 0.
    else:

        resolution = 0.

    return resolution


def compute_autocorrelation(psd, wavenumber, display=False):
    """
    Compute autocorrelation function from power spectrum
    :param psd:
    :param wavenumber:
    :param display:
    :return:
    """

    autocorrelation = np.fft.ifft(psd).real
    autocorrelation /= autocorrelation[0]
    delta_f = 1 / (wavenumber[-1] - wavenumber[-2])
    distance = np.linspace(0, int(delta_f/2), int(autocorrelation.size / 2))

    if display:
        plt.plot(distance, autocorrelation[:int(autocorrelation.size/2)], lw=2, color='r')
        plt.xlabel("DISTANCE (km)")
        plt.ylabel("AUTOCORRELATION")
        plt.axhline(y=0., color='k', lw=2)
        plt.show()

    return autocorrelation[:int(autocorrelation.size/2)], distance
