
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.basemap import Basemap

from mod_constant import *
from mod_io import *
from mod_geo import *


def compute_segment_xy_maps(msla_ref, lon, lat, lenght_scale, msla_study=None, debug=False):
    """
    Segment computation in zonal and meridional direction
    :param msla_ref:
    :param lon:
    :param lat:
    :param lenght_scale:
    :param msla_study:
    :param debug:
    :return:
    """

    list_lat_segment_x = []
    list_lon_segment_x = []
    list_sla_ref_segment_x = []
    list_lat_segment_y = []
    list_lon_segment_y = []
    list_sla_ref_segment_y = []

    if msla_study is not None:
        list_sla_study_segment_x = []
        list_sla_study_segment_y = []

    all_npt_x = []
    # LOOP along latitude to get mean segment lenght in x-direction (true segment length is changing with lat)
    for jj in range(lat.size):
        if np.abs(lat[jj]) < 60:
            # Take only latitude that are lower than 60degree otherwise mean segment is too long (CAN BE IMPROVED)
            delta_x = haversine(lon[0], lat[jj], lon[1], lat[jj])
            all_npt_x.append(int(lenght_scale / delta_x))
    npt_x = np.int(np.mean(np.asarray(all_npt_x)))

    if debug:
        # Initialisation
        nb_fig_x = 0
        nb_fig_y = 0

    for tt in range(msla_ref[:, 0, 0].size):

        # LOOP along longitude to get segment in y-direction
        for jj in range(lat.size):

            if msla_ref[tt, jj, :].size >= npt_x:

                lon_vector = np.ma.masked_where(msla_ref[tt, jj, :].filled(-999.) == -999., lon)
                # cut  when diff lon longer delta_x
                indi = np.where(lon_vector.filled(-999.) == -999.)[0]
                indi = np.append(indi, msla_ref[tt, jj, :].size)

                if indi.size:
                    segment_lenght = np.insert(np.diff(indi), [0], indi[0])
                else:
                    indi = [0, lon_vector.size]
                    segment_lenght = np.insert(np.diff(indi), [0], indi[0])

                selected_segment = np.where(segment_lenght >= npt_x)[0]

                if selected_segment.size > 0:
                    for track in selected_segment:
                        if track - 1 >= 0:
                            index_start_selected_track = indi[track-1]
                            index_end_selected_track = indi[track]
                        else:
                            index_start_selected_track = 0
                            index_end_selected_track = indi[track]

                        start_point = index_start_selected_track
                        end_point = index_end_selected_track

                        for sub_segment_point in range(start_point, end_point - npt_x, int(npt_x*segment_overlapping)):

                            mean_lon_sub_segment = np.median(lon[sub_segment_point:sub_segment_point + npt_x])
                            mean_lat_sub_segment = lat[jj]

                            if debug:
                                plot_segment(lon_vector[start_point:end_point],
                                             lat[jj] * np.ones(lon_vector[start_point:end_point].size),
                                             lon_vector[sub_segment_point:sub_segment_point + npt_x],
                                             lat[jj] * np.ones(
                                                 lon_vector[sub_segment_point:sub_segment_point + npt_x].size),
                                             mean_lon_sub_segment,
                                             mean_lat_sub_segment,
                                             "fig_x_segment",
                                             nb_fig_x)
                                nb_fig_x += 1

                            msla_ref_segment = np.ma.masked_invalid(
                                msla_ref[tt, jj, sub_segment_point:sub_segment_point + npt_x])
                            msla_ref_segment = np.ma.masked_where(np.abs(msla_ref_segment) > 1.E10, msla_ref_segment)

                            if msla_study is not None:
                                msla_study_segment = np.ma.masked_invalid(
                                    msla_study[tt, jj, sub_segment_point:sub_segment_point + npt_x])
                                msla_study_segment = np.ma.masked_where(
                                    np.abs(msla_study_segment) > 1.E10, msla_study_segment)

                            if msla_ref_segment.compressed().size == npt_x:
                                list_sla_ref_segment_x.append(msla_ref_segment)
                                list_lon_segment_x.append(mean_lon_sub_segment)
                                list_lat_segment_x.append(mean_lat_sub_segment)
                                if msla_study is not None:
                                    list_sla_study_segment_x.append(msla_study_segment)

        # Loop along longitude to get segment in y-direction
        for ii in range(lon.size):

            delta_y = haversine(lon[ii], lat[0], lon[ii], lat[1])
            npt_y = int(lenght_scale / delta_y)

            if msla_ref[tt, :, ii].size >= npt_y:

                lat_vector = np.ma.masked_where(msla_ref[tt, :, ii].filled(-999.) == -999., lat)
                # cut  when diff lat longer delta_y
                indi = np.where(lat_vector.filled(-999.) == -999.)[0]
                indi = np.append(indi, msla_ref[tt, :, ii].size)

                if indi.size:
                    segment_lenght = np.insert(np.diff(indi), [0], indi[0])
                else:
                    indi = [0, lat_vector.size]
                    segment_lenght = np.insert(np.diff(indi), [0], indi[0])

                selected_segment = np.where(segment_lenght >= npt_y)[0]

                if selected_segment.size > 0:
                    for track in selected_segment:

                        if track - 1 >= 0:
                            index_start_selected_track = indi[track - 1]
                            index_end_selected_track = indi[track]
                        else:
                            index_start_selected_track = 0
                            index_end_selected_track = indi[track]

                        start_point = index_start_selected_track
                        end_point = index_end_selected_track

                        for sub_segment_point in range(start_point, end_point - npt_y, int(npt_y*segment_overlapping)):

                            mean_lon_sub_segment = lon[ii]
                            mean_lat_sub_segment = np.median(lat[sub_segment_point:sub_segment_point + npt_y])

                            if debug:
                                plot_segment(lon[ii] * np.ones(lat_vector[start_point:end_point].size),
                                             lat_vector[start_point:end_point],
                                             lon[ii] * np.ones(
                                                 lat_vector[sub_segment_point:sub_segment_point + npt_y].size),
                                             lat_vector[sub_segment_point:sub_segment_point + npt_y],
                                             mean_lon_sub_segment,
                                             mean_lat_sub_segment,
                                             "fig_y_segment",
                                             nb_fig_y)
                                nb_fig_y += 1

                            msla_ref_segment = np.ma.masked_invalid(
                                msla_ref[tt, sub_segment_point:sub_segment_point + npt_y, ii])
                            msla_ref_segment = np.ma.masked_where(np.abs(msla_ref_segment) > 1.E10, msla_ref_segment)

                            if msla_study is not None:
                                msla_study_segment = np.ma.masked_invalid(
                                    msla_study[tt, sub_segment_point:sub_segment_point + npt_y, ii])
                                msla_study_segment = np.ma.masked_where(
                                    np.abs(msla_study_segment) > 1.E10, msla_study_segment)

                            if msla_ref_segment.compressed().size == npt_y:
                                list_sla_ref_segment_y.append(msla_ref_segment)
                                list_lon_segment_y.append(mean_lon_sub_segment)
                                list_lat_segment_y.append(mean_lat_sub_segment)
                                if msla_study is not None:
                                    list_sla_study_segment_y.append(msla_study_segment)

    if msla_study is not None:
        return list_lat_segment_x, list_lon_segment_x, list_sla_ref_segment_x, delta_x, npt_x, \
               list_lat_segment_y, list_lon_segment_y, list_sla_ref_segment_y, delta_y, npt_y, \
               list_sla_study_segment_x, list_sla_study_segment_y
    else:
        return list_lat_segment_x, list_lon_segment_x, list_sla_ref_segment_x, delta_x, npt_x, \
               list_lat_segment_y, list_lon_segment_y, list_sla_ref_segment_y, delta_y, npt_y


def compute_segment_alongtrack(sla, lon_along_track, lat_along_track, time_along_track, mission, mission_management,
                               lenght_scale, msla=None, debug=False):
    """
    Segment computation in alongtrack direction
    :param sla:
    :param lon_along_track:
    :param lat_along_track:
    :param time_along_track:
    :param mission:
    :param mission_management:
    :param lenght_scale:
    :param msla:
    :param debug:
    :return:
    """

    list_lat_segment = []
    list_lon_segment = []
    list_sla_segment = []
    list_crosscorrelation_segment = []

    if msla is not None:
        list_msla_segment = []

    # Get number of point to consider for resolution = lenghtscale in km
    delta_t = get_deltat(mission.upper(), mission_management)
    delta_x = get_velocity(mission.upper(), mission_management) * delta_t
    delta_t_jd = delta_t / (3600 * 24)
    npt = int(lenght_scale / delta_x)

    # cut track when diff time longer than 4*delta_t
    indi = np.where((np.diff(time_along_track) > 4 * delta_t_jd))[0]
    track_segment_lenght = np.insert(np.diff(indi), [0], indi[0])

    # Long track >= npt
    selected_track_segment = np.where(track_segment_lenght >= npt)[0]

    if debug:
        nb_fig = 0

    if selected_track_segment.size > 0:

        for track in selected_track_segment:

            if track-1 >= 0:
                index_start_selected_track = indi[track-1]
                index_end_selected_track = indi[track]
            else:
                index_start_selected_track = 0
                index_end_selected_track = indi[track]

            start_point = index_start_selected_track
            end_point = index_end_selected_track

            for sub_segment_point in range(start_point, end_point - npt, int(npt*segment_overlapping)):

                # Near Greenwhich case
                if ((lon_along_track[sub_segment_point + npt - 1] < 50.)
                    and (lon_along_track[sub_segment_point] > 320.)) \
                        or ((lon_along_track[sub_segment_point + npt - 1] > 320.)
                            and (lon_along_track[sub_segment_point] < 50.)):

                    tmp_lon = np.where(lon_along_track[sub_segment_point:sub_segment_point + npt] > 180,
                                       lon_along_track[sub_segment_point:sub_segment_point + npt] - 360,
                                       lon_along_track[sub_segment_point:sub_segment_point + npt])
                    mean_lon_sub_segment = np.median(tmp_lon)

                    if mean_lon_sub_segment < 0:
                        mean_lon_sub_segment = mean_lon_sub_segment + 360.
                else:

                    mean_lon_sub_segment = np.median(lon_along_track[sub_segment_point:sub_segment_point + npt])

                mean_lat_sub_segment = np.median(lat_along_track[sub_segment_point:sub_segment_point + npt])

                if debug:
                    plot_segment(lon_along_track[start_point:end_point],
                                 lat_along_track[start_point:end_point],
                                 lon_along_track[sub_segment_point:sub_segment_point + npt],
                                 lat_along_track[sub_segment_point:sub_segment_point + npt],
                                 mean_lon_sub_segment,
                                 mean_lat_sub_segment,
                                 "fig_alongtrack_segment",
                                 nb_fig)
                    nb_fig += 1

                sla_segment = np.ma.masked_invalid(sla[sub_segment_point:sub_segment_point + npt])

                if msla is not None:
                    msla_segment = np.ma.masked_invalid(msla[sub_segment_point:sub_segment_point + npt])
                    if np.ma.is_masked(msla_segment):
                        # print "MASK detected"
                        sla_segment = np.ma.compressed(np.ma.masked_where(np.ma.is_masked(msla_segment), sla_segment))
                        msla_segment = np.ma.compressed(msla_segment)

                    #plt.plot(sla_segment, color='r', label="SLA along-track")
                    #plt.plot(msla_segment, color='b', label='interpolated MSLA')
                    #plt.legend(loc="best")
                    #plt.show()
                    # Cross correlation
                    #import scipy
                    # normalization
                    fld1 = (sla_segment - np.mean(sla_segment)) / (np.std(sla_segment) * len(sla_segment))
                    fld2 = (msla_segment - np.mean(msla_segment)) / (np.std(msla_segment))
                    cross_correlation = np.correlate(fld1, fld2, mode='same')
                    #cross_correlation = np.correlate(sla_segment-np.mean(sla_segment), msla_segment, mode = 'same')
                    #cross_correlation = np.square(scipy.signal.correlate(sla_segment, msla_segment, mode='full'))

                if sla_segment.size > 0:
                    list_sla_segment.append(sla_segment)
                    list_lon_segment.append(mean_lon_sub_segment)
                    list_lat_segment.append(mean_lat_sub_segment)
                    if msla is not None:
                        list_crosscorrelation_segment.append(cross_correlation)
                        list_msla_segment.append(msla_segment)

    if msla is not None:
        return list_sla_segment, list_lon_segment, list_lat_segment, delta_x, npt, list_msla_segment, \
               list_crosscorrelation_segment

    else:
        return list_sla_segment, list_lon_segment, list_lat_segment, delta_x, npt


def compute_segment_tide_gauge(sla_tg, time_tg, lenght_scale, delta_t, msla=None):
    """

    :param sla_tg:
    :param time_tg:
    :param lenght_scale:
    :param msla:
    :return:
    """

    list_lat_segment = []
    list_lon_segment = []
    list_sla_segment = []
    list_crosscorrelation_segment = []

    if msla is not None:
        list_msla_segment = []

    # Get number of point to consider for resolution = lenghscale
    npt = int(lenght_scale / delta_t)

    # cut track when diff time longer than 4*delta_t
    indi = np.where((np.diff(time_tg) > delta_t))[0]

    if len(indi) > 1:
        track_segment_lenght = np.insert(np.diff(indi), [0], indi[0])
    else:
        track_segment_lenght = [time_tg.size]
        indi = [time_tg.size]

    # Long track >= npt
    selected_track_segment = np.where(track_segment_lenght >= npt)[0]

    if selected_track_segment.size > 0:

        for track in selected_track_segment:

            if track-1 >= 0:
                index_start_selected_track = indi[track-1]
                index_end_selected_track = indi[track]
            else:
                index_start_selected_track = 0
                index_end_selected_track = indi[track]

            start_point = index_start_selected_track
            end_point = index_end_selected_track

            for sub_segment_point in range(start_point, end_point - npt, int(npt*segment_overlapping)):

                sla_segment = np.ma.masked_invalid(sla_tg[sub_segment_point:sub_segment_point + npt])

                if msla is not None:
                    msla_segment = np.ma.masked_invalid(msla[sub_segment_point:sub_segment_point + npt])
                    if np.ma.is_masked(msla_segment):
                        # print "MASK detected"
                        sla_segment = np.ma.compressed(np.ma.masked_where(np.ma.is_masked(msla_segment), sla_segment))
                        msla_segment = np.ma.compressed(msla_segment)

                    # plt.plot(sla_segment, color='r', label="SLA tide gauge")
                    # plt.plot(msla_segment, color='b', label='best correlated MSLA')
                    # plt.legend(loc="best")
                    # plt.show()

                    # Cross correlation
                    # normalization
                    fld1 = (sla_segment - np.mean(sla_segment)) / (np.std(sla_segment) * len(sla_segment))
                    fld2 = (msla_segment - np.mean(msla_segment)) / (np.std(msla_segment))
                    cross_correlation = np.correlate(fld1, fld2, mode='same')

                if sla_segment.size > 0:
                    list_sla_segment.append(sla_segment)
                    list_crosscorrelation_segment.append(cross_correlation)
                    if msla is not None:
                        list_msla_segment.append(msla_segment)

    if msla is not None:
        return list_sla_segment, list_msla_segment

    else:
        return list_sla_segment


def cross_correlation(a1, a2):
        lags = range(-len(a1)+1, len(a2))
        cs = []
        for lag in lags:
            idx_lower_a1 = max(lag, 0)
            idx_lower_a2 = max(-lag, 0)
            idx_upper_a1 = min(len(a1), len(a1)+lag)
            idx_upper_a2 = min(len(a2), len(a2)-lag)
            b1 = a1[idx_lower_a1:idx_upper_a1]
            b2 = a2[idx_lower_a2:idx_upper_a2]
            c = np.correlate(b1, b2)[0]
            c = c / np.sqrt((b1**2).sum() * (b2**2).sum())
            cs.append(c)
        return cs


def plot_segment(xlon_segment, xlat_segment, xlon_subsegment, xlat_subsegment, mean_lon, mean_lat, prefix, nb_fig):
    """
    Display segment on maps (useful for debugging)
    :param xlon_segment:
    :param xlat_segment:
    :param xlon_subsegment:
    :param xlat_subsegment:
    :param mean_lon:
    :param mean_lat:
    :param prefix:
    :param nb_fig:
    :return:
    """

    plt.figure()
    bmap = Basemap(
        projection='cyl', resolution='l', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180,
        urcrnrlon=360)
    bmap.fillcontinents(color='grey', lake_color='white')
    bmap.drawcoastlines()
    bmap.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0], fontsize=15, dashes=[4, 4],
                       linewidth=0.25)
    bmap.drawmeridians(np.arange(-200, 360, 60), labels=[0, 0, 0, 1], fontsize=15,
                       dashes=[4, 4],
                       linewidth=0.25)
    x, y = bmap(xlon_segment, xlat_segment)
    bmap.scatter(x, y, s=0.1, marker="o", color='g')
    x, y = bmap(xlon_subsegment, xlat_subsegment)
    bmap.scatter(x, y, s=0.1, marker="o", color='r')
    bmap.scatter(mean_lon, mean_lat, s=100, color='k', marker='x')
    plt.title("Segments used in the computation")
    plt.savefig('%s_%04d.png' % (prefix, nb_fig))
    plt.close()
