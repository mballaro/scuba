
import numpy as np
import matplotlib.pylab as plt
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.interpolate import interp1d
from sys import exit
from mod_constant import *
from mod_io import *
from mod_geo import *


def slicing(lon, lat, vector, direction, npt, segment_overlapping, output_seg_x=None):
    """

    :param lon:
    :param lat:
    :param vector:
    :param direction:
    :param npt:
    :param segment_overlapping:
    :param output_seg_x:
    :return:
    """

    list_sla_ref_segment = []
    list_lon_segment = []
    list_lat_segment = []

    if vector.size >= npt:

        indi = np.where(vector.filled(-999.) == -999.)[0]
        indi = np.append(indi, vector.size)

        if indi.size:
            segment_lenght = np.insert(np.diff(indi), [0], indi[0])
        else:
            indi = [0, vector.size]
            segment_lenght = np.insert(np.diff(indi), [0], indi[0])

        selected_segment = np.where(segment_lenght >= npt)[0]

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

                for sub_segment_point in range(start_point, end_point - npt, int(npt * segment_overlapping)):

                    msla_ref_segment = vector[sub_segment_point:sub_segment_point + npt]

                    if direction == "zonal":
                        # coordinates segment center
                        mean_lon_sub_segment = np.mean(lon[sub_segment_point:sub_segment_point + npt])
                        mean_lat_sub_segment = lat

                        # Interpolation of sla onto longest segment
                        x_seg = np.linspace(0., 1., msla_ref_segment.size)
                        f_interp = interp1d(x_seg, msla_ref_segment)
                        interpolated_seg_value = f_interp(output_seg_x)
                        interpolated_seg_value = np.ma.masked_invalid(interpolated_seg_value)
                        # msla_ref_segment = np.ma.masked_where(np.abs(interpolated_seg_value) > 500.,
                        #                                       interpolated_seg_value)
                        msla_ref_segment = np.ma.masked_outside(interpolated_seg_value, -500., 500.)

                    elif direction == "meridional":
                        # coordinates segment center
                        mean_lon_sub_segment = lon
                        mean_lat_sub_segment = np.mean(lat[sub_segment_point:sub_segment_point + npt])

                    else:
                        mean_lon_sub_segment = []
                        mean_lat_sub_segment = []
                        print('Unknown direction in slicing (zonal / meridional ?)')
                        exit(0)

                    if not np.ma.is_masked(msla_ref_segment):
                        list_sla_ref_segment.append(msla_ref_segment)
                        list_lon_segment.append(mean_lon_sub_segment)
                        list_lat_segment.append(mean_lat_sub_segment)

    return list_lon_segment, list_lat_segment, list_sla_ref_segment


def compute_segment_maps(array_ref, lon, lat, direction, config):
    """
    Segment computation in zonal and meridional direction
    :param array_ref:
    :param lon:
    :param lat:
    :param direction:
    :param config:
    :return:
    """

    lenght_scale = config['properties']['spectral_parameters']['lenght_scale']
    segment_overlapping = config['properties']['spectral_parameters']['segment_overlapping']

    list_lat_segment = []
    list_lon_segment = []
    list_sla_segment = []

    min_delta_x = 0
    delta_km = 0
    npt = 0

    if direction == "zonal":
        direction_dim = lat.size
        all_npt_x = []
        # LOOP along latitude to get max segment lenght in x-direction (true segment length is changing with lat)
        for jj in range(lat.size):
            if np.abs(lat[jj]) < 90:
                delta_x = haversine(lon[0], lat[jj], lon[1], lat[jj])
                all_npt_x.append(int(lenght_scale / delta_x))
        max_number = np.max(all_npt_x)
        output_seg_x = np.linspace(0., 1., max_number)
        min_delta_x = lenght_scale / max_number

    elif direction == "meridional":
        direction_dim = lon.size
        output_seg_x = []

    else:
        direction_dim = []
        output_seg_x = []
        print('Unknown direction (zonal / meridional ?)')
        exit(0)

    for tt in range(array_ref[:, 0, 0].size):

        for jj in range(direction_dim):

            if direction == "zonal":
                delta_km = haversine(lon[0], lat[jj], lon[1], lat[jj])
                npt = int(lenght_scale / delta_km)
                lon_segment, lat_segment, sla_segment = slicing(lon, lat[jj], array_ref[tt, jj, :],
                                                                direction, npt, segment_overlapping, output_seg_x)

            elif direction == "meridional":
                delta_km = haversine(lon[jj], lat[0], lon[jj], lat[1])
                npt = int(lenght_scale / delta_km)
                lon_segment, lat_segment, sla_segment = slicing(lon[jj], lat, array_ref[tt, :, jj],
                                                                direction, npt, segment_overlapping)

            else:
                lon_segment = []
                lat_segment = []
                sla_segment = []
                delta_km = []
                print('Unknown direction (zonal / meridional ?)')
                exit(0)

            list_lon_segment = list_lon_segment + lon_segment
            list_lat_segment = list_lat_segment + lat_segment
            list_sla_segment = list_sla_segment + sla_segment

    if direction == "zonal":
        delta_km = min_delta_x
    else:
        pass

    return list_sla_segment, list_lon_segment, list_lat_segment, delta_km, npt


def compute_segment_alongtrack(sla, lon_along_track, lat_along_track, time_along_track, config, msla=None, debug=False):
    """
    Segment computation in alongtrack direction
    :param sla:
    :param lon_along_track:
    :param lat_along_track:
    :param time_along_track:
    :param config
    :param msla:
    :param debug:
    :return:
    """

    mission = config['inputs']['mission']
    mission_management = config['properties']['mission_management']
    lenght_scale = config['properties']['spectral_parameters']['lenght_scale']
    segment_overlapping = config['properties']['spectral_parameters']['segment_overlapping']

    list_lat_segment = []
    list_lon_segment = []
    list_sla_segment = []
    # list_crosscorrelation_segment = []
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

                msla_segment = []

                if msla is not None:
                    msla_segment = np.ma.masked_invalid(msla[sub_segment_point:sub_segment_point + npt])
                    if np.ma.is_masked(msla_segment):
                        sla_segment = np.ma.compressed(np.ma.masked_where(np.ma.is_masked(msla_segment), sla_segment))
                        msla_segment = np.ma.compressed(msla_segment)

                if sla_segment.size > 0:
                    list_sla_segment.append(sla_segment)
                    list_lon_segment.append(mean_lon_sub_segment)
                    list_lat_segment.append(mean_lat_sub_segment)
                    if msla is not None:
                        # list_crosscorrelation_segment.append(cross_correlation)
                        list_msla_segment.append(msla_segment)

    if msla is not None:
        return list_sla_segment, list_lon_segment, list_lat_segment, delta_x, npt, list_msla_segment

    else:
        return list_sla_segment, list_lon_segment, list_lat_segment, delta_x, npt


def compute_segment_tide_gauge_tao(sla_tg, time_tg, lenght_scale, delta_t, segment_overlapping, msla=None):
    """

    :param sla_tg:
    :param time_tg:
    :param lenght_scale:
    :param delta_t:
    :param segment_overlapping:
    :param msla:
    :return:
    """

    list_sla_segment = []
    # list_crosscorrelation_segment = []
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
    selected_track_segment = np.where(np.asarray(track_segment_lenght) >= npt)[0]

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

                msla_segment = []

                if msla is not None:
                    msla_segment = np.ma.masked_invalid(msla[sub_segment_point:sub_segment_point + npt])
                    if np.ma.is_masked(msla_segment):
                        sla_segment = np.ma.compressed(np.ma.masked_where(np.ma.is_masked(msla_segment), sla_segment))
                        msla_segment = np.ma.compressed(msla_segment)

                if sla_segment.size > 0:
                    list_sla_segment.append(sla_segment)
                    # list_crosscorrelation_segment.append(cross_correlation)
                    if msla is not None:
                        list_msla_segment.append(msla_segment)

    if msla is not None:
        return list_sla_segment, list_msla_segment

    else:
        return list_sla_segment


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

    plt.subplots()
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    ax.coastlines()
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND)
    ax.set_xticks(np.linspace(np.min(lon), np.max(lon), 5), crs=projection)
    ax.set_yticks(np.linspace(np.min(lat), np.max(lat), 5), crs=projection)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.scatter(xlon_segment, xlat_segment, s=0.1, marker="o", color='g', transform=projection)
    ax.scatter(xlon_subsegment, xlat_subsegment, s=0.1, marker="o", color='r', transform=projection)
    ax.scatter(mean_lon, mean_lat, s=100, color='k', marker='x', transform=projection)
    ax.set_title("Segments used in the computation", fontweight='bold')
    plt.savefig('%s_%04d.png' % (prefix, nb_fig))
    plt.close()
