import numpy as np
from math import sqrt, cos, sin, asin, radians
from mpl_toolkits.basemap import Basemap
import matplotlib.pylab as plt

from mod_constant import *


def find_nearest_index(array, value):
    """
    Given an array and a value, return nearest value index in array
    :param array:
    :param value:
    :return:
    """
    index = (np.abs(array - value)).argmin()
    return index


def find_nearest_index_lonlat(array_lon, array_lat, value_lon, value_lat):
    """
    Function find nearest index
    :param array_lon:
    :param array_lat:
    :param value_lon:
    :param value_lat:
    :return:
    """
    idy = np.argmin(np.abs(array_lat - value_lat), axis=0)
    idx = np.argmin(np.abs(array_lon[idy[0], :] - value_lon))

    return idx, idy[0]


def find_nearest_common_index(array_lon, array_lat, value_lon, value_lat):
    """
    Function find common nearest index
    :param array_lon:
    :param array_lat:
    :param value_lon:
    :param value_lat:
    :return:
    """
    idx = np.argsort(np.abs(array_lon - value_lon), axis=0)
    idy = np.argsort(np.abs(array_lat - value_lat), axis=0)

    ii = 0
    location_idx = []
    location_idy = []
    for value in idx:
        location_idx.append(ii)
        ii += 1
        location_idy.append(np.int(np.where(idy == value)[0]))

    min_index = np.where(np.array(location_idx) + np.array(location_idy) ==
                         np.min(np.array(location_idx) + np.array(location_idy)))[0]

    return idx[min_index][0]


def change_in_latitude(kilometers):
    """
    Given a distance north, return the change in latitude.
    :param kilometers:
    :return:
    """
    return (kilometers/earth_radius)*radians_to_degrees


def change_in_longitude(latitude, kilometers):
    """
    Given a latitude and a distance west, return the change in longitude.
    :param latitude:
    :param kilometers:
    :return:
    """
    # Find the radius of a circle around the earth at given latitude.
    r = earth_radius*cos(latitude*degrees_to_radians)
    return (kilometers/r)*radians_to_degrees


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    :param lon1:
    :param lat1:
    :param lon2:
    :param lat2:
    :return:
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    ca1 = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    cc1 = 2 * asin(sqrt(ca1))
    return cc1 * earth_radius


def selection_in_latlonbox(lon_array, lat_array, lon_min, lon_max, lat_min, lat_max,
                           greenwhich_start=False, debug=False):
    """
    Selection of segment within a box define by lon_min, lon_max, lat_min, lat_max
    :param lon_array:
    :param lat_array:
    :param lon_min:
    :param lon_max:
    :param lat_min:
    :param lat_max:
    :param greenwhich_start:
    :param debug:
    :return:
    """

    selected_lat_index = np.where(np.logical_and(lat_array >= lat_min, lat_array <= lat_max))[0]

    if greenwhich_start:

        if lon_min < 0.:
            selected_lon_index1 = np.where(np.logical_and(lon_array >= 0, lon_array <= lon_max))[0]
            selected_lon_index2 = np.where(np.logical_and(lon_array >= lon_min + 360., lon_array <= 360))[0]
            selected_lon_index = np.concatenate((selected_lon_index1, selected_lon_index2))
        elif lon_max > 360.:
            selected_lon_index1 = np.where(np.logical_and(lon_array >= lon_min, lon_array <= 360.))[0]
            selected_lon_index2 = np.where(np.logical_and(lon_array >= 0., lon_array <= lon_max - 360))[0]
            selected_lon_index = np.concatenate((selected_lon_index1, selected_lon_index2))
        else:
            selected_lon_index = np.where(np.logical_and(lon_array >= lon_min, lon_array <= lon_max))[0]

    else:

        if lon_min < -180.:
            selected_lon_index1 = np.where(np.logical_and(lon_array >= -180, lon_array <= lon_max))[0]
            selected_lon_index2 = np.where(np.logical_and(lon_array >= lon_min + 360., lon_array <= 180.))[0]
            selected_lon_index = np.concatenate((selected_lon_index1, selected_lon_index2))
        elif lon_max > 180.:
            selected_lon_index1 = np.where(np.logical_and(lon_array >= lon_min, lon_array <= 180.))[0]
            selected_lon_index2 = np.where(np.logical_and(lon_array >= -180., lon_array <= lon_max - 360))[0]
            selected_lon_index = np.concatenate((selected_lon_index1, selected_lon_index2))
        else:
            selected_lon_index = np.where(np.logical_and(lon_array >= lon_min, lon_array <= lon_max))[0]

    selected_index = np.intersect1d(selected_lon_index, selected_lat_index)

    if debug:
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
        x, y = bmap(lon_array[selected_index], lat_array[selected_index])
        bmap.scatter(x, y, s=0.1, marker="o", color='r')
        plt.title("Point selected to estimate main value at lon=%s and lat =%s " %
                  (str(0.5*(lon_min+lon_max)), str(0.5*(lat_min+lat_max))))
        plt.show()
        # plt.savefig('%s_%04d.png' % ("box_selection", nb_fig_selection))
        # plt.close()

        # nb_fig_selection += 1

    return selected_index
