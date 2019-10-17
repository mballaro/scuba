from mod_geo import *
import numpy as np
import scipy.signal
import matplotlib.pylab as plt
from mod_constant import *
import scipy.stats
import xarray as xr
import datetime


def timeseries_statistic_computation(config, data_ref, data_study, time):
    """

    :param config:
    :param data_ref:
    :param data_study:
    :param time:
    :return:
    """

    list_nobs = []
    list_min = []
    list_max = []
    list_mean = []
    list_variance = []
    list_skewness = []
    list_kurtosis = []
    list_rmse = []
    list_mae = []
    list_correlation = []
    list_pvalue = []
    list_mean_ref = []
    list_variance_ref = []
    list_mean_study = []
    list_variance_study = []

    data = data_ref - data_study
    vtime = np.arange(np.min(time), np.max(time) + 1., 1.)

    for itime in vtime:
        time_min = itime - 0.5
        time_max = itime + 0.5
        selected_time_index = np.where(np.logical_and(time >= time_min, time <= time_max))[0]
        
        if len(selected_time_index) > 0:
            selected_data = np.ma.masked_where(data[selected_time_index].flatten() > 1.E10,
                                               data[selected_time_index].flatten())
            selected_data_ref = np.ma.masked_where(data_ref[selected_time_index].flatten() > 1.E10,
                                                   data_ref[selected_time_index].flatten())
            selected_data_study = np.ma.masked_where(data_study[selected_time_index].flatten() > 1.E10,
                                                     data_study[selected_time_index].flatten())
       
            nobs = np.ma.count(selected_data)
            variance = np.ma.var(selected_data)
            minmax = [np.ma.min(selected_data), np.ma.max(selected_data)]
            mean = np.ma.mean(selected_data)
            skewness = scipy.stats.mstats.skew(selected_data)
            kurtosis = scipy.stats.mstats.kurtosis(selected_data)
            rmse = np.ma.sqrt((selected_data ** 2).mean())
            mae = np.ma.mean(np.absolute(selected_data))
            correlation, pvalue = scipy.stats.mstats.pearsonr(selected_data_ref, selected_data_study)
            variance_ref = np.ma.var(selected_data_ref)
            variance_study = np.ma.var(selected_data_study)
            mean_ref = np.ma.mean(selected_data_ref)
            mean_study = np.ma.mean(selected_data_study)

            list_nobs.append(nobs)
            list_min.append(minmax[0])
            list_max.append(minmax[1])
            list_mean.append(mean)
            list_variance.append(variance)
            list_skewness.append(skewness)
            list_kurtosis.append(kurtosis)
            list_rmse.append(rmse)
            list_mae.append(mae)
            list_correlation.append(correlation)
            list_pvalue.append(pvalue)
            list_mean_ref.append(mean_ref)
            list_variance_ref.append(variance_ref)
            list_mean_study.append(mean_study)
            list_variance_study.append(variance_study)


        else:

            list_nobs.append(0)
            list_min.append(0.)
            list_max.append(0.)
            list_mean.append(0.)
            list_variance.append(0.)
            list_skewness.append(0.)
            list_kurtosis.append(0.)
            list_rmse.append(0.)
            list_mae.append(0.)
            list_correlation.append(0.)
            list_pvalue.append(0.)
            list_mean_ref.append(0.)
            list_variance_ref.append(0.)
            list_mean_study.append(0.)
            list_variance_study.append(0.)

    nobs = np.asarray(list_nobs)
    vmin = np.asarray(list_min)
    vmax = np.asarray(list_max)
    mean = np.asarray(list_mean)
    variance = np.asarray(list_variance)
    skewness = np.asarray(list_skewness)
    kurtosis = np.asarray(list_kurtosis)
    rmse = np.asarray(list_rmse)
    mae = np.asarray(list_mae)
    correlation = np.asarray(list_correlation)
    pvalue = np.asarray(list_pvalue)
    mean_ref = np.asarray(list_mean_ref)
    variance_ref = np.asarray(list_variance_ref)
    mean_study = np.asarray(list_mean_study)
    variance_study = np.asarray(list_variance_study)

    return nobs, vmin, vmax, mean, variance, skewness, kurtosis, rmse, mae, correlation, pvalue, vtime, variance_ref, \
           variance_study, mean_ref, mean_study


def statistic_computation(config, data_ref, data_study, lon, lat):
    """

    :param config:
    :param data_ref:
    :param data_study:
    :param lon:
    :param lat:
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

    data = data_ref - data_study

    list_nobs = []
    list_min = []
    list_max = []
    list_mean = []
    list_variance = []
    list_skewness = []
    list_kurtosis = []
    list_rmse = []
    list_mae = []
    list_correlation = []
    list_pvalue = []
    list_variance_ref = []
    list_variance_study = []
    list_mean_ref = []
    list_mean_study = []

    # Loop over output lon/lat boxes and selection of the segment within the box plus/minus delta_lon/lat
    for ilat in grid_lat:

        # print(ilat)

        lat_min = ilat - 0.5*delta_lat
        lat_max = ilat + 0.5*delta_lat

        selected_lat_index = np.where(np.logical_and(lat >= lat_min, lat <= lat_max))[0]
        data_tmp = data[selected_lat_index]
        data_ref_tmp = data_ref[selected_lat_index]
        data_study_tmp = data_study[selected_lat_index]

        for ilon in grid_lon % 360.:

            lon_min = ilon - 0.5*delta_lon
            lon_max = ilon + 0.5*delta_lon

            if (lon_min < 0.) and (lon_max > 0.):
                selected_index = np.where(np.logical_or(lon[selected_lat_index] % 360. >= lon_min + 360.,
                                                          lon[selected_lat_index] % 360. <= lon_max))[0]
            elif (lon_min > 0.) and (lon_max > 360.):
                selected_index = np.where(np.logical_or(lon[selected_lat_index] % 360. >= lon_min,
                                                          lon[selected_lat_index] % 360. <= lon_max - 360.))[0]
            else:
                selected_index = np.where(np.logical_and(lon[selected_lat_index] % 360. >= lon_min,
                                                           lon[selected_lat_index] % 360. <= lon_max))[0]

            if len(selected_index) > 0:
                selected_data = np.ma.masked_where(data_tmp[selected_index].flatten() > 1.E10,
                                                           data_tmp[selected_index].flatten())
                selected_data_ref = np.ma.masked_where(data_ref_tmp[selected_index].flatten() > 1.E10,
                                                           data_ref_tmp[selected_index].flatten())
                selected_data_study = np.ma.masked_where(data_study_tmp[selected_index].flatten() > 1.E10,
                                                           data_study_tmp[selected_index].flatten())

                # nobs, minmax, mean, variance, skewness, kurtosis = scipy.stats.describe(selected_data)
                nobs = np.ma.count(selected_data)
                variance = np.ma.var(selected_data)
                minmax = [np.ma.min(selected_data), np.ma.max(selected_data)]
                mean = np.ma.mean(selected_data)
                skewness = scipy.stats.mstats.skew(selected_data)
                kurtosis = scipy.stats.mstats.kurtosis(selected_data)
                rmse = np.ma.sqrt((selected_data ** 2).mean())
                mae = np.ma.mean(np.absolute(selected_data))
                correlation, pvalue = scipy.stats.mstats.pearsonr(selected_data_ref, selected_data_study)
                variance_ref = np.ma.var(selected_data_ref)
                variance_study = np.ma.var(selected_data_study)
                mean_ref = np.ma.mean(selected_data_ref)
                mean_study = np.ma.mean(selected_data_study)

                list_nobs.append(nobs)
                list_min.append(minmax[0])
                list_max.append(minmax[1])
                list_mean.append(mean)
                list_variance.append(variance)
                list_skewness.append(skewness)
                list_kurtosis.append(kurtosis)
                list_rmse.append(rmse)
                list_mae.append(mae)
                list_correlation.append(correlation)
                list_pvalue.append(pvalue)
                list_variance_ref.append(variance_ref)
                list_variance_study.append(variance_study)
                list_mean_ref.append(mean_ref)
                list_mean_study.append(mean_study)

            else:

                list_nobs.append(0)
                list_min.append(0.)
                list_max.append(0.)
                list_mean.append(0.)
                list_variance.append(0.)
                list_skewness.append(0.)
                list_kurtosis.append(0.)
                list_rmse.append(0.)
                list_mae.append(0.)
                list_correlation.append(0.)
                list_pvalue.append(0.)
                list_variance_ref.append(0.)
                list_variance_study.append(0.)
                list_mean_ref.append(0.)
                list_mean_study.append(0.)

    nobs = np.asarray(list_nobs).reshape(grid_lat.size, grid_lon.size)
    min = np.asarray(list_min).reshape(grid_lat.size, grid_lon.size)
    max = np.asarray(list_max).reshape(grid_lat.size, grid_lon.size)
    mean = np.asarray(list_mean).reshape(grid_lat.size, grid_lon.size)
    variance = np.asarray(list_variance).reshape(grid_lat.size, grid_lon.size)
    skewness = np.asarray(list_skewness).reshape(grid_lat.size, grid_lon.size)
    kurtosis = np.asarray(list_kurtosis).reshape(grid_lat.size, grid_lon.size)
    rmse = np.asarray(list_rmse).reshape(grid_lat.size, grid_lon.size)
    mae = np.asarray(list_mae).reshape(grid_lat.size, grid_lon.size)
    correlation = np.asarray(list_correlation).reshape(grid_lat.size, grid_lon.size)
    pvalue = np.asarray(list_pvalue).reshape(grid_lat.size, grid_lon.size)
    variance_ref = np.asarray(list_variance_ref).reshape(grid_lat.size, grid_lon.size)
    variance_study = np.asarray(list_variance_study).reshape(grid_lat.size, grid_lon.size)
    mean_ref = np.asarray(list_mean_ref).reshape(grid_lat.size, grid_lon.size)
    mean_study = np.asarray(list_mean_study).reshape(grid_lat.size, grid_lon.size)

    return nobs, min, max, mean, variance, skewness, kurtosis, rmse, mae, correlation, pvalue, variance_ref, \
           variance_study, mean_ref, mean_study
