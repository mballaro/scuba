import numpy as np
from netCDF4 import Dataset
from sys import argv
import matplotlib.pylab as plt

input_file = argv[1]
output_file = argv[2]

# Read inpt PSD and frequency
nc = Dataset(input_file, 'r')
freq = nc.variables['wavenumber'][:]
lon = nc.variables['lon'][:]
lat = nc.variables['lat'][:]
psd_ref = nc.variables['psd_ref'][:, :, :]
psd_study = nc.variables['psd_study'][:, :, :]
coherence = nc.variables['coherence'][:, :, :]
nc.close()

ratio = psd_study/psd_ref

skill_score = 2.0*(1.0 + coherence) / (ratio + 1/ratio)**2

resolution = np.empty((lat.size, lon.size))


def compute_crossing(array, wavenumber, threshold=0.5):
    """

    :param array:
    :param wavenumber:
    :param threshold:
    :return:
    """

    flag_multiple_crossing = False
    zero_crossings = np.where(np.diff(np.sign(array - threshold)))[0]
    if len(zero_crossings) > 1:
        print('Multiple crossing', len(zero_crossings))
        flag_multiple_crossing = True
        # MB add for large scale bais
        zero_crossings[0] = zero_crossings[-1]

    if len(zero_crossings) > 0:
        if zero_crossings[0] + 1 < array.size:

            array1 = array[zero_crossings[0]] - threshold
            array2 = array[zero_crossings[0] + 1] - threshold
            dist1 = np.log(wavenumber[zero_crossings[0]])
            dist2 = np.log(wavenumber[zero_crossings[0] + 1])
            log_wavenumber_crossing = dist1 - array1 * (dist1 - dist2) / (array1 - array2)
            resolution_scale = 1. / np.exp(log_wavenumber_crossing)

        else:
            resolution_scale = 0.

    else:
        resolution_scale = 0.

    return resolution_scale, flag_multiple_crossing


for jj in range(lat.size):
    for ii in range(lon.size):
        if not np.ma.is_masked(skill_score[:, jj, ii]):
            resolution[jj, ii], flag = compute_crossing(skill_score[:, jj, ii], freq)
            if flag:
                print('computation at lon = %s and lat = %s' % (str(lon[ii]), str(lat[jj])))


nc_out = Dataset(output_file, 'w', format='NETCDF4')
nc_out.createDimension('lat', lat.size)
nc_out.createDimension('lon', lon.size)
nc_out.createDimension('wavenumber', freq.size)

freq_out = nc_out.createVariable('wavenumber', 'f8', ('wavenumber',))
freq_out[:] = np.ma.masked_invalid(freq)

skill_score_out = nc_out.createVariable('skill_score', 'f8', ('wavenumber', 'lat', 'lon'))
skill_score_out[:, :, :] = np.ma.masked_invalid(skill_score)

lat_out = nc_out.createVariable('lat', 'f8', ('lat',))
lat_out[:] = lat
lon_out = nc_out.createVariable('lon', 'f8', ('lon',))
lon_out[:] = lon

zero_crossing_out = nc_out.createVariable('effective_resolution', 'f8', ('lat', 'lon'))
zero_crossing_out.units = 'km'
zero_crossing_out.coordinates = "lat lon"
zero_crossing_out.long_name = "0.5 of Skill score function"
zero_crossing_out[:, :] = np.ma.masked_where(resolution == 0., resolution)

nc_out.close()
