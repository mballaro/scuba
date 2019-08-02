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
psd_ref = nc.variables['psd_ref'][:, :]
psd_diff = nc.variables['psd_diff'][:, :]
nc.close()

print(np.round(1./freq, 1))

ratio = psd_diff/psd_ref

resolution = np.empty(lat.size)


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

    # if flag_multiple_crossing:
    #     print(resolution_scale)
    #     plt.plot(1./wavenumber, array)
    #     plt.show()

    return resolution_scale, flag_multiple_crossing


for jj in range(lat.size):
    if not np.ma.is_masked(ratio[jj, :]):
        resolution[jj], flag = compute_crossing(ratio[jj, :], freq)
        if flag:
            print('computation at lon = %s and lat = %s' % (str(lon[jj]), str(lat[jj])))

# resolution = np.ma.masked_invalid(np.ma.masked_outside(resolution, 8., 50.))

print('MEAN TEMPORAL RESOLUTION = %s days' % str(np.ma.mean(resolution)))

nc_out = Dataset(output_file, 'w', format='NETCDF4')
nc_out.createDimension('sensor', lat.size)
lat_out = nc_out.createVariable('lat', 'f8', 'sensor')
lat_out[:] = lat
lon_out = nc_out.createVariable('lon', 'f8', 'sensor')
lon_out[:] = lon
zero_crossing_out = nc_out.createVariable('effective_resolution', 'f8', 'sensor')
zero_crossing_out.units = 'days'
zero_crossing_out.coordinates = "sensor"
zero_crossing_out.long_name = "0.5 of Ratio"
zero_crossing_out[:] = np.ma.masked_where(resolution == 0., resolution)
nc_out.close()
