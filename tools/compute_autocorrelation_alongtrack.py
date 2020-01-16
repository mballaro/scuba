import numpy as np
from netCDF4 import Dataset
from sys import argv
import matplotlib.pylab as plt

input_file = argv[1]
input_varname = argv[2]
output_file = argv[3]


# Read inpt PSD and frequency
nc = Dataset(input_file, 'r')
freq = nc.variables['wavenumber'][:]
lon = nc.variables['lon'][:]
lat = nc.variables['lat'][:]
psd = nc.variables[input_varname][:, :, :]
nc.close()

autocorrelation = np.zeros((int(0.5*freq.size), lat.size, lon.size))
distance = np.zeros(int(0.5*freq.size))
zero_crossing = np.zeros((lat.size, lon.size))


def compute_autocorrelation(power_spectrum, wavenumber, display=False):
    """
    Compute autocorrelation function from power spectrum
    :param power_spectrum:
    :param wavenumber:
    :param display:
    :return:
    """

    autocorr = np.fft.ifft(power_spectrum).real
    autocorr /= autocorr[0]
    delta_f = 1 / (wavenumber[-1] - wavenumber[-2])
    dist = np.linspace(0, int(delta_f/2), int(autocorr.size / 2))

    if display:
        plt.plot(dist, autocorr[:int(autocorr.size/2)], lw=2, color='r')
        plt.xlabel("DISTANCE (km)")
        plt.ylabel("AUTOCORRELATION")
        plt.axhline(y=0., color='k', lw=2)
        plt.show()

    return autocorr[:int(autocorr.size/2)], dist


def compute_crossing(array, xdistance):
    """

    :param array:
    :param xdistance:
    :return:
    """

    zero_crossings = np.where(np.diff(np.sign(array)))[0]

    if len(zero_crossings) > 0:
        if zero_crossings[0] + 1 < array.size:
            array1 = array[zero_crossings[0]]
            array2 = array[zero_crossings[0] + 1]
            dist1 = xdistance[zero_crossings[0]]
            dist2 = xdistance[zero_crossings[0] + 1]
            distance_crossing = dist1 - array1 * (dist1 - dist2)/(array1-array2)

        else:
            distance_crossing = 0.

    else:
        distance_crossing = 0.

    return distance_crossing


for jj in range(np.shape(autocorrelation)[1]):
    for ii in range(np.shape(autocorrelation)[2]):
        if not np.ma.is_masked(psd[:, jj, ii]):
            autocorrelation[:, jj, ii], distance = compute_autocorrelation(psd[:, jj, ii], freq, display=False)
            zero_crossing[jj, ii] = compute_crossing(autocorrelation[:, jj, ii], distance)

nc_out = Dataset(output_file, 'w', format='NETCDF4')
nc_out.createDimension('dist', distance.size)
nc_out.createDimension('lat', lat.size)
nc_out.createDimension('lon', lon.size)

lat_out = nc_out.createVariable('lat', 'f8', ('lat',))
lat_out[:] = lat
lon_out = nc_out.createVariable('lon', 'f8', ('lon',))
lon_out[:] = lon

autocorrelation_distance = nc_out.createVariable('dist', 'f8', 'dist')
autocorrelation_distance.units = "km"
autocorrelation_distance[:] = distance

autocorrelation_out = nc_out.createVariable('autocorrelation', 'f8', ('dist', 'lat', 'lon'))
autocorrelation_out.coordinates = "distance lat lon"
autocorrelation_out.long_name = "autocorrelation function of variable %s in file %s " \
                                "computed from power spectrum density" \
                                % (input_varname, input_file)
autocorrelation_out[:, :, :] = np.ma.masked_where(autocorrelation == 0., autocorrelation)

zero_crossing_out = nc_out.createVariable('zero_crossing', 'f8', ('lat', 'lon'))
zero_crossing_out.coordinates = "lat lon"
zero_crossing_out.long_name = "zero_crossing of autocorrelation function"
zero_crossing_out[:, :] = np.ma.masked_where(zero_crossing == 0., zero_crossing)

nc_out.close()
