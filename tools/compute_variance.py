import numpy as np
from netCDF4 import Dataset
from sys import argv
import matplotlib.pylab as plt

input_file = argv[1]
input_varname = argv[2]
output_file = argv[3]

# Read inpt PSD and frequency
nc = Dataset(input_file, 'r')
freq = nc.variables['freq'][:]
lon = nc.variables['lon'][:]
lat = nc.variables['lat'][:]
psd = nc.variables[input_varname][:, :, :]
nc.close()

variance = np.zeros((lat.size, lon.size))


def compute_variance(array, wavenumber):
    """

    :param array:
    :param wavenumber:
    :return:
    """

    dx = np.diff(wavenumber)

    return np.trapz(array, x=wavenumber, dx=dx)


for jj in range(lat.size):
    for ii in range(lon.size):
        if not np.ma.is_masked(psd[:, jj, ii]):
            variance[jj, ii] = compute_variance(psd[:, jj, ii], freq)

nc_out = Dataset(output_file, 'w', format='NETCDF4')
nc_out.createDimension('lat', lat.size)
nc_out.createDimension('lon', lon.size)

lat_out = nc_out.createVariable('lat', 'f8', ('lat',))
lat_out[:] = lat
lon_out = nc_out.createVariable('lon', 'f8', ('lon',))
lon_out[:] = lon

variance_out = nc_out.createVariable('variance', 'f8', ('lat', 'lon'))
variance_out.coordinates = "lat lon"
variance_out.units = 'm2'
variance_out.long_name = "variance as integral of psd function"
variance_out[:, :] = np.ma.masked_where(variance == 0., variance)

nc_out.close()
