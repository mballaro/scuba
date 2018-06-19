#!/usr/bin/python
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic
from math import sin, cos, atan2, pi, sqrt
from netCDF4 import Dataset
from sys import argv, exit
from numpy import arange, zeros, meshgrid, ma
from matplotlib import rc, colors
#rc('text', usetex=True)


input_file = argv[1]

nc = Dataset(input_file)
effective_resolution = np.ma.masked_invalid(nc.variables['effective_resolution'][:, :])
useful_resolution = np.ma.masked_invalid(nc.variables['useful_resolution'][:, :])
lat = nc.variables['lat2D'][:, :]
lon = nc.variables['lon2D'][:, :]
nc.close()

effective_resolution = np.ma.masked_where(effective_resolution==0., effective_resolution)
useful_resolution = np.ma.masked_where(useful_resolution==0., useful_resolution)

fig = plt.figure()
bmap = Basemap(projection='cyl',
               llcrnrlat=np.min(lat),
               urcrnrlat=np.max(lat),
               llcrnrlon=np.min(lon),
               urcrnrlon=np.max(lon),
               resolution='i')

ax = fig.add_subplot(211)
bmap.drawcoastlines(zorder=5, linewidth=0.25)
bmap.fillcontinents(color='grey', lake_color='white')
bmap.drawparallels(arange(-90, 90, 30), labels=[1, 0, 0, 0], fontsize=20, dashes=[4, 4], linewidth=0.25)
bmap.drawmeridians(arange(-180, 360, 45), labels=[0, 0, 0, 1], fontsize=20, dashes=[4, 4], linewidth=0.25)
C1 = bmap.contourf(lon, lat, effective_resolution, np.arange(100,800,50), extend='both', cmap='Spectral_r')
C2 = bmap.contour(lon, lat, effective_resolution, np.arange(100,800,50), colors='grey', alpha=0.5)
cbar = bmap.colorbar(C1, location='bottom', pad='15%')
cbar.set_label('Effective Resolution (km)', fontweight='bold')

ax = fig.add_subplot(212)
bmap.drawcoastlines(zorder=5, linewidth=0.25)
bmap.fillcontinents(color='grey', lake_color='white')
bmap.drawparallels(arange(-90, 90, 30), labels=[1, 0, 0, 0], fontsize=20, dashes=[4, 4], linewidth=0.25)
bmap.drawmeridians(arange(-180, 360, 45), labels=[0, 0, 0, 1], fontsize=20, dashes=[4, 4], linewidth=0.25)
C3 = bmap.contourf(lon, lat, useful_resolution, np.arange(100,800,50), extend='both', cmap='Spectral_r')
C4 = bmap.contour(lon, lat, useful_resolution, np.arange(100,800,50), colors='grey', alpha=0.5)
cbar = bmap.colorbar(C3, location='bottom', pad='15%')
cbar.set_label('Useful Resolution (km)', fontweight='bold')

plt.show()
