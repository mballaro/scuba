import cartopy
import cartopy.crs as ccrs
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

import matplotlib.pylab as plt
from sys import argv
from netCDF4 import Dataset
import matplotlib as mpl
mpl.rc('text', usetex=True)

mpl.rcParams['savefig.dpi'] = 300

nc = Dataset(argv[1], 'r')
data = nc.variables['effective_resolution'][:, :]
lon = nc.variables['lon'][:]
lat = nc.variables['lat'][:]
nc.close()
data = np.ma.masked_outside(data, 10, 1000)
lon = np.roll(lon, 179)
lon = np.where(lon > 180, lon - 360., lon)
lon2d, lat2d = np.meshgrid(lon, lat)

plt.figure()

projection = ccrs.PlateCarree(central_longitude=180)
ax = plt.axes(projection=projection)
ax.set_global()
ax.coastlines(resolution='50m', lw=0.5, zorder=4)
gl = ax.gridlines(draw_labels=True, linestyle='--', xlocs=[-120, -60, 0, 60, 120, 180, 240], zorder=6)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
# ax.add_feature(cartopy.feature.LAND, facecolor='0.5', zorder=3)

ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='face',
                                            facecolor=cfeature.COLORS['land_alt1']),
               zorder=3)

pc1 = ax.contourf(lon2d, lat2d, data, np.arange(100, 750, 50), cmap='Spectral_r', extend='both', transform=projection)
pc2 = ax.contour(lon2d, lat2d, data, np.arange(100, 750, 50), colors='0.5', linewidths=0.5, transform=projection)
cb = plt.colorbar(pc1, orientation='vertical', fraction=0.022, pad=0.01)
plt.title('Effective spatial resolution')
cb.set_ticks(np.arange(100, 750, 50))
cb.set_ticklabels(["100km", "150km", "200km", "250km", "300km", "350km", "400km", "450km", "500km", "550km",
                   "600km", "650km", "700km"])

plt.show()
