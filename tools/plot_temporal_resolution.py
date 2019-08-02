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
data = nc.variables['effective_resolution'][:]
lon = nc.variables['lon'][:]
lat = nc.variables['lat'][:]
nc.close()

cmap = plt.cm.Spectral_r

cmaplist = [cmap(i) for i in range(cmap.N)]

cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
# define the bins and normalize
bounds = np.linspace(0, 42, 7)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

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

pc1 = ax.scatter(lon, lat, c=data, s=18, cmap=cmap, norm=norm, zorder=10, vmin=0, vmax=42, transform=ccrs.Geodetic(),
                 linewidths=0.5, edgecolors='k')
plt.title('Effective temporal resolution')
cb = plt.colorbar(pc1, orientation='vertical', fraction=0.024, pad=0.01)
cb.set_ticks(np.arange(0, 49, 7))
cb.set_ticklabels(["0", "7days", "14days", "21days", "28days", "35days", "42days"])
plt.show()
