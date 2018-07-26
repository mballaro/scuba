"""
Display Spectral Taylor Diagram analysis for spectral lovers...
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic
from math import sin, cos, atan2, pi, sqrt
from netCDF4 import Dataset
from sys import argv, path
from matplotlib import rc, colors

path.insert(0, '../src/')
from mod_geo import *

input_file = argv[1]

nc = Dataset(input_file)
wavenumber = nc.variables['frequency'][:, :, :].filled(0)
psd_ref = nc.variables['psd_ref'][:, :, :]
psd_map = nc.variables['psd_study'][:, :, :]
coherence = nc.variables['coherence'][:, :, :]
lat = nc.variables['lat2D'][:, :]
lon = nc.variables['lon2D'][:, :]
nb_segment = nc.variables['nb_segment'][:, :]
nc.close()

lon = np.where(lon >= 360, lon-360, lon)
lon = np.where(lon < 0, lon+360, lon)


def updateThetaAxis(ax):
    """

    :param ax:
    :return:
    """
    labels = []
    for f in ax.get_xticks():
        labels.append(np.round(np.cos(f), 2))
    ax.set_xticklabels(labels[::2], fontsize=10)

    return labels


def plt_spectral_taylor_diagram(cix, ciy):
    """
    Plot spectral taylor diagram
    :param cix:
    :param ciy:
    :return:
    """

    ilon, ilat = find_nearest_index_lonlat(lon, lat, cix, ciy)

    normalized_std = psd_map[:, ilat, ilon]/psd_ref[:, ilat, ilon]
    angle_coherence = np.arccos(coherence[:, ilat, ilon])

    plt.ion()
    plt.show()

    r_theta05 = np.arange(0, 1.2, 0.01)
    
    # r_theta_area = np.linspace(0.5, 1, 120)
    
    theta05 = np.arccos(0.5)*np.ones(np.shape(r_theta05))
    # theta0 = 0*np.ones(np.shape(r_theta05))
    theta_r05 = np.arccos(np.arange(0, 1.5, 0.01))
    r05 = 0.5*np.ones(np.shape(theta_r05))
    r1 = np.ones(np.shape(theta_r05))
    tt = np.arange(0, np.arccos(0.49), 0.01)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, polar=True)
    ax1.fill_between(tt, 0.5, 1, color='0.3', alpha=0.5)
    # ax1.fill_betweenx(r_theta_area, theta0, theta05, color='0.3', alpha=0.5)
    ax1.plot(angle_coherence, normalized_std, color='0.5', alpha=0.5)
    c = ax1.scatter(angle_coherence, normalized_std, c=1/wavenumber[:, ilat, ilon],
                    s=50, cmap='Spectral_r', lw=0, vmin=100, vmax=800)
    ax1.scatter(0, 1, c='white', s=150, marker=(5, 1), zorder=10, facecolors='none', edgecolors='r')
    ax1.plot(theta05, r_theta05, color='k', lw=2)
    ax1.plot(theta_r05, r05, color='k', lw=2)
    ax1.plot(theta_r05, r1, color='k', lw=2)
    ax1.set_thetamin(0)
    ax1.set_thetamax(90)
    ax1.set_rmax(1.2)
    ax1.grid(True)
    # updateThetaAxis(ax1)
    ax1.set_ylabel("Normalized PSD", fontweight='bold', fontsize=12)
    plt.annotate("Spectral Taylor Diagram at lon = %s and lat = %s"
                 % (str(np.round(lon[ilat, ilon], 2)), str(np.round(lat[ilat, ilon], 2))),
                 xy=(0.3, 1.07), xycoords='axes fraction', color='black',
                 bbox=dict(facecolor='none', edgecolor='black', pad=10.0))
    cbar = plt.colorbar(c)
    cbar.set_label('Wavelength (km)', fontweight='bold')
    plt.annotate('Coherence', xy=(0.75, 0.85), xycoords='axes fraction', rotation=-50, fontweight='bold', fontsize=12)
    plt.annotate('REF', xy=(0.85, 0.04), xycoords='axes fraction', fontweight='bold', color='red', fontsize=12)
    plt.annotate('ratio = 0.5', xy=(pi/3.5, 0.45), xycoords='data', rotation=-50, fontsize=8)
    plt.annotate('ratio = 1.0', xy=(pi/4, 1.01),  xycoords='data', rotation=-50, fontsize=8)
    plt.annotate('Coherence = 0.5', xy=(pi/2.1, 0.3), xycoords='data', rotation=60, fontsize=8)

    # labels = updateThetaAxis(ax1)
    # ax1.set_xticks(labels)
    plt.show()


def onclick(event):
    """
    Function onclick
    :param event:
    :return:
    """
    toolbar = plt.get_current_fig_manager().toolbar
    if event.button == 1 and event.inaxes and toolbar.mode == '':
        # global indexx, indexy
        indexx, indexy = event.xdata, event.ydata
        print(' ')
        print('lon = ', indexx, ';', 'lat = ', indexy)
        plt_spectral_taylor_diagram(indexx, indexy)
    else:
        print('Clicked ouside axes bounds but inside plot window \n')


fig, ax = plt.subplots()
bmap = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-10, urcrnrlon=365, resolution='i')
bmap.drawcoastlines(zorder=5, linewidth=0.25)
bmap.fillcontinents(color='grey', lake_color='white')
bmap.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0], fontsize=20, dashes=[4, 4], linewidth=0.25)
bmap.drawmeridians(np.arange(-180, 360, 45), labels=[0, 0, 0, 1], fontsize=20, dashes=[4, 4], linewidth=0.25)
xx, yy = bmap(lon, lat)
scatter = bmap.scatter(xx, yy, c=nb_segment, s=2, lw=0)
cbar_scatter = bmap.colorbar(scatter, location='bottom', pad='15%')
cbar_scatter.set_label('Nb segment in computation')
ax.set_title("-- CLICK ON THE MAP TO DISPLAY SPECTRAL TAYLOR DIAGRAM --", fontweight='bold')
fig.canvas.callbacks.connect('button_press_event', onclick)
plt.show()
