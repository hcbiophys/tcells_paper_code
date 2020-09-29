import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import h5py # Hierarchical Data Format 5
import nibabel as nib
from scipy.ndimage import zoom
from scipy.special import sph_harm
from matplotlib import cm, colors
from mayavi import mlab
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from lymphocyte_snap_class import *
from lymphocyte_series_class import *
from utils import *

from lymphocytes.data.dataloader import stack_quads_list
from lymphocytes.lymph_series.lymph_series_class import Lymphocyte_Serieses

from niigz_tools import Niigz_Tools


if __name__ == '__main__':


    lymph_serieses = Lymphocyte_Serieses(stack_quads_list)

    filenames = ['2', '3', '4', '5', '7', '9', '10', '405s2', '405s3', '406s2']
    for idx_series in range(len(lymph_serieses.lymph_serieses)):
        fig = plt.figure()

        initials = []
        perc_increases = []
        colors = []
        for idx_lymph in range(len(lymph_serieses.lymph_serieses[idx_series])):

            lymph = lymph_serieses.lymph_serieses[idx_series][idx_lymph]

            if lymph.exited:
                colors.append('red')
            else:
                colors.append('blue')

            voxels = lymph.voxels

            a = np.sum(voxels)
            initials.append(a)

            #lw, num = measurements.label(voxels_inv)
            #area = measurements.sum(voxels_inv, lw, index=np.arange(lw.max() + 1))

            voxels = Niigz_Tools.keep_only_largest_object(voxels)
            voxels = Niigz_Tools.binary_fill_holes(voxels).astype(int)


            b = np.sum(voxels)

            perc_increase = np.round(100*(b-a)/a, 3)
            perc_increases.append(perc_increase)

            print(idx_lymph, a, perc_increase)

        ax = fig.add_subplot(2, 1, 1)
        ax.plot(initials)

        ax.scatter([i for i in range(len(initials))], initials, c = colors)
        ax.set_ylim([0, 800000])
        ax.set_title('Initial cell volume')
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(perc_increases)
        ax.scatter([i for i in range(len(perc_increases))], perc_increases, c = colors)
        ax.set_ylim([0, 35])
        ax.set_title('Percentage increase')
        plt.savefig('/Users/harry/Desktop/temporary/{}.png'.format(filenames[idx_series]))



    """
    idx = 0
    lymph = lymph_serieses.lymph_serieses[0][idx]
    print('idx', lymph.idx)

    voxels = 1-np.asarray(lymph.voxels)
    lw, num = measurements.label(voxels)
    area = measurements.sum(voxels, lw, index=np.arange(lw.max() + 1))
    print('num', num, 'area', area)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection = '3d')
    lymph.SH_plotRecon_singleDeg(ax, max_l=5, color_param = 'thetas', elev = None, azim = None, normaliseScale = False)
    #lymph.ORIG_scatter_vertices(1)
    """


    """
    print(lymph_serieses.lymph_serieses[0][65].idx)
    lymph = lymph_serieses.lymph_serieses[0][65]

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection = '3d')
    lymph.SH_plotRecon_singleDeg(ax, max_l=2, color_param = 'xs', elev = None, azim = None, normaliseScale = False)

    ax = fig.add_subplot(1, 3, 2, projection = '3d')
    lymph.SH_plotRecon_singleDeg(ax, max_l=2, color_param = 'ys', elev = None, azim = None, normaliseScale = False)

    ax = fig.add_subplot(1, 3, 3, projection = '3d')
    lymph.SH_plotRecon_singleDeg(ax, max_l=2, color_param = 'zs', elev = None, azim = None, normaliseScale = False)
    """
    #lymph_serieses.plot_migratingCell(idx_cell = 0, max_l = 5, plot_every = 15)


    #lymph_serieses.plot_cofms(colorBy = 'speed')


    #lymph_serieses.plot_individAndHists(variable = string)



    #lymph_serieses.C_plot_raw_volumes_series(zoom_factor = 0.2)
    #lymph_serieses.C_plot_series_niigz(plot_every = 20)
    #lymph_serieses.C_plot_recon_series(max_l = 5, plot_every = 25, color_param = None)

    #lymph_serieses.plot_rotInv_mean_std(maxl = 5)
    #lymph_serieses.C_plot_rotInv_series_bars(maxl = 5, plot_every = 1, means_adjusted = False, colorBy = 'pc_vals')
    #lymph_serieses.plot_recons_increasing_l(lmax = 1, l = 1)
    #lymph_serieses.plot_rotInv_2Dmanifold(grid_size = 5, max_l = 5, pca = True, just_x = False, just_y = False)


    #lymph_serieses.pca_plot_sampling(max_l = 10, num_samples = 5, color_param = None, rotInv = True)
    #lymph_serieses.plot_speeds_angles()
    #lymph_serieses.pca_plot_shape_trajectories(max_l = 3, rotInv = True, colorBy = 'time')
    #lymph_serieses.correlate_with_speedAngle(max_l = 5, rotInv = True, n_components = 6, pca = False)
    #lymph_serieses.plot_pca_recons(n_pca_components, max_l, plot_every)

    ###  DO MANIFOLD WITH SH 1 AND 2
    ###  DO MANIFOLD WITH STANDARDIZED ROT INV FOR PCA, AND LOOK AT SAMPLING ETC
    ###  DECIDE WHETHER RAW ROTINV SH, PCA OR STANDARDIZED PCA BEST FOR LOW DIM lowDimRep

    plt.show()
