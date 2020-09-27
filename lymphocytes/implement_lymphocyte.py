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



mat_filename_405s2 = '../batch1/405s2/mat/405s2.mat'
coeffPathStart_405s2 = '../batch1/zoom0.2_coeffs/405s2/405s2_'
niigzDir_405s2 = '../batch1/niigz_zoom0.2/405s2/'
exit_idxs_405s2 = []

mat_filename_405s3 = '../batch1/405s3/mat/405s3.mat'
coeffPathStart_405s3 = '../batch1/zoom0.2_coeffs/405s3/405s3_'
niigzDir_405s3 = '../batch1/niigz_zoom0.2/405s3/'
exit_idxs_405s3 = []

mat_filename_406s2 = '../batch1/406s2/mat/406s2.mat'
coeffPathStart_406s2 = '../batch1/zoom0.2_coeffs/406s2/406s2_'
niigzDir_406s2 = '../batch1/niigz_zoom0.2/406s2/'
exit_idxs_406s2 = []

mat_filename_406s2_SMALL = '../zoom0.08_406s2/406s2.mat'
coeffPathStart_406s2_SMALL = '../zoom0.08_406s2/coeffs/406s2_'
niigzDir_406s2_SMALL = '../zoom0.08_406s2/niigz/'



# -------------------

mat_filename_stack2 = '../batch2/Stack2.mat'
coeffPathStart_stack2 = '../batch2/zoom0.2_coeffs/stack2/stack2_'
niigzDir_stack2 = '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack2/'
exit_idxs_stack2 = list(range(285, 288)) + list(range(294, 300))

mat_filename_stack3 = '../batch2/Stack3.mat'
coeffPathStart_stack3 = '../batch2/zoom0.2_coeffs/stack3/stack3_'
niigzDir_stack3 = '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack3/'
exit_idxs_stack3 = list(range(90, 100)) + list(range(117, 131)) + list(range(139, 300))

mat_filename_stack4 = '../batch2/Stack4.mat'
coeffPathStart_stack4 = '../batch2/zoom0.2_coeffs/stack4/stack4_'
niigzDir_stack4 = '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack4/'
exit_idxs_stack4 = list(range(2, 11)) + list(range(20, 34)) + list(range(37, 175))

mat_filename_stack5 = '../batch2/Stack5.mat'
coeffPathStart_stack5 = '../batch2/zoom0.2_coeffs/stack5/stack5_'
niigzDir_stack5 = '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack5/'
exit_idxs_stack5  = list(range(50, 115))

mat_filename_stack7 = '../batch2/Stack7.mat'
coeffPathStart_stack7 = '../batch2/zoom0.2_coeffs/stack7/stack7_'
niigzDir_stack7 = '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack7/'
exit_idxs_stack7 = list(range(9, 71))

mat_filename_stack9 = '../batch2/Stack9.mat'
coeffPathStart_stack9 = '../batch2/zoom0.2_coeffs/stack9/stack9_'
niigzDir_stack9 = '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack9/'
exit_idxs_stack9 = list(range(39, 45)) + list(range(59, 132))

mat_filename_stack10 = '../batch2/Stack10.mat'
coeffPathStart_stack10 = '../batch2/zoom0.2_coeffs/stack10/stack10_'
niigzDir_stack10 = '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack10/'
exit_idxs_stack10 = list(range(23, 57)) + list(range(60, 63)) + list(range(68, 202))


lymph_series_405s2 = [mat_filename_405s2, coeffPathStart_405s2, niigzDir_405s2, exit_idxs_405s2]
lymph_series_405s3 = [mat_filename_405s3, coeffPathStart_405s3, niigzDir_405s3, exit_idxs_405s3]
lymph_series_406s2 = [mat_filename_406s2, coeffPathStart_406s2, niigzDir_406s2, exit_idxs_406s2]

lymph_series_406s2_SMALL = [mat_filename_406s2_SMALL, coeffPathStart_406s2_SMALL, niigzDir_406s2_SMALL]

lymph_series_stack2 = [mat_filename_stack2, coeffPathStart_stack2, niigzDir_stack2, exit_idxs_stack2]
lymph_series_stack3 = [mat_filename_stack3, coeffPathStart_stack3, niigzDir_stack3, exit_idxs_stack3]
lymph_series_stack4 = [mat_filename_stack4, coeffPathStart_stack4, niigzDir_stack4, exit_idxs_stack4]
lymph_series_stack5 = [mat_filename_stack5, coeffPathStart_stack5, niigzDir_stack5, exit_idxs_stack5]
lymph_series_stack7 = [mat_filename_stack7, coeffPathStart_stack7, niigzDir_stack7, exit_idxs_stack7]
lymph_series_stack9 = [mat_filename_stack9, coeffPathStart_stack9, niigzDir_stack9, exit_idxs_stack9]
lymph_series_stack10 = [mat_filename_stack10, coeffPathStart_stack10, niigzDir_stack10, exit_idxs_stack10]






if __name__ == '__main__':

    lymph_serieses = LymphocyteSerieses([lymph_series_stack2, lymph_series_stack3, lymph_series_stack4,
                                            lymph_series_stack5, lymph_series_stack7,lymph_series_stack9,
                                             lymph_series_stack10, lymph_series_405s2, lymph_series_405s3, lymph_series_406s2])


    #lymph_serieses = LymphocyteSerieses([lymph_series_stack4])



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

            voxels = keep_only_largest_object(voxels)
            voxels = binary_fill_holes(voxels).astype(int)


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
