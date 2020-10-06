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

from lymphocytes.lymph_serieses.lymph_serieses_class import Lymph_Serieses

from lymphocytes.data.dataloader_good_segs import stack_triplets



if __name__ == '__main__':

    lymph_serieses = Lymph_Serieses(stack_triplets)
    #lymph_serieses.plot_zoomedVoxels_volumes(zoom_factor = 1)
    #lymph_serieses.plot_recon_series(max_l = 5, plot_every = 20, color_param = 'phis')
    lymph_serieses.plot_rotInvRep_2Dmanifold(grid_size = 5, max_l = 5, pca = True, just_x = False, just_y = False)


    """
    lymph_serieses.plot_migratingCell(idx_cell = 0, max_l = 5, plot_every = 15)

    lymph_serieses.plot_cofms(colorBy = 'speed')

    lymph_serieses.plot_individAndHists(variable = string)

    lymph_serieses.C_plot_raw_volumes_series(zoom_factor = 0.2)
    lymph_serieses.C_plot_series_niigz(plot_every = 20)
    lymph_serieses.C_plot_recon_series(max_l = 5, plot_every = 25, color_param = None)

    lymph_serieses.plot_rotInv_mean_std(maxl = 5)
    lymph_serieses.C_plot_rotInv_series_bars(maxl = 5, plot_every = 1, means_adjusted = False, colorBy = 'pc_vals')
    lymph_serieses.plot_recons_increasing_l(lmax = 1, l = 1)
    lymph_serieses.plot_rotInv_2Dmanifold(grid_size = 5, max_l = 5, pca = True, just_x = False, just_y = False)

    lymph_serieses.pca_plot_sampling(max_l = 10, num_samples = 5, color_param = None, rotInv = True)
    lymph_serieses.plot_speeds_angles()
    lymph_serieses.pca_plot_shape_trajectories(max_l = 3, rotInv = True, colorBy = 'time')
    lymph_serieses.correlate_with_speedAngle(max_l = 5, rotInv = True, n_components = 6, pca = False)
    lymph_serieses.plot_pca_recons(n_pca_components, max_l, plot_every)
    """


    plt.show()
