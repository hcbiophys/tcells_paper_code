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

from lymphocytes.data.dataloader_good_segs_1 import stack_triplets_1
from lymphocytes.data.dataloader_good_segs_2 import stack_triplets_2


if __name__ == '__main__':

    stack_triplets = stack_triplets_1 + stack_triplets_2
    lymph_serieses = Lymph_Serieses([stack_triplets[0]], max_l = 3)

    #lymph_serieses.lymph_serieses[0][0].show_voxels(origOrZoomed= 'zoomed')
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #lymph_serieses.lymph_serieses[0][0].plotRecon_singleDeg(ax, max_l = 6, color_param = 'thetas', elev = None, azim = None)
    #lymph_serieses.lymph_serieses[0][0].surface_plot(subsample_rate=50)

    #lymph_serieses.plot_volumes()
    #lymph_serieses.plot_rotInvRep_mean_std()
    #lymph_serieses.scatter_first2_descriptors(pca=True)
    #lymph_serieses.plot_recons_increasing_l(self, maxl, l)
    #lymph_serieses.plot_2Dmanifold(grid_size = 8, pca = False, just_x = False, just_y = False)

    ### single cell methods ###
    #lymph_serieses.plot_migratingCell(plot_every = 10)
    #lymph_serieses.plot_series_voxels(plot_every)
    lymph_serieses.plot_recon_series(plot_every=10)
    #lymph_serieses.plot_rotInvRep_series_bars(maxl = 5, plot_every = 1, means_adjusted = False)

    ### centroid variable methods ###
    #lymph_serieses.plot_cofms(colorBy = 'speed')
    #lymph_serieses.set_angles()
    #lymph_serieses.correlate_shape_with_speedAngle(max_l, n_components, pca = False)

    plt.show()
