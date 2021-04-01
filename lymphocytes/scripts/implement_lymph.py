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
import pyvista as pv
from mkalgo.mk import mk_eab
import pickle

from lymphocytes.data.dataloader_good_segs_1 import stack_triplets_1
from lymphocytes.data.dataloader_good_segs_2 import stack_triplets_2
from lymphocytes.lymph_serieses.lymph_serieses_class import Lymph_Serieses
import lymphocytes.utils.general as utils_general



if __name__ == '__main__':

    # IMPLEMENT TURNING BY ANGLE BETWEEN CENTROID AND ORIGIN
    stack_triplets = stack_triplets_1 + stack_triplets_2

    lymph_serieses = Lymph_Serieses(stack_triplets, cells_model = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], max_l = 15)



    #lymph_serieses._set_speeds()
    #lymphs = utils_general.list_all_lymphs(lymph_serieses)
    #plt.scatter([0 for i in lymphs], [lymph.speed for lymph in lymphs])



    """
    for idx_cell, lymph_series in lymph_serieses.lymph_serieses.items():
        if idx_cell in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]:
            file = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/uropods/cell_{}.pickle'.format(idx_cell),"rb")
            uropods = pickle.load(file)
            frames = []
            dists = []
            for lymph in lymph_series:
                frames.append(lymph.frame)
                dist = uropods[lymph.frame] - lymph.orig_centroid
                dists.append(np.linalg.norm(dist))
            plt.plot(frames, dists)
            plt.show()
            plt.close()
    """

    #lymph_serieses.lymph_serieses[0][0].show_voxels()
    #lymph_serieses.lymph_serieses[0][0].show_voxels()
    #lymph_serieses.lymph_serieses[0][0].surface_plot()
    #lymph_serieses.lymph_serieses[0][0].plotRecon_singleDeg(plotter=plotter, max_l = 3, uropod_align = True)


    ### single cell methods ###
    #lymph_serieses.plot_orig_series(idx_cell=idx_cell, uropod_align = False, plot_every=3)
    #lymph_serieses.select_uropods(idx_cell=idx_cell, plot_every=1)
    #lymph_serieses.plot_uropod_trajectory(idx_cell = 9)
    #lymph_serieses.plot_series_PCs(idx_cell=6, plot_every=1)
    #lymph_serieses.plot_migratingCell(idx_cell=1, plot_every = 10)
    #lymph_serieses.plot_series_voxels(plot_every)
    #lymph_serieses.plot_recon_series(plot_every=5)


    ### many cells methods ###
    #lymph_serieses.plot_volumes()
    #lymph_serieses.plot_RIvector_mean_var()
    #lymph_serieses.plot_mean_lymph()
    #lymph_serieses.scatter_first2_descriptors(pca=True)
    #lymph_serieses.line_plot_3D(centroid_uropod_pca = 'uropod', color_by = None)
    #lymph_serieses.plot_recons_increasing_l(self, maxl, l)
    lymph_serieses.PC_sampling(n_components = 3)
    lymph_serieses.plot_component_lymphs(grid_size=8, pca=True)
    #lymph_serieses.plot_2D_embeddings(pca = True, components = (0, 2))



    ### centroid variable methods ###
    #lymph_serieses.plot_centroids(color_by = 'speed')
    #lymph_serieses.set_angles()
    #lymph_serieses.correlate_shape_with_speedAngle(max_l, n_components, pca = False)

    plt.show()
