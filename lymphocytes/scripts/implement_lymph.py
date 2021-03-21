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


from lymphocytes.lymph_serieses.lymph_serieses_class import Lymph_Serieses

from lymphocytes.data.dataloader_good_segs_1 import stack_triplets_1
from lymphocytes.data.dataloader_good_segs_2 import stack_triplets_2




if __name__ == '__main__':

    stack_triplets = stack_triplets_1 + stack_triplets_2
    lymph_serieses = Lymph_Serieses(stack_triplets, cells_model = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11], max_l = 15)

    #lymph_serieses.lymph_serieses[0][0].show_voxels()
    #lymph_serieses.lymph_serieses[0][0].show_voxels()
    #lymph_serieses.lymph_serieses[0][0].surface_plot()

    ### single cell methods ###
    #lymph_serieses.plot_orig_series(idx_cell=idx_cell, plot_every=3)
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
    #lymph_serieses.plot_first3_descriptors(pca = True, color_by = None)
    #lymph_serieses.plot_recons_increasing_l(self, maxl, l)
    lymph_serieses.PC_sampling(n_components = 3)
    lymph_serieses.plot_component_lymphs(grid_size=8, pca=True)
    lymph_serieses.plot_2D_embeddings(pca = True, components = (0, 1))


    ### centroid variable methods ###
    #lymph_serieses.plot_centroids(color_by = 'speed')
    #lymph_serieses.set_angles()
    #lymph_serieses.correlate_shape_with_speedAngle(max_l, n_components, pca = False)

    plt.show()
