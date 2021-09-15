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
pv.set_plot_theme("document")
from mkalgo.mk import mk_eab
import pickle

#from lymphocytes.data.dataloader_good_segs_1 import stack_quads_1
from lymphocytes.data.dataloader_good_segs_2 import stack_quads_2
from lymphocytes.data.dataloader_good_segs_3 import stack_quads_3
from lymphocytes.cells.cells_class import Cells
import lymphocytes.utils.general as utils_general


if __name__ == '__main__':
    #idx_cells = ['2_{}'.format(i) for i in range(10)] + ['3_0_0' + '3_0_1'] + ['3_1_{}'.format(i) for i in range(6)]

    idx_cell = '3_0_2'
    cells = Cells(stack_quads_2 + stack_quads_3, cells_model = [idx_cell], max_l = 15) # I think I have to ignore stack_quads_2 as these are duplicates?
    #for attribute in ['pca0', 'pca1', 'pca2', 'morph_deriv', 'delta_centroid', 'delta_sensing_direction']:
    ### single frame methods ###
    #cells._set_delta_centroids()
    #lymphs = utils_general.list_all_lymphs(cells)
    #cells.cells[idx_cell][0].surface_plot(plotter=plotter)
    #cells.cells[0][0].show_voxels()
    #cells.cells[idx_cell][50].surface_plot(plotter = plotter, uropod_align = True)
    #cells.plot_l_truncations(idx_cell=idx_cell)

    ### single cell methods ###
    #cells.plot_migratingCell(idx_cell=idx_cell, plot_every = 15)
    #cells.plot_orig_series(idx_cell=idx_cell, uropod_align = False, color_by = None, plot_every = 2)
    cells.select_uropods(idx_cell=idx_cell)
    #cells.plot_uropod_centroid_line(idx_cell = idx_cell, plot_every = 1)
    #cells.plot_uropod_trajectory(idx_cell = 4)
    #cells.plot_attribute(idx_cell, attribute = 'RI_vector0')
    #cells.plot_series_PCs(idx_cell=idx_cell, plot_every=15)
    #cells.plot_series_voxels(plot_every)
    #cells.plot_recon_series(idx_cell = idx_cell, max_l = 2, color_by = 'pca2', plot_every=5)
    #cells.gather_time_series()




    ### many cells methods ###
    #cells.plot_attributes(attributes = ['volume', 'pca0', 'pca1', 'pca2', 'morph_deriv', 'delta_centroid', 'delta_sensing_direction'])
    #cells.plot_attributes(attributes = [ 'morph_deriv'])
    #cells.plot_RIvector_mean_std()
    #cells.plot_mean_lymph()
    #cells.PC_sampling(n_components = 3)
    #cells.plot_component_lymphs(grid_size=8, pca=True, plot_original = False)
    #cells.plot_component_lymphs(grid_size=8, pca=True, plot_original = True)
    #cells.line_plot_3D(centroid_uropod_pca = 'pca', color_by = None)
    #cells.plot_2D_embeddings(pca = True, components = (0, 1))
    #cells.correlation(independents = ['pca0', 'pca1', 'pca2', 'morph_deriv', 'delta_centroid', 'delta_sensing_direction'], dependents = ['morph_deriv', 'delta_centroid', 'delta_sensing_direction'])
    #cells.rigid_motions()
    #cells.plot_PC_space(plot_original = False)
    #cells.plot_PC_space(plot_original = True)
    #cells.plot_PC_space()

    ### centroid variable methods ###
    #cells.plot_centroids(color_by = 'delta_centroid')
    #cells.set_angles()
    #cells.correlate_shape_with_delta_centroidAngle(max_l, n_components, pca = False)
    #cells.gather_time_series()
    #plt.show()
