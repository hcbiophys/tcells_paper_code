import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import h5py # Hierarchical Data Format 5
import nibabel as nib
from scipy.ndimage import zoom
from scipy.ndimage import measurements
from scipy.special import sph_harm
from matplotlib import cm, colors
from mayavi import mlab
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
pv.set_plot_theme("document")
from mkalgo.mk import mk_eab
import pickle
import random


from lymphocytes.dataloader.all import stack_attributes_all
from lymphocytes.dataloader.stereotypical import stack_attributes_stereotypical



from lymphocytes.videos.videos_class import Videos
import lymphocytes.utils.general as utils_general
from lymphocytes.videos.uncertainties import save_PC_uncertainties, save_curvatures


"""
TO LOOK INTO:
& the functions that change files, e.g.show_video
"""

all_run_stop = sys.argv[1]
idx_cell = None


print(utils_general.cell_idxs_conversion)
sys.exit()



if all_run_stop == 'all':
    idx_cells =  ['2_{}'.format(i) for i in range(1, 10)] + ['3_1_{}'.format(i) for i in range(6) if i != 0] + ['zm_3_3_{}'.format(i) for i in range(8)] + ['zm_3_4_{}'.format(i) for i in range(4)] + ['zm_3_5_2', 'zm_3_6_0'] + ['zm_3_5_1']
    stack_attributes = stack_attributes_2 + stack_attributes_3
elif all_run_stop == 'run':
    idx_cells = ['zm_3_3_5', 'zm_3_5_1', 'zm_3_3_4', 'zm_3_4_1']
    stack_attributes = stack_attributes_stereotypical
elif all_run_stop == 'stop':
    idx_cells = ['2_1', 'zm_3_4_0', 'zm_3_3_3', 'zm_3_6_0']
    stack_attributes = stack_attributes_stereotypical


pca_obj_cells_all = pickle.load(open('../data/pca_obj.pickle', "rb"))
cells = Videos(stack_attributes, cells_model = idx_cells, uropods_bool = True)


cells.pca_obj = pca_obj_cells_all # load the PCA object (so can use PCs computed across all cells even if loading only 1 cell)
cells._set_pca(n_components=3)

"""
SINGLE CELL METHODS
"""

#cells.plot_l_truncations(idx_cell=idx_cell)
#cells.plot_orig_series(idx_cell = idx_cell, uropod_align = False, color_by = 'pca1', plot_every = 6, plot_flat = False)
#cells.plot_recon_series(idx_cell = idx_cell, max_l = 1, color_by = None, plot_every=1)
#cells.plot_migratingCell(idx_cell=idx_cell, opacity = 0.2, plot_every = 41)
#cells.plot_uropod_centroid_line(idx_cell = idx_cell, plot_every = 1)
#cells.plot_series_PCs(idx_cell=idx_cell, plot_every=5)

"""
SINGLE CELL METHODS that EDIT FILES
"""
#cells.select_uropods(idx_cell=idx_cell)
#cells.select_uropods_add_frames(idx_cell = idx_cell)
#cells.show_video(idx_cell=idx_cell, color_by = None, save = True)
#cells.add_colorbar_pic(idx_cell = idx_cell, old_frame_dir = '/Users/harry/Desktop/lymph_vids/stop/no_colorbar/{}/'.format(idx_cell), new_frame_dir = '/Users/harry/Desktop/lymph_vids/stop/colorbar/{}/'.format(idx_cell), pc012 = 2)


"""
MANY CELL METHODS
"""

#cells.alignments(min_length = 0.0025)
#cells.speeds_histogram()
#cells.plot_cumulatives()
#cells.bimodality_emergence()
#cells.plot_component_frames(bin_size=7, pca=True, plot_original = False, max_l = 3)
#cells.PC_sampling()
#cells.plot_PC_space(plot_original = False, max_l = 3)
#cells.plot_attributes(attributes = ['volume', 'pca0', 'pca1', 'pca2', 'morph_deriv'])
#cells.correlation(attributes = ['pca0', 'pca1', 'pca2', 'speed_uropod'])
#cells.scatter_annotate('speed_uropod', 'speed_centroid')
#cells.expl_var_bar_plot()
#cells.low_high_PC1_vecs()

"""
MANY CELL METHODS that EDIT FILES
"""
#save_curvatures(cells, [idx_cell])
#save_PC_uncertainties(cells, idx_cells_orig)
#cells.gather_time_series(save_name = 'None')
