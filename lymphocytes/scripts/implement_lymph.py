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


from lymphocytes.data.dataloader_good_segs_2 import stack_attributes_2
from lymphocytes.data.dataloader_good_segs_3 import stack_attributes_3
from lymphocytes.data.stereotypical import stack_attributes_stereotypical

from lymphocytes.cells.cells_class import Cells
import lymphocytes.utils.general as utils_general
from lymphocytes.cells.uncertainties import save_PC_uncertainties, save_curvatures



idx_cells_orig =  ['2_{}'.format(i) for i in range(10)] + ['3_1_{}'.format(i) for i in range(6) if i != 0] + ['zm_3_3_{}'.format(i) for i in range(8)] + \
                     ['zm_3_4_{}'.format(i) for i in range(4)] + ['zm_3_5_2', 'zm_3_6_0'] # zm_3_6_0 is a stop cell
idx_cells_stop = ['2_1', 'zm_3_4_0', 'zm_3_3_3', 'zm_3_6_0']
idx_cells_run = ['zm_3_3_5', 'zm_3_3_2', 'zm_3_3_4', 'zm_3_4_1']

pca_obj_cells_all = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/pca_obj.pickle', "rb"))

if __name__ == '__main__':

    # ALL
    cells = Cells(stack_attributes_2 + stack_attributes_3, cells_model = idx_cells_orig, max_l = 15, uropods_bool = True)

    # STOP
    #cells = Cells(stack_attributes_stereotypical, cells_model = idx_cells_stop, max_l = 15, uropods_bool = True)

    # RUN
    #cells = Cells(stack_attributes_stereotypical, cells_model = idx_cells_run, max_l = 15, uropods_bool = True)

    cells.pca_obj = pca_obj_cells_all

    #cells_stop.pca_obj = pca_obj

    #save_PC_uncertainties(cells_all)





    """
    cells_run = Cells(stack_attributes_stereotypical, cells_model = idx_cells_run, max_l = 15, uropods_bool = True)
    cells_run.pca_obj = cells_all_pca_obj
    cells_run._set_pca(n_components = 3)
    cells_run.gather_time_series(save_name = 'shape_series_run')

    cells_stop = Cells(stack_attributes_stereotypical, cells_model = idx_cells_stop, max_l = 15, uropods_bool = True)
    cells_stop.pca_obj = cells_all_pca_obj
    cells_stop._set_pca(n_components = 3)
    cells_stop.gather_time_series(save_name = 'shape_series_stop')
    """





    #save_curvatures(cells, idx_cells_orig)

    #cells_stop.plot_orig_series(idx_cell = idx_cell, uropod_align = False, color_by = None, plot_every = 5)



    #cells_all.plot_uropod_centroid_line(idx_cell = idx_cell, plot_every = 1)

    #save_curvatures(cells, idx_cells_orig)


    #cells_all.plot_cumulatives()
    #cells_all.histogram()



    #cells_all.alignments(min_length = 1/300, max_diff = 0.002)

    #cells.plot_l_truncations(idx_cell=idx_cell)


    ### single cell methods ###

    #cells.plot_migratingCell(idx_cell=idx_cell, color_by = 'time', plot_every = 1)

    #cells.plot_uropod_centroid_line(idx_cell=idx_cell, plot_every=1)
    #cells_all.plot_orig_series(idx_cell = '2_1', uropod_align = False, color_by = None, plot_every = 6)
    #cells_run.plot_rotations( time_either_side = time_either_side)
    #cells.plot_voxels_series(idx_cell=idx_cell, plot_every = 10)
    #cells.select_uropods(idx_cell=idx_cell)
    #cells.select_uropods_add_frames(idx_cell = idx_cell)
    #cells.plot_attribute(idx_cell, attribute = 'RI_vector0')
    #cells.plot_series_PCs(idx_cell=idx_cell, plot_every=5)
    #cells.plot_series_voxels(plot_every)
    #cells.plot_recon_series(idx_cell = idx_cell, max_l = 1, color_by = None, plot_every=1)


    ### many cells methods ###
    #cells.add_cells_set_PCs(stack_attributes_stereotypical, idx_cells_run)
    #cells.plot_attributes(attributes = ['volume', 'pca0', 'pca1', 'pca2', 'morph_deriv', 'delta_centroid'])
    #cells.plot_attributes(attributes = [ 'volume'])
    #cells.plot_RIvector_mean_std()
    #cells.plot_mean_lymph()
    #cells_all.PC_sampling()
    #cells.PC_arrows()
    #cells_all.plot_component_lymphs(grid_size=7, pca=True, plot_original = False, max_l = 2)
    #cells.line_plot_3D(centroid_uropod_pca = 'pca', color_by = None)
    #cells_all.correlation(attributes = ['pca0', 'pca1', 'pca2',  'run_uropod', 'run_uropod_running_mean', 'morph_deriv'])
    #cells_all.correlation(attributes = ['pca0',  'pca1', 'run_uropod', 'run_uropod_running_mean'])
    #cells_all.scatter_run_running_means()
    #cells.correlation(attributes = ['run', 'run_centroid'], widths = [7])
    #cells.correlation_annotate( 'run', 'run_centroid')
    #cells_all.plot_PC_space(plot_original = False, max_l = 2)
    #cells.plot_PC_space(plot_original = True)

    ### centroid variable methods ###
    #cells.gather_time_series(save_name = 'shape_series_run')
