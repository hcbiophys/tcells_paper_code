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
                     ['zm_3_4_{}'.format(i) for i in range(4)] + ['zm_3_5_2', 'zm_3_6_0'] + ['zm_3_5_1']
idx_cells_stop = ['2_1', 'zm_3_4_0', 'zm_3_3_3', 'zm_3_6_0']
idx_cells_run = ['zm_3_3_5', 'zm_3_5_1', 'zm_3_3_4', 'zm_3_4_1']

pca_obj_cells_all = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/pca_obj.pickle', "rb"))

if __name__ == '__main__':


    idx_cell = 'zm_3_4_0'
    # ALL
    #cells = Cells(stack_attributes_2 + stack_attributes_3, cells_model = idx_cells_orig, max_l = 15, uropods_bool = True)
    # STOP
    #cells = Cells(stack_attributes_stereotypical, cells_model = idx_cells_stop, max_l = 15, uropods_bool = True)
    # RUN
    cells = Cells(stack_attributes_stereotypical, cells_model = [idx_cell], max_l = 15, uropods_bool = True)

    cells.pca_obj = pca_obj_cells_all
    cells._set_pca(n_components=3)

    lymph_series = cells.cells[idx_cell]

    """
    #plt.plot([i.pca0 for i in cells.cells[idx_cell]], c = 'red')
    times = [i.frame*lymph_series[0].t_res for i in lymph_series]
    plt.plot(times, [i.pca1 for i in lymph_series], c = 'blue')
    #plt.plot([i.pca2 for i in cells.cells[idx_cell]], c = 'green')
    plt.show()
    """


    #save_PC_uncertainties(cells, idx_cells_orig)

    cells.show_video(idx_cell=idx_cell, color_by = None, save = True)




    #save_curvatures(cells, [idx_cell])

    #cells.plot_orig_series(idx_cell = idx_cell, uropod_align = False, color_by = None, plot_every = 6, plot_flat = False)




    #cells.plot_uropod_centroid_line(idx_cell = idx_cell, plot_every = 1)

    #cells.plot_cumulatives()
    #cells.histogram()



    #cells.alignments(min_length = 0.0025)

    #cells.plot_l_truncations(idx_cell=idx_cell)


    ### single cell methods ###

    #cells.plot_migratingCell(idx_cell=idx_cell, color_by = 'time', plot_every = 50)

    #cells.plot_uropod_centroid_line(idx_cell=idx_cell, plot_every=1)
    #cells.plot_orig_series(idx_cell = 'zm_3_4_1', uropod_align = True, color_by = 'pca1', plot_every = 6, plot_flat = False)
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
    #cells.PC_sampling()
    #cells.PC_arrows()
    #cells.plot_component_lymphs(grid_size=7, pca=True, plot_original = False, max_l = 3)
    #cells.plot_component_lymphs(grid_size=7, pca=True, plot_original = True, max_l = 3)
    #cells.line_plot_3D(centroid_uropod_pca = 'pca', color_by = None)
    #cells.correlation(attributes = ['pca0', 'pca1', 'pca2',  'run_uropod', 'run_uropod_running_mean'])
    #cells_all.correlation(attributes = ['pca0',  'pca1', 'run_uropod', 'run_uropod_running_mean'])
    #cells.scatter_run_running_means()
    #cells.correlation(attributes = ['run', 'run_centroid'], widths = [7])
    #cells.correlation_annotate( 'run', 'run_centroid')
    #cells.plot_PC_space(plot_original = False, max_l = 3)
    #cells.plot_PC_space(plot_original = True)

    ### centroid variable methods ###
    #cells.gather_time_series(save_name = 'shape_series_go')
