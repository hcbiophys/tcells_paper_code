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


#from lymphocytes.data.dataloader_good_segs_1 import stack_attributes_1
from lymphocytes.data.dataloader_good_segs_2 import stack_attributes_2
from lymphocytes.data.dataloader_good_segs_3 import stack_attributes_3
from lymphocytes.cells.cells_class import Cells
import lymphocytes.utils.general as utils_general


if __name__ == '__main__':
    idx_cells =  ['2_{}'.format(i) for i in range(10)] + ['3_1_{}'.format(i) for i in range(6)] + ['zm_3_3_{}'.format(i) for i in range(8)] + ['zm_3_4_{}'.format(i) for i in range(4)]
    random.shuffle(idx_cells)

    idx_cell = 'zm_3_4_1'
    idx_cells = [idx_cell]


    cells = Cells(stack_attributes_2 + stack_attributes_3, cells_model = idx_cells, max_l = 15)

    #cells.scatter_run_running_means()
    #cells.run_power_spectrum(attribute = 'run', time_either_side = 7, high_or_low_run = 'low')
    #cells.run_power_spectrum(attribute = 'run', time_either_side = 7, high_or_low_run = 'high')




    #cells._set_searching(time_either_side=None, time_either_side_mean=None)

    #for attribute in ['pca0', 'pca1', 'pca2', 'morph_deriv', 'delta_centroid', 'delta_sensing_direction']:
    ### single frame methods ###
    #cells._set_delta_centroids()
    #lymphs = utils_general.list_all_lymphs(cells)

    """
    idx_frame = 6
    single_frame = [i for i in cells.cells[idx_cell] if i.frame == idx_frame][0]
    plotter = pv.Plotter()
    cells.cells[idx_cell][idx_frame].surface_plot(plotter, color = (0.5, 1, 1), opacity = 0.5)
    #single_frame.voxel_point_cloud(plotter)
    #cells.cells[idx_cell][idx_frame].plotRecon_singleDeg(plotter, max_l = 15, uropod_align = False, color = (1, 1, 0.5), opacity = 0.5)
    plotter.show()
    sys.exit()
    """


    #cells.cells[idx_cell][0].show_voxels()
    #cells.cells[idx_cell][50].surface_plot(plotter = plotter, uropod_align = True)
    #cells.plot_l_truncations(idx_cell=idx_cell)


    ### single cell methods ###

    #cells.plot_migratingCell(idx_cell=idx_cell, color_by = 'time', plot_every = 15)
    cells.plot_orig_series(idx_cell=idx_cell, uropod_align = False, color_by = 'run', plot_every = 5)
    #cells.plot_voxels_series(idx_cell=idx_cell, plot_every = 10)
    #cells.select_uropods(idx_cell=idx_cell)
    #cells.select_uropods_add_frames(idx_cell = idx_cell)
    #cells.save_calibrations(idx_cell=idx_cell)
    #cells.plot_uropod_centroid_line(idx_cell = idx_cell, plot_every = 2)
    #cells.plot_uropod_trajectory(idx_cell = 4)
    #cells.plot_attribute(idx_cell, attribute = 'RI_vector0')
    #cells.plot_series_PCs(idx_cell=idx_cell, plot_every=15)
    #cells.plot_series_voxels(plot_every)
    #cells.plot_recon_series(idx_cell = idx_cell, max_l = 1, color_by = None, plot_every=1)
    #cells.gather_time_series()




    ### many cells methods ###
    #cells.plot_attributes(attributes = ['volume', 'pca0', 'pca1', 'pca2', 'morph_deriv', 'delta_centroid', 'delta_sensing_direction'])
    #cells.plot_attributes(attributes = [ 'volume'])
    #cells.plot_RIvector_mean_std()
    #cells.plot_mean_lymph()
    #cells.PC_sampling(n_components = 3)
    #cells.plot_component_lymphs(grid_size=7, pca=True, plot_original = False)
    #cells.plot_component_lymphs(grid_size=7, pca=True, plot_original = True)
    #cells.line_plot_3D(centroid_uropod_pca = 'pca', color_by = None)
    #cells.plot_2D_embeddings(pca = True, components = (0, 1))
    #cells.correlation(attributes = ['pca0', 'pca1', 'pca2', 'morph_deriv',  'run', 'searching', 'searching_mean'], searching_widths = [7, 50, 100])
    #cells.correlation(attributes = ['pca0', 'pca1', 'pca2', 'run'], run_widths = [ 150, 200])
    #cells.correlation_annotate( 'pca0', 'run')
    #cells.rigid_motions()
    #cells.plot_PC_space(plot_original = False)
    #cells.plot_PC_space(plot_original = True)

    ### centroid variable methods ###
    #cells.plot_centroids(color_by = 'delta_centroid')
    #cells.set_angles()
    #cells.correlate_shape_with_delta_centroidAngle(max_l, n_components, pca = False)
    #cells.gather_time_series()
    plt.show()
