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
from lymphocytes.data.stereotypical import stack_attributes_stereotypical

from lymphocytes.cells.cells_class import Cells
import lymphocytes.utils.general as utils_general
from lymphocytes.cells.uncertainties import uncertainties, save_curvatures


# UROPODS TO CHECK



# UROPODS TO EDIT
#  'zm_3_3_4' approx 600s
# 'zm_3_3_6' approx 230s
# 'zm_3_4_0' (check what uncertainties look like)
# 'zm_3_4_2' look into PCs & redo uropods


if __name__ == '__main__':
    idx_cells_run = ['zm_3_3_5', 'zm_3_3_2', 'zm_3_3_4', 'zm_3_4_1']
    idx_cells_stop = ['2_1', 'zm_3_4_0', 'zm_3_3_3']
    idx_cells_orig =  ['2_{}'.format(i) for i in range(10)] + ['3_1_{}'.format(i) for i in range(6) if i != 0] + ['zm_3_3_{}'.format(i) for i in range(8)] + \
                         ['zm_3_4_{}'.format(i) for i in range(4)] + ['zm_3_5_2', 'zm_3_6_0'] # zm_3_6_0 is a stop cell


    random.shuffle(idx_cells_orig)

    for idx_cell in idx_cells_orig:




        cells = Cells(stack_attributes_2 + stack_attributes_3, cells_model = [idx_cell], max_l = 15, keep_every_random = 1)
        cells.plot_uropod_centroid_line(idx_cell=idx_cell, plot_every=1, time_either_side=None)


        #cells.plot_uropod_centroid_line(idx_cell = idx_cell, plot_every = 1, time_either_side = None)

        #save_curvatures(cells, idx_cells_orig)



        #uncertainties(cells)


        #cells.plot_cumulatives()




        #cells.alignments()



        #cells.histogram()



        #cells.plot_l_truncations(idx_cell=idx_cell)


        ### single cell methods ###

        #cells.plot_migratingCell(idx_cell=idx_cell, color_by = 'time', plot_every = 1)
        #cells.scatter_run_running_means()
        #cells.plot_orig_series(idx_cell = idx_cell, uropod_align = False, color_by = 'angle', plot_every = 5)
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
        #cells.PC_sampling(n_components = 3)
        #cells.PC_arrows()
        #cells.plot_component_lymphs(grid_size=7, pca=True, plot_original = False)
        #cells.plot_component_lymphs(grid_size=7, pca=True, plot_original = True)
        #cells.line_plot_3D(centroid_uropod_pca = 'pca', color_by = None)
        #cells.correlation(attributes = ['pca0', 'pca1', 'pca2', 'morph_deriv',  'run', 'searching', 'searching_mean'], searching_widths = [7, 50, 100])
        #cells.correlation(attributes = ['run', 'run_centroid'], widths = [7])
        #cells.correlation_annotate( 'run', 'run_centroid')
        #cells.plot_PC_space(plot_original = False)
        #cells.plot_PC_space(plot_original = True)

        ### centroid variable methods ###
        #cells_stereotypical.gather_time_series(save_name = 'shape_series_run')
        #plt.show()
