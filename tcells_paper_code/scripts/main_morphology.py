import sys
import pyvista as pv
pv.set_plot_theme("document")
import pickle

from tcells_paper_code.dataloader.all import stack_attributes_all
from tcells_paper_code.dataloader.stereotypical import stack_attributes_stereotypical
from tcells_paper_code.videos.videos_class import Videos
from tcells_paper_code.videos.uncertainties import save_PC_uncertainties, save_curvatures


"""
TO LOOK INTO:
& the functions that change files, e.g.show_video
"""

all_run_stop = sys.argv[1]
idx_cell = None



if all_run_stop == 'all':
    idx_cells = ['CELL1', 'CELL2', 'CELL3', 'CELL4', 'CELL5', 'CELL6', 'CELL7', 'CELL8', 'CELL9', 'CELL11', 'CELL12', 'CELL13', 'CELL14', 'CELL15', 'CELL16', 'CELL17', 'CELL18', 'CELL19', 'CELL20', 'CELL21', 'CELL22', 'CELL23', 'CELL24', 'CELL25', 'CELL26', 'CELL27',  'CELL29', 'CELL30', 'CELL31']
    stack_attributes = stack_attributes_all
elif all_run_stop == 'run':
    idx_cells = ['CELL21', 'CELL29', 'CELL20', 'CELL25']
    stack_attributes = stack_attributes_stereotypical
elif all_run_stop == 'stop':
    idx_cells = ['CELL1', 'CELL24', 'CELL19', 'CELL31']
    stack_attributes = stack_attributes_stereotypical


pca_obj_cells_all = pickle.load(open('../data/pca_obj.pickle', "rb"))
cells = Videos(stack_attributes, cells_model = idx_cells, uropods_bool = True)


cells.pca_obj = pca_obj_cells_all # load the PCA object (so can use PCs computed across all cells even if loading only 1 cell)
cells._set_pca(n_components=3)


cells.gather_time_series('shape_series_stop')
sys.exit()

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
#cells.plot_attributes(attributes = ['volume', 'pca0', 'pca1', 'pca2'])
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
