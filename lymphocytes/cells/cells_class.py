import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import h5py # Hierarchical Data Format 5
import nibabel as nib
from scipy.ndimage import zoom
from scipy.special import sph_harm
from matplotlib import cm, colors
import matplotlib.tri as mtri
from mayavi import mlab
import pyvista as pv
import os
from sklearn.decomposition import PCA
import scipy.stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import pickle
import random
from pykdtree.kdtree import KDTree
import pandas as pd

from lymphocytes.cells.pca_methods import PCA_Methods
from lymphocytes.cells.single_cell_methods import Single_Cell_Methods
from lymphocytes.cells.centroid_variable_methods import Centroid_Variable_Methods
from lymphocytes.cell_frame.cell_frame_class import Cell_Frame
from lymphocytes.behavior_analysis.consecutive_frames_class import Consecutive_Frames

import lymphocytes.utils.disk as utils_disk
import lymphocytes.utils.plotting as utils_plotting
import lymphocytes.utils.general as utils_general



class Cells(Single_Cell_Methods, PCA_Methods, Centroid_Variable_Methods):
    """
    Class for all lymphocyte serieses
    Mixins are:
    - Single_Cell_Methods: methods suitable for a single cell series
    - PCA_Methods: methods without involving reduced-dimensionality representation (via PCA)
    - Centroid_Variable_Methods: methods to set attributes based on centroid and uropod, e.g. delta_centroid and delta_sensing_direction
    """


    def __init__(self, stack_attributes, cells_model, max_l):
        """
        - stack_attributes: (idx_cell, mat_filename, coeffPathFormat, zoomedVoxelsPathFormat, xyz_res)
        - cells_model: indexes of the cells to model, e.g. ['3_1_0', '3_1_2']
        """


        self.cells = {}
        self.cell_colors = {}

        for (idx_cell, mat_filename, coeffPathFormat, xyz_res, color, t_res, zoom_factor) in stack_attributes:

            if cells_model == 'all' or idx_cell in cells_model:
                print('idx_cell: {}'.format(idx_cell))
                lymph_series = []

                uropods = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/uropods/cell_{}.pickle'.format(idx_cell), "rb"))


                if idx_cell[:2] == 'zs':
                    frames_all, voxels_all, vertices_all, faces_all = utils_disk.get_attribute_from_mat(mat_filename=mat_filename, zeiss_type='zeiss_single')
                elif idx_cell[:2] == 'zm':
                    frames_all, voxels_all, vertices_all, faces_all = utils_disk.get_attribute_from_mat(mat_filename=mat_filename, zeiss_type='zeiss_many', idx_cell = int(idx_cell[-1]), include_voxels = True)
                    #calibrations = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/calibrations/cell_{}.pickle'.format(idx_cell), "rb"))
                else:
                    frames_all, voxels_all, vertices_all, faces_all = utils_disk.get_attribute_from_mat(mat_filename=mat_filename, zeiss_type='not_zeiss', include_voxels = True)

                for idx in range(len(frames_all)):

                    if os.path.isfile(coeffPathFormat.format(int(frames_all[idx]))): # if it's within arena and SPHARM-PDM worked
                        snap = Cell_Frame(mat_filename = mat_filename, frame = frames_all[idx], coeffPathFormat = coeffPathFormat, voxels = voxels_all[idx], xyz_res = xyz_res, zoom_factor = zoom_factor, idx_cell = idx_cell, max_l = max_l, uropod = np.array(uropods[frames_all[idx]]), vertices = vertices_all[idx], faces = faces_all[idx])

                        #snap = Cell_Frame(mat_filename = mat_filename, frame = frames_all[idx], coeffPathFormat = coeffPathFormat, voxels = voxels_all[idx], xyz_res = xyz_res, zoom_factor = zoom_factor, idx_cell = idx_cell, max_l = max_l, uropod  = None, calibration = None, vertices = vertices_all[idx], faces = faces_all[idx])

                        snap.color = color
                        snap.t_res = t_res
                        lymph_series.append(snap)


                self.cells[idx_cell] = lymph_series
                print('max_frame: {}'.format(max(frames_all)))

        self.pca_set = False


    """
    def set_curvatures(self):

        for lymph in utils_general.list_all_lymphs(self):

            surf = pv.PolyData(lymph.vertices, lymph.faces)

            surf = surf.smooth(n_iter=5000)
            surf = surf.decimate(0.98)
            curvature = surf.curvature()
            surf_tree = KDTree(surf.points.astype(np.double))
            dist, idx = surf_tree.query(np.array([[lymph.uropod[0], lymph.uropod[1], lymph.uropod[2]]]))
            print(dist, idx)


            surf = pv.PolyData(lymph.vertices, lymph.faces)
            lymph.curvature = surf.curvature()
    """


    def plot_attributes(self, attributes):
        """
        Plot time series and histograms of cell attributes (e.g. volume, principal components etc. )
        """

        fig_lines, fig_hists = plt.figure(figsize = (2, 7)), plt.figure(figsize = (2, 7))

        axes_line = [fig_lines.add_subplot(len(attributes), 1, i+1) for i in range(len(attributes))]
        axes_hist = [fig_hists.add_subplot(len(attributes), 1, i+1) for i in range(len(attributes))]
        all_attributes = [[] for i in range(len(attributes))]

        for idx_attribute, attribute in enumerate(attributes):
            if attribute[:3] == 'pca':
                self._set_pca(n_components=3)
            if attribute == 'delta_centroid' or attribute == 'delta_sensing_direction':
                self._set_centroid_attributes(attribute)
            if attribute == 'morph_deriv':
                self._set_morph_derivs()

        for lymph_series in self.cells.values():
            color = np.random.rand(3,)
            for idx_attribute, attribute in enumerate(attributes):

                lymphsNested = utils_general.get_nestedList_connectedLymphs(lymph_series)
                for lymphs in lymphsNested:
                    frame_list = [lymph.frame for lymph in lymphs if getattr(lymph, attribute) is not None]
                    attribute_list = [getattr(lymph, attribute) for lymph in lymphs if getattr(lymph, attribute)  is not None]
                    axes_line[idx_attribute].plot([lymphs[0].t_res*i for i in frame_list], attribute_list, color = lymphs[0].color, label = lymphs[0].idx_cell)
                    all_attributes[idx_attribute] += attribute_list
                    if idx_attribute != len(attributes)-1:
                        axes_line[idx_attribute].set_xticks([])
                    axes_line[idx_attribute].set_yticks([])

        for idx_attribute in range(len(attributes)):
            axes_hist[idx_attribute].hist(all_attributes[idx_attribute], bins = 7, orientation = 'horizontal', color = 'darkblue')
            axes_hist[idx_attribute].invert_xaxis()
            if idx_attribute != len(attributes)-1:
                axes_hist[idx_attribute].set_xticks([])

        #fig_lines.legend()
        for fig in [fig_lines, fig_hists]:
            fig.tight_layout()
            fig.subplots_adjust(hspace = 0)
            for ax in fig.axes:
                ax.set_ylim(bottom=0)
        plt.show()



    def plot_RIvector_mean_std(self):
        """
        Plot the mean and standard deviation of rotationally-invariant shape descriptor
        """
        lymphs = utils_general.list_all_lymphs(self)
        vectors = np.array([lymph.RI_vector for lymph in lymphs])

        means = np.mean(vectors, axis = 0)
        vars = np.std(vectors, axis = 0)

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.bar(range(len(means)), means, color = 'red')
        ax = fig.add_subplot(1, 2, 2)
        ax.bar(range(len(vars)), vars, color = 'blue')

        plt.show()



    def line_plot_3D(self, centroid_uropod_pca, color_by):
        """
        3D plot the first 3 descriptors moving in time
        """
        if centroid_uropod_pca == 'pca':
            self._set_pca(n_components=3)
        all_vectors = [getattr(lymph, centroid_uropod_pca) for lymph in utils_general.list_all_lymphs(self)]
        [min0, min1, min2] = [min([i[j] for i in all_vectors]) for j in [0, 1, 2]]
        [max0, max1, max2] = [max([i[j] for i in all_vectors]) for j in [0, 1, 2]]
        [min0, min1, min2] = [i-0.1 for i in [min0, min1, min2]]
        [max0, max1, max2] = [i+0.1 for i in [max0, max1, max2]]

        if color_by == 'delta_centroid' or color_by == 'delta_sensing_direction':
            self._set_centroid_attributes(color_by)
            vmin, vmax = utils_general.get_color_lims(self, color_by = color_by)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        for idx_cell, lymph_series in self.cells.items():
            color = np.random.rand(3,)
            lymphsNested = utils_general.get_nestedList_connectedLymphs(lymph_series)

            for lymphs in lymphsNested:
                for idx in range(len(lymphs)-1):
                    if color_by == 'delta_centroid' or color_by == 'delta_sensing_direction':
                        color = utils_general.get_color(lymphs[idx], color_by = color_by, vmin = vmin, vmax = vmax)
                        print('h', getattr(lymphs[idx], centroid_uropod_pca)[0], getattr(lymphs[idx+1], centroid_uropod_pca)[0])
                    ax.plot((getattr(lymphs[idx], centroid_uropod_pca)[0], getattr(lymphs[idx+1], centroid_uropod_pca)[0]), (getattr(lymphs[idx], centroid_uropod_pca)[1],  getattr(lymphs[idx+1], centroid_uropod_pca)[1]), (getattr(lymphs[idx], centroid_uropod_pca)[2],  getattr(lymphs[idx+1], centroid_uropod_pca)[2]), c = color)
                    #ax.scatter(getattr(lymphs[idx], centroid_uropod_pca)[0], getattr(lymphs[idx], centroid_uropod_pca)[1], getattr(lymphs[idx], centroid_uropod_pca)[2], c = color)
                    ax.plot([min0, min0], (getattr(lymphs[idx], centroid_uropod_pca)[1],  getattr(lymphs[idx+1], centroid_uropod_pca)[1]), (getattr(lymphs[idx], centroid_uropod_pca)[2],  getattr(lymphs[idx+1], centroid_uropod_pca)[2]), c = color, alpha = 0.1)
                    ax.plot((getattr(lymphs[idx], centroid_uropod_pca)[0], getattr(lymphs[idx+1], centroid_uropod_pca)[0]), [max1, max1], (getattr(lymphs[idx], centroid_uropod_pca)[2],  getattr(lymphs[idx+1], centroid_uropod_pca)[2]), c = color, alpha = 0.1)
                    ax.plot((getattr(lymphs[idx], centroid_uropod_pca)[0], getattr(lymphs[idx+1], centroid_uropod_pca)[0]), (getattr(lymphs[idx], centroid_uropod_pca)[1],  getattr(lymphs[idx+1], centroid_uropod_pca)[1]), [min2, min2] , c = color, alpha = 0.1)


            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_zlabel('PC 3')
            ax.grid(False)
            ax.set_xlim([min0, max0])
            ax.set_ylim([min1, max1])
            ax.set_zlim([min2, max2])

        if centroid_uropod_pca == 'centroid' or centroid_uropod_pca == 'uropod':
            utils_plotting.set_limits_3D(*fig.axes)
        utils_plotting.equal_axes_notSquare_3D(*fig.axes)


    def plot_mean_lymph(self):
        """
        Plot the mesh closest to the mean
        """
        lymphs = utils_general.list_all_lymphs(self)
        vectors = np.array([lymph.RI_vector for lymph in lymphs])
        mean_vec = np.mean(vectors, axis = 0)
        dists = [np.sqrt(np.sum(np.square(vec-mean_vec))) for vec in vectors]
        lymph = lymphs[dists.index(min(dists))]
        plotter = pv.Plotter()
        lymph.surface_plot(plotter = plotter)
        plotter.show(cpos=[0, 1, 0])





    def _scatter_plotted_components(self, vectors, plotted_points_all):
        """
        Scatter points of the plotted meshes
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        for i in vectors:
            ax1.scatter(i[0], i[1], i[2], s = 1, c = 'lightskyblue')
        for i, color in zip(plotted_points_all, ['red', 'green', 'black']):
            for j in i:
                ax1.scatter(j[0], j[1], j[2], s = 6, c = color)
                ax2.scatter(j[0], j[1], j[2], s = 6, c = color)
        plt.show()





    def plot_2D_embeddings(self, pca, components):
        """
        Plot the meshes at their embeddings for 2 components
        """
        lymphs = utils_general.list_all_lymphs(self)
        if pca:
            self._set_pca(n_components = 3)


        random.shuffle(lymphs)

        plotter = pv.Plotter()
        for lymph in lymphs:
            if random.randint(1, 5) == 3:
                lymph._uropod_align(axis = np.array([1, 0, 0]))

                lymph.vertices *= 0.002

                if pca:
                    vector = np.take(lymph.pca, components)
                else:
                    vector = np.take(lymph.RI_vector, components)
                vector = np.append(vector, 0)
                lymph.vertices += vector

                lymph.surface_plot(plotter=plotter, uropod_align=False)


        plotter.show()




    def correlation(self,  attributes):
        """
        Get pearson correlation coefficient between independent and dependent variable
        """
        fig_scatt, fig_r, fig_p = plt.figure(figsize = (5, 6)), plt.figure(), plt.figure()
        r_values = np.empty((len(attributes), len(attributes)))
        p_values = np.empty((len(attributes), len(attributes)))
        r_values[:], p_values[:] = np.nan, np.nan
        for idx_row, dependent in enumerate(attributes):
            for idx_col, independent in enumerate(attributes):
                if dependent != independent and dependent[:3] != 'pca':
                    ax = fig_scatt.add_subplot(len(attributes), len(attributes), idx_row*len(attributes)+idx_col+1)
                    ax.set_xlabel(independent)
                    ax.set_ylabel(dependent)
                    if independent[:3] == 'pca' or dependent[:3] == 'pca' :
                        self._set_pca(n_components=3)
                    if independent == 'delta_centroid'or dependent == 'delta_centroid':
                        self._set_centroid_attributes('delta_centroid')
                    if independent == 'delta_sensing_direction' or dependent == 'delta_sensing_direction':
                        self._set_centroid_attributes('delta_sensing_direction')
                    if independent == 'run' or dependent == 'run':
                        self._set_centroid_attributes('run')
                    if independent == 'morph_deriv' or dependent == 'morph_deriv':
                        self._set_morph_derivs()

                    plot_lymphs = [lymph for lymph_series in self.cells.values() for lymph in lymph_series if getattr(lymph, independent) is not None and  getattr(lymph, dependent) is not None]
                    xs = [getattr(lymph, independent) for lymph in plot_lymphs]
                    ys = [getattr(lymph, dependent)  for lymph in plot_lymphs]
                    colors = [lymph.color  for lymph in plot_lymphs]
                    result = scipy.stats.linregress(np.array(xs), np.array(ys))
                    ax.scatter(xs, ys, s=0.1, c = colors)

                    model_xs = np.linspace(min(list(xs)), max(list(xs)), 50)
                    ax.plot(model_xs, [result.slope*i+result.intercept for i in model_xs], c = 'red')
                    ax.tick_params(axis="both",direction="in")
                    if idx_row != len(attributes)-1:
                        ax.set_xticks([])
                    if idx_col != 0:
                        ax.set_yticks([])

                    r_values[idx_row, idx_col] = result.rvalue
                    p_values[idx_row, idx_col] = result.pvalue
        fig_scatt.subplots_adjust(hspace=0, wspace=0)
        ax = fig_r.add_subplot(111)
        r = ax.imshow(r_values, cmap = 'Blues')
        matplotlib.cm.Blues.set_bad(color='white')
        ax.axis('off')
        ax.set_title('r_values')
        fig_r.colorbar(r, ax=ax, orientation='horizontal')
        ax = fig_p.add_subplot(111)
        p = ax.imshow(p_values, cmap = 'Reds')
        matplotlib.cm.Reds.set_bad(color='white')
        ax.set_title('p_values')
        fig_p.colorbar(p, ax=ax, orientation='horizontal')
        ax.axis('off')

        plt.show()


    def rigid_motions(self):
        """
        See where in PCA space are a) rigid rotations b) rigid translations with no rotation
        """

        """
        # temp, visualise distributions for setting thresholds
        self._set_morph_derivs()
        lymphs = [lymph for lymph_series in self.cells.values() for lymph in lymph_series if lymph.morph_deriv is not None]
        plt.scatter([random.randint(0, 5) for lymph in lymphs], [lymph.morph_deriv for lymph in lymphs])
        plt.show()

        self._set_centroid_attributes('delta_sensing_direction')
        lymphs = [lymph for lymph_series in self.cells.values() for lymph in lymph_series if lymph.delta_sensing_direction is not None]
        plt.scatter([random.randint(0, 5) for lymph in lymphs], [lymph.delta_sensing_direction for lymph in lymphs])
        plt.show()
        """

        morph_deriv_thresh_low = 0.025
        delta_sensing_direction_thresh_low = 0.01
        delta_sensing_direction_thresh_high = 0.03
        self._set_pca(n_components=3)

        self._set_morph_derivs()
        self._set_centroid_attributes('delta_sensing_direction')
        self._set_centroid_attributes('delta_centroid')

        fig = plt.figure()
        lymphs = [lymph for lymph_series in self.cells.values() for lymph in lymph_series if lymph.morph_deriv is not None and lymph.delta_sensing_direction is not None and lymph.delta_centroid is not None]


        for idx_pc, pc in enumerate(['pca0', 'pca1', 'pca2']):

            ax = fig.add_subplot(1, 3, idx_pc+1)
            for lymph in lymphs:
                if lymph.morph_deriv < morph_deriv_thresh_low:
                    if lymph.delta_sensing_direction < delta_sensing_direction_thresh_low:
                        ax.scatter(getattr(lymph, pc), lymph.morph_deriv,  color = lymph.color, marker = 'o', s = 5, alpha = 0.5, linewidths = 0)
            for lymph in lymphs:
                if lymph.morph_deriv < morph_deriv_thresh_low and lymph.delta_sensing_direction > delta_sensing_direction_thresh_low:
                    ax.scatter(getattr(lymph, pc), lymph.morph_deriv,  color = lymph.color, marker = 'x', s = 15)
            ax.tick_params(axis="both",direction="in")
            if idx_pc != 0:
                ax.set_yticks([])
            #ax.set_xticks([])

            """
            ax = fig.add_subplot(2, 3, 3+idx_pc+1)
            for lymph in lymphs:
                if lymph.morph_deriv > morph_deriv_thresh_low or lymph.delta_sensing_direction > delta_sensing_direction_thresh_low and lymph.delta_centroid < 0.15:
                    ax.scatter(getattr(lymph, pc), lymph.morph_deriv,  color = lymph.color, marker = 'o', s = 5, alpha = 0.5, linewidths = 0)
            for lymph in lymphs:
                if lymph.morph_deriv < morph_deriv_thresh_low and lymph.delta_sensing_direction < delta_sensing_direction_thresh_low and lymph.delta_centroid > 0.15:
                    ax.scatter(getattr(lymph, pc), lymph.morph_deriv,  color = lymph.color, marker = 'x', s = 15)
            ax.tick_params(axis="both",direction="in")
            if idx_pc != 0:
                ax.set_yticks([])
            """

        plt.subplots_adjust(hspace = 0, wspace = 0)
        plt.show()





    def gather_time_series(self):
        """
        Gather shape time series into dictionary with sub dictionaries containing joint (or gaps of size 1) frame series
        """


        self._set_pca(n_components=3)
        self._set_centroid_attributes('delta_centroid', num_either_side = 2)
        self._set_centroid_attributes('delta_sensing_direction', num_either_side = 2)
        self._set_centroid_attributes('run', num_either_side = 2, run_running_mean = True)

        """
        for lymph in utils_general.list_all_lymphs(self):
            lymph.pca = lymph.RI_vector
        """


        all_consecutive_frames = []

        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        for idx_cell, lymph_series in self.cells.items():

            count = 0
            consecutive_frames = Consecutive_Frames(name = str(idx_cell)+alphabet[count], t_res_initial = lymph_series[0].t_res)
            prev_values = [None, None, None, None, None, None]
            for idx_lymph, lymph in enumerate(lymph_series):
                if idx_lymph == 0 or lymph.frame-prev_values[0] == 1:
                    consecutive_frames.add(lymph.frame, lymph.pca[0], lymph.pca[1], lymph.pca[2], lymph.delta_centroid, lymph.delta_sensing_direction, lymph.run)
                elif lymph.frame-prev_values[0] == 2: #linear interpolation if only 1 frame missing
                    if consecutive_frames.delta_centroid_list[-1] is None or lymph.delta_centroid is None:
                        delta_centroid_staged = None
                    else:
                        delta_centroid_staged = (consecutive_frames.delta_centroid_list[-1]+lymph.delta_centroid)/2
                    if consecutive_frames.delta_sensing_direction_list[-1] is None or lymph.delta_sensing_direction is None:
                        delta_sensing_direction_staged = None
                    else:
                        delta_sensing_direction_staged = (consecutive_frames.delta_sensing_direction_list[-1]+lymph.delta_sensing_direction)/2
                    if consecutive_frames.run_list[-1] is None or lymph.run is None:
                        run_staged = None
                    else:
                        run_staged = (consecutive_frames.run_list[-1]+lymph.run)/2

                    consecutive_frames.add(lymph.frame, (consecutive_frames.pca0_list[-1]+lymph.pca[0])/2, (consecutive_frames.pca1_list[-1]+lymph.pca[1])/2, (consecutive_frames.pca2_list[-1]+lymph.pca[2])/2, delta_centroid_staged, delta_sensing_direction_staged, run_staged)
                else:
                    consecutive_frames.interpolate()
                    all_consecutive_frames.append(consecutive_frames)
                    count += 1
                    consecutive_frames = Consecutive_Frames(name = str(idx_cell)+alphabet[count], t_res_initial = lymph_series[0].t_res)
                    consecutive_frames.add(lymph.frame, lymph.pca[0], lymph.pca[1], lymph.pca[2], lymph.delta_centroid, lymph.delta_sensing_direction, lymph.run)
                prev_values = [lymph.frame, lymph.pca[0], lymph.pca[1], lymph.pca[2], lymph.delta_centroid, lymph.delta_sensing_direction, lymph.run]

            consecutive_frames.interpolate()
            all_consecutive_frames.append(consecutive_frames)

        pickle_out = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series.pickle','wb')
        pickle.dump(all_consecutive_frames, pickle_out)
