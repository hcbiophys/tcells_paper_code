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
from scipy import signal
from sklearn.decomposition import PCA
from pyvista import examples
import time

from lymphocytes.cells.pca_methods import PCA_Methods
from lymphocytes.cells.single_cell_methods import Single_Cell_Methods
from lymphocytes.cells.centroid_variable_methods import Centroid_Variable_Methods
from lymphocytes.cell_frame.cell_frame_class import Cell_Frame
from lymphocytes.behavior_analysis.consecutive_frames_class import Consecutive_Frames
from lymphocytes.cells.curvature_lists import all_lists
from lymphocytes.cells.uncertainties import save_PC_uncertainties, get_mean_time_diff, save_curvatures

import lymphocytes.utils.disk as utils_disk
import lymphocytes.utils.plotting as utils_plotting
import lymphocytes.utils.general as utils_general


class Cells(Single_Cell_Methods, PCA_Methods, Centroid_Variable_Methods):
    """
    Class for all lymphocyte serieses
    Mixins are:
    - Single_Cell_Methods: methods suitable for a single cell series
    - PCA_Methods: methods without involving reduced-dimensionality representation (via PCA)
    - Centroid_Variable_Methods: methods to set attributes based on centroid and uropod, e.g. delta_centroid
    """


    def __init__(self, stack_attributes, cells_model, max_l, uropods_bool, keep_every_random = 1):
        """
        - stack_attributes: (idx_cell, mat_filename, coeffPathFormat, zoomedVoxelsPathFormat, xyz_res)
        - cells_model: indexes of the cells to model, e.g. ['3_1_0', '3_1_2']
        """

        stack_attributes_dict = {i[0]:i for i in stack_attributes}

        self.cells = {}

        for idx_cell in cells_model:


            (idx_cell, mat_filename, coeffPathFormat, xyz_res, color, t_res) = stack_attributes_dict[idx_cell]

            print('idx_cell: {}'.format(idx_cell))
            lymph_series = []


            self.uropods = uropods_bool
            if self.uropods:
                uropods = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/uropods/cell_{}.pickle'.format(idx_cell), "rb"))



            if idx_cell[:2] == 'zs':
                frames_all, voxels_all, vertices_all, faces_all = utils_disk.get_attribute_from_mat(mat_filename=mat_filename, zeiss_type='zeiss_single', include_voxels = False)
            elif idx_cell[:2] == 'zm':
                frames_all, voxels_all, vertices_all, faces_all = utils_disk.get_attribute_from_mat(mat_filename=mat_filename, zeiss_type='zeiss_many', idx_cell = int(idx_cell[-1]), include_voxels = False)
            else:
                frames_all, voxels_all, vertices_all, faces_all = utils_disk.get_attribute_from_mat(mat_filename=mat_filename, zeiss_type='not_zeiss', include_voxels = False)

            for frame in range(int(max(frames_all)+1)):
                if os.path.isfile(coeffPathFormat.format(frame)): # if it's within arena and SPHARM-PDM worked
                    if np.random.randint(0, keep_every_random) == 0:
                        idx = frames_all.index(frame)


                        if self.uropods:
                            snap = Cell_Frame(mat_filename = mat_filename, frame = frames_all[idx], coeffPathFormat = coeffPathFormat, voxels = voxels_all[idx], xyz_res = xyz_res,  idx_cell = idx_cell, max_l = max_l, uropod = np.array(uropods[frames_all[idx]]), vertices = vertices_all[idx], faces = faces_all[idx])
                        else:
                            snap = Cell_Frame(mat_filename = mat_filename, frame = frames_all[idx], coeffPathFormat = coeffPathFormat, voxels = voxels_all[idx], xyz_res = xyz_res,  idx_cell = idx_cell, max_l = max_l, uropod = None, vertices = vertices_all[idx], faces = faces_all[idx])

                        snap.color = np.array(color)
                        snap.t_res = t_res

                        lymph_series.append(snap)


            self.cells[idx_cell] = lymph_series
            print('max_frame: {}'.format(max(frames_all)))


        if self.uropods and keep_every_random == 1:
            self.interoplate_SPHARM()

            for idx_cell, lymph_series in self.cells.items():
                mean_time_diff = get_mean_time_diff(idx_cell, lymph_series)
                for snap in lymph_series:
                    snap.mean_time_diff = mean_time_diff

            for idx_cell in self.cells.keys():
                print(idx_cell, ' mean_time_diff: {}'.format(self.cells[idx_cell][0].mean_time_diff))
                self._set_mean_uropod_and_centroid(idx_cell = idx_cell, time_either_side = self.cells[idx_cell][0].mean_time_diff/2)


        self.pca_obj = None





    def interoplate_SPHARM(self):

        for idx_cell, lymph_series in self.cells.items():
            lymph_series_new = []
            dict = utils_general.get_frame_dict(lymph_series)

            frames = list(dict.keys())
            for i in range(int(frames[0]), int(max(frames))+1):
                if i in frames:
                    lymph_series_new.append(dict[i])

                elif i not in frames and i-1 in frames and i+1 in frames:
                    uropod_interpolated = (dict[i-1].uropod + dict[i+1].uropod)/2
                    snap = Cell_Frame(mat_filename = None, frame = i, coeffPathFormat = None, voxels = None, xyz_res = None,  idx_cell = lymph_series[0].idx_cell, max_l = None, uropod = uropod_interpolated, vertices = None, faces = None)
                    snap.color = dict[i-1].color
                    snap.t_res = dict[i-1].t_res
                    snap.centroid = (dict[i-1].centroid + dict[i+1].centroid)/2
                    snap.volume = (dict[i-1].volume + dict[i+1].volume)/2
                    snap.RI_vector = (dict[i-1].RI_vector + dict[i+1].RI_vector)/2
                    snap.is_interpolation = True
                    lymph_series_new.append(snap)
            self.cells[idx_cell] = lymph_series_new



    def blebs(self):
        fig = plt.figure()
        for idx_vec in range(15):
            ax = fig.add_subplot(3, 5, idx_vec + 1)
            for i, lymph_series in enumerate(self.cells.values()):
                ys = [lymph.RI_vector[idx_vec] for lymph in lymph_series]
                xs = [i for _ in ys]
                ax.scatter(xs, ys)
        plt.show()





    def plot_cumulatives(self):
        """
        Plot cumulatives of run_uropod, run_centroid etc
        """

        labeled = []


        fig = plt.figure(figsize = (30, 30))
        ax1 = fig.add_subplot(141)
        ax2 = fig.add_subplot(142)
        ax3 = fig.add_subplot(143)
        ax4 = fig.add_subplot(144)

        axes = [ax2, ax3, ax4]
        attributes = ['run_centroid', 'delta_uropod', 'delta_centroid']
        norm_bools = [False, True, True]

        self._set_centroid_attributes('run')

        for lymph_series in self.cells.values():
            #linestyle = random.choice(['--', '-.', '-', ':'])
            linestyle = random.choice(['-'])
            for lymph in lymph_series:
                lymph.linestyle = linestyle

        for lymph_series in self.cells.values():
            t_res = lymph_series[0].t_res

            all_lists = utils_general.split_by_consecutive_frames(lymph_series, attribute='run_uropod', and_nan = True)

            for i in all_lists:

                runs = [j.run_uropod for j in i]
                runs_sum = np.nancumsum(runs)
                times = [t_res*i for i,j in enumerate(runs_sum)]
                color = max([np.linalg.norm(j.centroid-i[0].centroid) for j in i])
                color /= (max(times)-min(times))
                print('color', color)
                color_lim = 0.07

                if lymph_series[0].idx_cell not in labeled:
                    ax1.plot(times, runs_sum, label = lymph_series[0].idx_cell, c = (0, min(1, color/color_lim), 0), linestyle = lymph_series[0].linestyle)
                    #ax1.plot(times, runs_sum, label = lymph_series[0].idx_cell, c = lymph_series[0].color, linestyle = lymph_series[0].linestyle)
                    labeled.append(lymph_series[0].idx_cell)
                else:
                    ax1.plot(times, runs_sum, c = (0, min(1, color/color_lim), 0), linestyle = lymph_series[0].linestyle)
                    #ax1.plot(times, runs_sum, c = lymph_series[0].color, linestyle = lymph_series[0].linestyle)



            for ax, attribute, norm_bool in zip(axes, attributes, norm_bools):


                all_lists = utils_general.split_by_consecutive_frames(lymph_series, attribute='run_uropod', and_nan = True)

                for i in all_lists:
                    runs = [getattr(lymph, attribute) for  lymph in i]
                    if norm_bool:
                        runs = [np.linalg.norm(i) if i is not None else None for i in runs]
                    runs = [i if i is not None else np.nan for i in runs]

                    runs_sum = np.nancumsum(runs)
                    times = [t_res*i for i,j in enumerate(runs_sum)]
                    ax.plot(times, runs_sum, c = (0, min(1, color/color_lim), 0), linestyle = lymph_series[0].linestyle)
                    #ax.plot(times, runs_sum, c = i[0].color, linestyle = lymph_series[0].linestyle)



        #ax1.legend(bbox_to_anchor=(0, 1), loc='upper left')
        plt.show()




    def alignments(self, min_length, min_time_either_side = 50):
        """
        Find which axis cells follow when both uropod & centroid move with similar vector
        """

        for idx_cell in self.cells.keys():
            #self._set_mean_uropod_and_centroid(idx_cell = idx_cell, time_either_side = max(min_time_either_side, self.cells[idx_cell][0].mean_time_diff/2))
            self._set_mean_uropod_and_centroid(idx_cell = idx_cell, time_either_side = 100)

        self._set_run()
        self._set_centroid_attributes('searching', time_either_side = -1)

        diffs = []
        UC_uropod_angles = []
        ellipsoid_uropod_angles = []

        for lymph in utils_general.list_all_lymphs(self):

            if lymph.delta_uropod is not None and lymph.delta_centroid is not None and lymph.ellipsoid_vec is not None:

                if np.linalg.norm(lymph.delta_uropod) > min_length: # if it's moving enough
                    max_diff = np.linalg.norm(lymph.delta_uropod) /2
                    if np.linalg.norm(lymph.delta_uropod - lymph.delta_centroid) < max_diff: # if uropod & centroid are moving in same direction

                        print(lymph.idx_cell)


                        diffs.append(np.linalg.norm(lymph.delta_uropod - lymph.delta_centroid))

                        vec1 = lymph.delta_uropod
                        vec2 = lymph.mean_centroid - lymph.mean_uropod

                        cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                        if cos_angle < 0:
                            vec1 = - lymph.delta_uropod
                            cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                        UC_uropod_angle = (360/6.283)*np.arccos(cos_angle)
                        UC_uropod_angles.append(UC_uropod_angle)

                        vec1 = lymph.delta_uropod
                        vec2 = lymph.ellipsoid_vec
                        cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                        if cos_angle < 0:
                            vec1 = - lymph.delta_uropod
                            cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                        ellipsoid_uropod_angle = (360/6.283)*np.arccos(cos_angle)
                        ellipsoid_uropod_angles.append(ellipsoid_uropod_angle)

                    """
                    mins, maxs = np.min(utils_general.list_all_lymphs(self)[0].vertices, axis = 0), np.max(utils_general.list_all_lymphs(self)[0].vertices, axis = 0)
                    box = pv.Box(bounds=(mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]))
                    plotter = pv.Plotter()
                    #lymph.surface_plot(plotter = plotter, opacity = 0.5, box = box)
                    plotter.add_lines(np.array([[0, 0, 0], lymph.delta_uropod]), color = (1, 0, 0))
                    plotter.add_lines(np.array([[0, 0, 0], lymph.delta_centroid]), color = (0, 1, 0))
                    plotter.add_mesh(pv.Sphere(radius=max_diff, center=lymph.delta_uropod), color = (1, 0, 0), opacity = 0.5)
                    plotter.show(cpos=[1, 0, 0])
                    """


        plt.hist([UC_uropod_angles, ellipsoid_uropod_angles], bins=20, color = ['red', 'blue'])
        plt.show()



    def scatter_run_running_means(self):


        self._set_centroid_attributes('run')

        fig_scat = plt.figure(figsize = (20, 20))
        axes = [fig_scat.add_subplot(3, 1, i+1) for i in range(3)]
        width_points = [[] for _ in range(3)]
        for lymph_series in self.cells.values():
            print(lymph_series[0].idx_cell)
            color = lymph_series[0].color
            for idx_width, width in enumerate([-1, 50, 100]):


                self._set_run_uropod_running_means(idx_cell = lymph_series[0].idx_cell, time_either_side = width)

                run_uropod_running_means = [lymph.run_uropod_running_mean if lymph.run_uropod_running_mean is not None else np.nan for lymph in lymph_series]
                times = [lymph.frame*lymph.t_res for lymph in lymph_series]
                print('run_uropod_running_means', run_uropod_running_means)
                axes[idx_width].scatter(times, run_uropod_running_means, s = 1, c = color)
                #axes[idx_width].set_ylim(bottom=0)
                width_points[idx_width] += run_uropod_running_means




        fig_hist = plt.figure()
        for idx, i in enumerate(width_points):
            i = [j for j in i if not np.isnan(j)]
            ax = fig_hist.add_subplot(3, 1, idx+1)
            ax.hist(i, bins = 15, orientation = 'horizontal', color = 'black')
            ax.set_yticks([])
            #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


        plt.show()








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
            if attribute == 'delta_centroid':
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

        if color_by == 'delta_centroid':
            self._set_centroid_attributes(color_by)
            vmin, vmax = utils_general.get_color_lims(self, color_by = color_by)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        for idx_cell, lymph_series in self.cells.items():
            color = np.random.rand(3,)
            lymphsNested = utils_general.get_nestedList_connectedLymphs(lymph_series)

            for lymphs in lymphsNested:
                for idx in range(len(lymphs)-1):
                    if color_by == 'delta_centroid':
                        color = utils_general.get_color(lymphs[idx], color_by = color_by, vmin = vmin, vmax = vmax)
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




    def histogram(self):

        def filter(list):
            return [i for i in list if i < 0 and i > -0.04]


        for idx_cell in self.cells.keys():
            self._set_mean_uropod_and_centroid(idx_cell = idx_cell, time_either_side = self.cells[idx_cell][0].mean_time_diff/2)
        self._set_run()



        lymphs = utils_general.list_all_lymphs(self)

        run_uropods = [lymph.run_uropod for lymph in lymphs if lymph.run_uropod is not None]
        run_centroids = [lymph.run_centroid for lymph in lymphs if lymph.run_centroid is not None]


        fig = plt.figure(figsize = (10, 10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.hist([run_uropods, run_centroids], bins=30, color = ['red', 'blue'])
        ax2.hist([filter(run_uropods), filter(run_centroids)], bins=30, color = ['red', 'blue'])
        plt.show()

        print('run_uropods, std:{}, var:{}'.format(np.std(run_uropods), np.var(run_uropods)))
        print('run_centroids, std:{}, var:{}'.format(np.std(run_centroids), np.var(run_centroids)))






    def correlation(self,  attributes):
        """
        Get pearson correlation coefficient between independent and dependent variable
        """

        self._set_pca(n_components=3)
        self._set_morph_derivs()
        self._set_centroid_attributes('run')
        self._set_run_uropod_running_means(time_either_side = 80)



        fig_scatt, fig_r, fig_p = plt.figure(figsize = (20, 20)), plt.figure(), plt.figure()
        r_values = np.empty((len(attributes), len(attributes)))
        p_values = np.empty((len(attributes), len(attributes)))
        r_values[:], p_values[:] = np.nan, np.nan
        for idx_row, dependent in enumerate(attributes):
            for idx_col, independent in enumerate(attributes):
                if dependent != independent and dependent[:3] != 'pca' and idx_col < idx_row:
                    if independent[:3] != 'run':

                        print(independent, dependent)

                        ax = fig_scatt.add_subplot(len(attributes), len(attributes), idx_row*len(attributes)+idx_col+1)
                        #ax.set_xlabel(independent)
                        #ax.set_ylabel(dependent)

                        plot_lymphs = [lymph for lymph_series in self.cells.values() for lymph in lymph_series if getattr(lymph, independent) is not None and  getattr(lymph, dependent) is not None]
                        xs = [getattr(lymph, independent) for lymph in plot_lymphs]
                        ys = [getattr(lymph, dependent)  for lymph in plot_lymphs]





                        colors = [lymph.color  for lymph in plot_lymphs]
                        result = scipy.stats.linregress(np.array(xs), np.array(ys))
                        ax.scatter(xs, ys, s=1, c = colors)

                        model_xs = np.linspace(min(list(xs)), max(list(xs)), 50)
                        #ax.plot(model_xs, [result.slope*i+result.intercept for i in model_xs], c = 'red')
                        ax.tick_params(axis="both",direction="in")
                        if idx_row != len(attributes)-1:
                            ax.set_xticks([])
                        if idx_col != 0:
                            ax.set_yticks([])

                        r_values[idx_row, idx_col] = result.rvalue
                        p_values[idx_row, idx_col] = result.pvalue
        fig_scatt.subplots_adjust(hspace=0, wspace=0)
        ax = fig_r.add_subplot(111)
        r_extreme = np.nanmax(abs(r_values))
        print(r_values, r_extreme)
        r = ax.imshow(r_values, cmap = 'PiYG', vmin = -r_extreme, vmax = r_extreme)
        matplotlib.cm.Blues.set_bad(color='white')
        fig_r.colorbar(r, ax=ax, orientation='horizontal')
        ax = fig_p.add_subplot(111)
        p = ax.imshow(p_values, cmap = 'Reds')
        matplotlib.cm.Reds.set_bad(color='white')
        fig_p.colorbar(p, ax=ax, orientation='horizontal')



        plt.show()


    def correlation_annotate(self,  independent, dependent):

        fig = plt.figure()

        if independent[:3] == 'pca' or dependent[:3] == 'pca' :
            self._set_pca(n_components=3)
        if independent == 'delta_centroid' or dependent == 'delta_centroid' or independent == 'delta_uropod' or dependent == 'delta_uropod':
            self._set_centroid_attributes('delta_centroid_uropod')
        if independent[:4] == 'spin' or dependent[:4] == 'spin' or  independent == 'angle' or dependent == 'angle':
            self._set_centroid_attributes('searching', time_either_side = None)
        if independent[:3] == 'run' or dependent[:3] == 'run':
            self._set_centroid_attributes('run', time_either_side = 7)
            lymphs = utils_general.list_all_lymphs(self)


        if independent == 'morph_deriv' or dependent == 'morph_deriv':
            self._set_morph_derivs()




        plot_lymphs = [lymph for lymph_series in self.cells.values() for lymph in lymph_series if getattr(lymph, independent) is not None and  getattr(lymph, dependent) is not None]
        xs = [getattr(lymph, independent) for lymph in plot_lymphs]
        ys = [getattr(lymph, dependent)  for lymph in plot_lymphs]
        colors = [lymph.color  for lymph in plot_lymphs]

        names = [lymph.idx_cell + '-{}'.format(lymph.frame) for lymph in plot_lymphs]

        ax = fig.add_subplot(111)
        sc = ax.scatter(xs, ys, s = 0.5, c = colors)

        ax.set_xlabel(independent)
        ax.set_ylabel(dependent)


        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = "{}".format(" ".join([names[n] for n in ind["ind"]]))
            annot.set_text(text)
            #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
        annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)



        fig.canvas.mpl_connect("motion_notify_event", hover)
        fig2 = plt.figure()
        i = [j for j in i if not np.isnan(j)]
        ax = fig2.add_subplot(1, 1, 1)
        ax.hist(i, bins = 10, orientation = 'horizontal')

        plt.show()




    def plot_rotations(self,  time_either_side):
        self._set_centroid_attributes('searching', time_either_side = time_either_side)


        plotter = pv.Plotter(shape=(2, 4), border=False)


        for idx, lymphs_plot in enumerate(self.cells.values()):
            plotter.subplot(0, idx)
            for idx_plot, lymph in enumerate(lymphs_plot):


                vec = lymph.spin_vec
                if vec is not None:
                    plotter.add_lines(np.array([np.array([0, 0, 0]), vec]), color = (1, idx_plot/(len(lymphs_plot)-1), 1))
                    plotter.add_lines(np.array([[0, 0, 0], [0.005, 0, 0]]), color = (0.9, 0.9, 0.9))
                    plotter.add_lines(np.array([[0, 0, 0], [0, 0.005, 0]]), color = (0.9, 0.9, 0.9))
                    plotter.add_lines(np.array([[0, 0, 0], [0, 0, 0.005]]), color = (0.9, 0.9, 0.9))

            plotter.subplot(1, idx)
            for idx_plot, lymph in enumerate(lymphs_plot):
                vec = lymph.ellipsoid_vec_smoothed
                if vec is not None:
                    plotter.add_lines(np.array([-vec, vec]), color = (1, idx_plot/(len(lymphs_plot)-1), 1))

        plotter.show()






    def gather_time_series(self, save_name):
        """
        Gather shape time series into dictionary with sub dictionaries containing joint (or gaps of size 1) frame series
        """


        self._set_pca(n_components=3)
        self._set_centroid_attributes('run')
        self._set_run_uropod_running_means(time_either_side = 80)
        self._set_searching(time_either_side = 75)


        all_consecutive_frames = []

        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        for idx_cell, lymph_series in self.cells.items():
            print('Gathering',  idx_cell)

            count = 0
            consecutive_frames = Consecutive_Frames(name = str(idx_cell)+alphabet[count], t_res_initial = lymph_series[0].t_res)
            prev_frame = None
            for idx_lymph, lymph in enumerate(lymph_series):
                if idx_lymph == 0 or lymph.frame-prev_frame == 1:
                    consecutive_frames.add(lymph.frame, lymph.pca[0], lymph.pca[1], lymph.pca[2],  lymph.run_uropod, lymph.run_uropod_running_mean, lymph.turning)

                else:
                    consecutive_frames.interpolate()
                    all_consecutive_frames.append(consecutive_frames)
                    count += 1
                    consecutive_frames = Consecutive_Frames(name = str(idx_cell)+alphabet[count], t_res_initial = lymph_series[0].t_res)
                    consecutive_frames.add(lymph.frame, lymph.pca[0], lymph.pca[1], lymph.pca[2], lymph.run_uropod, lymph.run_uropod_running_mean, lymph.turning)
                prev_frame = lymph.frame

            consecutive_frames.interpolate()
            all_consecutive_frames.append(consecutive_frames)

        pickle_out = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}.pickle'.format(save_name),'wb')
        pickle.dump(all_consecutive_frames, pickle_out)
