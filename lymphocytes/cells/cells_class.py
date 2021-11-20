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

        stack_attributes_dict = {i[0]:i for i in stack_attributes}

        self.cells = {}

        for idx_cell in cells_model:

            (idx_cell, mat_filename, coeffPathFormat, xyz_res, color, t_res) = stack_attributes_dict[idx_cell]

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
            for frame in range(int(max(frames_all)+1)):

                if os.path.isfile(coeffPathFormat.format(frame)): # if it's within arena and SPHARM-PDM worked

                    #if np.random.randint(0, 200) == 5:

                    idx = frames_all.index(frame)
                    snap = Cell_Frame(mat_filename = mat_filename, frame = frames_all[idx], coeffPathFormat = coeffPathFormat, voxels = voxels_all[idx], xyz_res = xyz_res,  idx_cell = idx_cell, max_l = max_l, uropod = np.array(uropods[frames_all[idx]]), vertices = vertices_all[idx], faces = faces_all[idx])
                    #snap = Cell_Frame(mat_filename = mat_filename, frame = frame, coeffPathFormat = coeffPathFormat, voxels = voxels_all[idx], xyz_res = xyz_res,  idx_cell = idx_cell, max_l = max_l, uropod  = None,  vertices = vertices_all[idx], faces = faces_all[idx])

                    snap.color = color
                    snap.t_res = t_res
                    lymph_series.append(snap)


            self.cells[idx_cell] = lymph_series
            print('max_frame: {}'.format(max(frames_all)))

        self.pca_set = False
        self.attributes_set = []



    def curvatures(self):

        for lymph in utils_general.list_all_lymphs(self):

            surf = pv.PolyData(lymph.vertices, lymph.faces)

            surf = surf.smooth(n_iter=5000)
            surf = surf.decimate(0.98)
            curvature = surf.curvature()
            #surf_tree = KDTree(surf.points.astype(np.double))
            #dist, idx = surf_tree.query(np.array([[lymph.uropod[0], lymph.uropod[1], lymph.uropod[2]]]))

            curvatures = surf.curvature('Minimum')


            plotter = pv.Plotter()
            plotter.add_mesh(surf, scalars = curvatures, clim = [-0.5, 1])

            idxs = surf.find_closest_point(lymph.uropod, n = 10)
            mean_curvature = np.mean([curvatures[idx] for idx in idxs])
            points = [surf.points[idx] for idx in idxs]
            for point in points:
                plotter.add_mesh(pv.Sphere(radius=0.1, center=point), color = (0, 0, 1))


            plotter.add_lines(np.array([lymph.uropod, lymph.uropod + np.array([0, 0, 0.5/mean_curvature])]), color = (0, 0, 0))

            #uropods = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/uropods/cell_{}.pickle'.format(lymph.idx_cell), "rb"))
            #for uropod in uropods.values():
            #    plotter.add_mesh(pv.Sphere(radius=0.3, center=uropod), color = (0, 1, 0))

            plotter.add_mesh(pv.Sphere(radius=0.3, center=lymph.uropod), color = (1, 0, 0))
            plotter.add_text("{}".format(mean_curvature), font_size=10)
            plotter.show()



    def rear_orientations(self):

        def _plane(num_points, plotter, color):
            idxs = surf.find_closest_point(lymph.uropod, n = num_points)
            vertices = [lymph.vertices[idx, :] for idx in idxs]

            pca_obj = PCA(n_components = 2)
            pca_obj.fit(np.array(vertices))
            normal = np.cross(pca_obj.components_[0, :], pca_obj.components_[1, :])

            plane = pv.Plane(center=lymph.uropod, direction=normal, i_size=4, j_size=4)
            plotter.add_mesh(plane, opacity = 0.5, color = color)

            points = [surf.points[idx] for idx in idxs]
            for point in points:
                plotter.add_mesh(pv.Sphere(radius=0.1, center=point), color = color)


        for lymph in utils_general.list_all_lymphs(self):
            surf = pv.PolyData(lymph.vertices, lymph.faces)

            plotter = pv.Plotter()

            _plane(num_points=10, plotter=plotter, color = (0, 0, 1))

            _plane(num_points=50, plotter=plotter, color = (0, 1, 0))

            plotter.add_mesh(surf, clim = [-0.5, 1])
            plotter.add_mesh(pv.Sphere(radius=0.15, center=lymph.uropod), color = (1, 0, 0))



            plotter.show()




    def scatter_run_running_means(self):
        fig_scat = plt.figure()
        axes = [fig_scat.add_subplot(5, 1, i+1) for i in range(5)]
        width_points = [[] for _ in range(5)]
        for lymph_series in self.cells.values():
            print(lymph_series[0].idx_cell)
            color = np.random.rand(3,)
            for idx_width, width in enumerate([7, 50, 100, 150, 200]):

                self._set_centroid_attributes_to_NONE()
                self.attributes_set = []
                self._set_centroid_attributes('run', time_either_side = width, idx_cell = lymph_series[0].idx_cell)

                runs = [lymph.run if lymph.run is not None else np.nan for lymph in lymph_series]
                print(np.nanmax(runs))
                times = [lymph.frame*lymph.t_res for lymph in lymph_series]
                axes[idx_width].scatter(times, runs, s = 5, c = color)
                width_points[idx_width] += runs
        fig_hist = plt.figure()
        for idx, i in enumerate(width_points):
            i = [j for j in i if not np.isnan(j)]
            ax = fig_hist.add_subplot(5, 1, idx+1)
            ax.hist(i, bins = 10, orientation = 'horizontal')
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


        vars = np.std(vectors, axis = 0)

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.bar(range(len(means)), means, color = 'red')
        ax = fig.add_subplot(1, 2, 2)
        ax.bar(range(len(vars)), vars, color = 'blue')

        plt.show()





    def plot_centroids(self,  plot_every):
        idx_cells = []
        mean_centroids = []
        plotter = pv.Plotter()
        colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
        for idx, lymph_series in enumerate(self.cells.values()):
            idx_cells.append(lymph_series[0].idx_cell)

            lymphs_plot = lymph_series[::plot_every]
            for idx_plot, lymph in enumerate(lymphs_plot):
                plotter.add_mesh(pv.Sphere(radius=0.1, center=lymph.centroid), color = colors[idx])

            mean_centroid = np.mean(np.array([lymph.centroid for lymph in lymph_series]), axis = 0)
            mean_centroids.append(mean_centroid)

        poly = pv.PolyData(np.vstack(mean_centroids))
        poly["My Labels"] = idx_cells
        plotter.add_point_labels(poly, "My Labels")
        plotter.show()





    def plot_mean_directions_and_spin_vecs(self, time_either_side, time_either_side_2):
        self._set_centroid_attributes('searching', time_either_side = time_either_side, time_either_side_2 = time_either_side_2, idx_cell = None)
        plotter = pv.Plotter(shape = (2, len(list(self.cells.values()))))

        for idx, lymph_series in enumerate(self.cells.values()):
            print('h',lymph_series[0].idx_cell)

            lymphs_plot = lymph_series[::7]
            plotter.subplot(0, idx)
            plotter.add_text("{}".format(lymphs_plot[0].idx_cell), font_size=10)
            for idx_plot, lymph in enumerate(lymphs_plot):
                if lymph.direction_mean is not None:
                    plotter.add_lines(np.array([[0, 0, 0], lymph.direction_mean]), color = (1, idx_plot/(len(lymphs_plot)-1), 1))
            plotter.add_axes()
            plotter.add_lines(np.array([[-0.5, 0, 0], [0.5, 0, 0]]), color = (0, 0, 0))
            plotter.add_lines(np.array([[0, -0.5, 0], [0, 0.5, 0]]), color = (0, 0, 0))
            plotter.add_lines(np.array([[0, 0, -0.5], [0, 0, 0.5]]), color = (0, 0, 0))

            plotter.subplot(1, idx)
            plotter.add_text("{}".format(lymphs_plot[0].idx_cell), font_size=10)
            for idx_plot, lymph in enumerate(lymphs_plot):
                if lymph.spin_vec_2 is not None:
                    plotter.add_lines(np.array([[0, 0, 0], lymph.spin_vec_2]), color = (1, idx_plot/(len(lymphs_plot)-1), 1))
            plotter.add_axes()

            plotter.add_lines(np.array([[-0.001, 0, 0], [0.001, 0, 0]]), color = (0, 0, 0))
            plotter.add_lines(np.array([[0, -0.001, 0], [0, 0.001, 0]]), color = (0, 0, 0))
            plotter.add_lines(np.array([[0, 0, -0.001], [0, 0, 0.001]]), color = (0, 0, 0))

            spin_vec_2_list = [lymph.spin_vec_2 for lymph in lymph_series if lymph.spin_vec_2 is not None]
            num = len(spin_vec_2_list)
            print('mean', np.mean([np.linalg.norm(i) for i in spin_vec_2_list]))
            print('std', np.sum(np.std(np.array(spin_vec_2_list), axis = 0))/num)
            print('var', np.sum(np.var(np.array(spin_vec_2_list), axis = 0))/num)




        plotter.show()




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




    def correlation(self,  attributes, widths):
        """
        Get pearson correlation coefficient between independent and dependent variable
        """
        for width in widths:
            self.attributes_set = []
            self._set_centroid_attributes_to_NONE()

            fig_scatt, fig_r, fig_p = plt.figure(figsize = (5, 6)), plt.figure(), plt.figure()
            r_values = np.empty((len(attributes), len(attributes)))
            p_values = np.empty((len(attributes), len(attributes)))
            r_values[:], p_values[:] = np.nan, np.nan
            for idx_row, dependent in enumerate(attributes):
                for idx_col, independent in enumerate(attributes):
                    if dependent != independent and dependent[:3] != 'pca' and idx_col < idx_row:

                        print(independent, dependent)

                        ax = fig_scatt.add_subplot(len(attributes), len(attributes), idx_row*len(attributes)+idx_col+1)
                        ax.set_xlabel(independent)
                        ax.set_ylabel(dependent)
                        if independent[:3] == 'pca' or dependent[:3] == 'pca' :
                            self._set_pca(n_components=3)
                        if independent == 'delta_centroid' or dependent == 'delta_centroid' or independent == 'delta_uropod' or dependent == 'delta_uropod':
                            self._set_centroid_attributes('delta_centroid_uropod')
                        if independent[:4] == 'spin' or dependent[:4] == 'spin' or  independent[:3] == 'dir' or dependent[:3] == 'dir':
                            self._set_centroid_attributes('searching', time_either_side = width)
                        if independent[:3] == 'run' or dependent[:3] == 'run':
                            self._set_centroid_attributes('run', time_either_side = width)
                            lymphs = utils_general.list_all_lymphs(self)


                        if independent == 'morph_deriv' or dependent == 'morph_deriv':
                            self._set_morph_derivs()

                        plot_lymphs = [lymph for lymph_series in self.cells.values() for lymph in lymph_series if getattr(lymph, independent) is not None and  getattr(lymph, dependent) is not None]
                        xs = [getattr(lymph, independent) for lymph in plot_lymphs]
                        ys = [getattr(lymph, dependent)  for lymph in plot_lymphs]

                        colors = [lymph.color  for lymph in plot_lymphs]
                        result = scipy.stats.linregress(np.array(xs), np.array(ys))
                        ax.scatter(xs, ys, s=1, c = colors)
                        ax.set_title(str(width))

                        model_xs = np.linspace(min(list(xs)), max(list(xs)), 50)
                        ax.plot(model_xs, [result.slope*i+result.intercept for i in model_xs], c = 'red')
                        print('{}, slope: {}'.format(width, result.slope))
                        ax.tick_params(axis="both",direction="in")
                        if idx_row != len(attributes)-1:
                            ax.set_xticks([])
                        if idx_col != 0:
                            ax.set_yticks([])

                        r_values[idx_row, idx_col] = result.rvalue
                        p_values[idx_row, idx_col] = result.pvalue
            fig_scatt.subplots_adjust(hspace=0, wspace=0)
            ax = fig_r.add_subplot(111)
            ax.set_title(str(width))
            r = ax.imshow(r_values, cmap = 'Blues')
            matplotlib.cm.Blues.set_bad(color='white')
            ax.axis('off')
            fig_r.colorbar(r, ax=ax, orientation='horizontal')
            ax = fig_p.add_subplot(111)
            ax.set_title(str(width))
            p = ax.imshow(p_values, cmap = 'Reds')
            matplotlib.cm.Reds.set_bad(color='white')
            fig_p.colorbar(p, ax=ax, orientation='horizontal')
            ax.axis('off')



            plt.show()


    def correlation_annotate(self,  independent, dependent):

        fig = plt.figure()

        if independent[:3] == 'pca' or dependent[:3] == 'pca' :
            self._set_pca(n_components=3)
        if independent == 'delta_centroid' or dependent == 'delta_centroid' or independent == 'delta_uropod' or dependent == 'delta_uropod':
            self._set_centroid_attributes('delta_centroid_uropod')
        if independent == 'delta_sensing_direction' or dependent == 'delta_sensing_direction':
            self._set_centroid_attributes('delta_sensing_direction')
        if independent[:4] == 'spin' or dependent[:4] == 'spin' or  independent[:3] == 'dir' or dependent[:3] == 'dir':
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


        annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

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

        fig.canvas.mpl_connect("motion_notify_event", hover)
        fig2 = plt.figure()
        i = [j for j in i if not np.isnan(j)]
        ax = fig2.add_subplot(1, 1, 1)
        ax.hist(i, bins = 10, orientation = 'horizontal')

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





    def gather_time_series(self, save_name):
        """
        Gather shape time series into dictionary with sub dictionaries containing joint (or gaps of size 1) frame series
        """


        self._set_pca(n_components=3)
        self._set_centroid_attributes('delta_centroid')
        self._set_centroid_attributes('delta_sensing_direction')
        self._set_centroid_attributes('run', time_either_side = 7)
        self._set_centroid_attributes('run_mean', time_either_side = 200)
        self._set_centroid_attributes('searching', time_either_side = 50,  time_either_side_2 = 50)

        """
        for lymph in utils_general.list_all_lymphs(self):
            lymph.pca = lymph.RI_vector
        """


        all_consecutive_frames = []

        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        for idx_cell, lymph_series in self.cells.items():

            count = 0
            consecutive_frames = Consecutive_Frames(name = str(idx_cell)+alphabet[count], t_res_initial = lymph_series[0].t_res)
            prev_values = [None, None, None, None, None, None, None, None, None, None, None, None, None]
            for idx_lymph, lymph in enumerate(lymph_series):
                if idx_lymph == 0 or lymph.frame-prev_values[0] == 1:
                    consecutive_frames.add(lymph.frame, lymph.pca[0], lymph.pca[1], lymph.pca[2], lymph.RI_vector0, lymph.delta_centroid, lymph.delta_sensing_direction, lymph.run, lymph.run_mean, lymph.spin_vec_magnitude, lymph.spin_vec_magnitude_mean, lymph.spin_vec_std, lymph.direction_std)
                elif lymph.frame-prev_values[0] == 2: #linear interpolation if only 1 frame missing


                    staged_list = []
                    for attribute in ['delta_centroid', 'delta_sensing_direction', 'run', 'run_mean', 'spin_vec_magnitude', 'spin_vec_magnitude_mean', 'spin_vec_std', 'direction_std']:
                        attribute_list = attribute + '_list'
                        if getattr(consecutive_frames, attribute_list)[-1] is None or getattr(lymph, attribute) is None:
                            staged = None
                        else:
                            staged = (getattr(consecutive_frames, attribute_list)[-1]+getattr(lymph, attribute))/2
                        staged_list.append(staged)



                    consecutive_frames.add(lymph.frame, (consecutive_frames.pca0_list[-1]+lymph.pca[0])/2, (consecutive_frames.pca1_list[-1]+lymph.pca[1])/2, (consecutive_frames.pca2_list[-1]+lymph.pca[2])/2, (consecutive_frames.RI_vector0_list[-1]+lymph.RI_vector0)/2, *staged_list)
                    consecutive_frames.add(lymph.frame, lymph.pca[0], lymph.pca[1], lymph.pca[2], lymph.RI_vector0, lymph.delta_centroid, lymph.delta_sensing_direction, lymph.run, lymph.run_mean, lymph.spin_vec_magnitude, lymph.spin_vec_magnitude_mean, lymph.spin_vec_std, lymph.direction_std)

                else:
                    consecutive_frames.interpolate()
                    all_consecutive_frames.append(consecutive_frames)
                    count += 1
                    consecutive_frames = Consecutive_Frames(name = str(idx_cell)+alphabet[count], t_res_initial = lymph_series[0].t_res)
                    consecutive_frames.add(lymph.frame, lymph.pca[0], lymph.pca[1], lymph.pca[2], lymph.RI_vector0, lymph.delta_centroid, lymph.delta_sensing_direction, lymph.run, lymph.run_mean, lymph.spin_vec_magnitude, lymph.spin_vec_magnitude_mean, lymph.spin_vec_std, lymph.direction_std)
                prev_values = [lymph.frame, lymph.pca[0], lymph.pca[1], lymph.pca[2], lymph.RI_vector0, lymph.delta_centroid, lymph.delta_sensing_direction, lymph.run, lymph.run_mean, lymph.spin_vec_magnitude, lymph.spin_vec_magnitude_mean, lymph.spin_vec_std, lymph.direction_std]

            consecutive_frames.interpolate()
            all_consecutive_frames.append(consecutive_frames)

        pickle_out = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}.pickle'.format(save_name),'wb')
        pickle.dump(all_consecutive_frames, pickle_out)
