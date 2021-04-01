import numpy as np
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
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import pickle
import random

from lymphocytes.lymph_serieses.pca_methods import PCA_Methods
from lymphocytes.lymph_serieses.single_cell_methods import Single_Cell_Methods
from lymphocytes.lymph_serieses.centroid_variable_methods import Centroid_Variable_Methods
from lymphocytes.lymph_snap.lymph_snap_class import Lymph_Snap


import lymphocytes.utils.voxels as utils_voxels
import lymphocytes.utils.plotting as utils_plotting
import lymphocytes.utils.general as utils_general



class Lymph_Serieses(Single_Cell_Methods, PCA_Methods, Centroid_Variable_Methods):
    """
    Class for all lymphocyte serieses.
    Mixins are:
    - PCA_Methods: methods without involving reduced-dimensionality representation (via PCA).
    - Single_Cell_Methods: methods suitable for a single cell series.
    """


    def __init__(self, stack_triplets, cells_model, max_l):
        """
        Args:
        -- stack_triplets: (mat_filename, coeffPathFormat, zoomedVoxelsPathFormat) triplet, where...
        - mat_filename: filename of .mat file containing all initial info.
        - coeffPathFormat: start path (without frame number) of SPHARM coefficients.
        - zoomedVoxelsPathFormat: start path (without frame number) for zoomed voxels (for speed reasons).
        """

        self.lymph_serieses = {}
        self.uropod_files = ['/Users/harry/OneDrive - Imperial College London/lymphocytes/uropods/cell_{}.pickle'.format(idx_cell) for idx_cell in range(len(stack_triplets))]

        for idx_cell, (mat_filename, coeffPathFormat, zoomedVoxelsPathFormat) in enumerate(stack_triplets):
            if cells_model == 'all' or idx_cell in cells_model:
                lymph_series = []

                f = h5py.File(mat_filename, 'r')
                frames = f['OUT/FRAME']

                max_frame = int(np.max(np.array(frames)))

                uropods = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/uropods/cell_{}.pickle'.format(idx_cell),"rb"))

                for frame in range(1, max_frame+1):
                    if np.any(np.array(frames) == frame) and os.path.isfile(coeffPathFormat.format(frame)): # if it's within arena and SPHARM-PDM worked
                        snap = Lymph_Snap(mat_filename = mat_filename, frame = frame, coeffPathFormat = coeffPathFormat, zoomedVoxelsPathFormat = zoomedVoxelsPathFormat, idx_cell = idx_cell, max_l = max_l, uropod = np.array(uropods[frame]))

                        lymph_series.append(snap)

                self.lymph_serieses[idx_cell] = lymph_series
                print('series added')

        self.num_serieses = len(self.lymph_serieses)


    def plot_volumes(self):
        """
        Plot volume changes as calculated from original voxel representations.
        Args:
        - zoom_factor: subsampling factor for the voxels.
        """
        for lymph_series in self.lymph_serieses.values():
            plt.plot([lymph.frame for lymph in lymph_series if lymph is not None], [lymph.volume for lymph in lymph_series if lymph is not None])
        plt.show()



    def plot_RIvector_mean_var(self):
        """
        Plot the mean and standard deviation of rotationally-invariant shape descriptor.
        Args:
        - maxl: truncations.
        """

        lymphs = utils_general.list_all_lymphs(self)
        vectors = np.array([lymph.RI_vector for lymph in lymphs])

        means = np.mean(vectors, axis = 0)
        vars = np.var(vectors, axis = 0)

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.bar(range(len(means)), means, color = 'red')
        ax = fig.add_subplot(1, 2, 2)
        ax.bar(range(len(vars)), vars, color = 'blue')

        plt.show()


    def plot_recons_increasing_l(self, maxl, l):
        """
        Plot a sample of reconstructions in order of ascending l.
        Args:
        - maxl: the truncation of the representation.
        - l: which l (energy) to order by (maximum l possible is the truncation, maxl).
        """
        lymphs = utils_general.list_all_lymphs(self)
        vectors = [lymph.RI_vector for lymph in lymphs]

        ls = [vec[l] for vec in vectors]
        lymphs = [j for i,j in sorted(zip(ls, lymphs))][::int(len(ls)/8)]

        fig = plt.figure()
        for i in range(len(lymphs)):
            ax = fig.add_subplot(1, len(lymphs), i+1, projection = '3d')
            lymphs[i].plotRecon_singleDeg(ax)
        equal_axes(*fig1.axes)



    def line_plot_3D(self, centroid_uropod_pca, color_by):
        """
        3D plot the first 3 descriptors moving in time.
        """
        if color_by == 'speed':
            self._set_speeds()
            vmin, vmax = utils_general.get_color_lims(self, color_by)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        for idx_cell, lymph_series in self.lymph_serieses.items():
            color = np.random.rand(3,)
            if centroid_uropod_pca == 'centroid':
                vectorsNested = utils_general.get_nestedList_connectedFrames(lymph_series, 'centroid')
            elif centroid_uropod_pca == 'uropod':
                vectorsNested = utils_general.get_nestedList_connectedFrames(lymph_series, 'uropod')
            elif centroid_uropod_pca == 'pca':
                self._set_pca(n_components=3,  removeSpeedNone = False, removeAngleNone = False)
                vectorsNested = utils_general.get_nestedList_connectedFrames(lymph_series, 'pca')


            if color_by == 'speed':

                for vectors in vectorsNested:
                    for idx in range(len(vectors)-1):
                        if getattr(lymph_series[idx], color_by) is not None:
                            color = utils_general.get_color(lymph_series[idx], color_by = color_by, vmin = vmin, vmax = vmax)
                            ax.plot((vectors[idx][0], vectors[idx+1][0]), (vectors[idx][1], vectors[idx+1][1]), (vectors[idx][2], vectors[idx+1][2]), c = color)

            else:
                for vectors in vectorsNested:
                    ax.plot([v[0] for v in vectors], [v[1] for v in vectors], [v[2] for v in vectors], c = color)
            ax.set_xlabel('#0')
            ax.set_ylabel('#1')
            ax.set_zlabel('#2')

        if centroid_uropod_pca == 'centroid' or centroid_uropod_pca == 'uropod':
            utils_plotting.set_limits_3D(*fig.axes)
        utils_plotting.label_axes_3D(*fig.axes)
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
        lymph.surface_plot()




    def _scatter_plotted_components(self, vectors, plotted_points_all):
        """
        Scatter points of the plotted meshes
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in vectors:
            ax.scatter(i[0], i[1], i[2], s = 1, c = 'lightskyblue')
        for i, color in zip(plotted_points_all, ['red', 'green', 'black']):
            for j in i:
                ax.scatter(j[0], j[1], j[2], s = 6, c = color)



    def plot_component_lymphs(self, grid_size, pca):
        """
        Plot seperate sampling of each of the 3 components (meshes and scatter)
        """
        plotter = pv.Plotter(shape=(3, grid_size))
        plotted_points_all = []

        lymphs = utils_general.list_all_lymphs(self)
        random.shuffle(lymphs)

        if pca:
            self._set_pca(n_components = 3)
            vectors = [lymph.pca for lymph in lymphs]
        else:
            vectors = [lymph.RI_vector for lymph in lymphs]



        for idx_component in range(3):
            plotted_points = []

            min_ = min([v[idx_component] for v in vectors])
            max_ = max([v[idx_component] for v in vectors])
            range_ = max_ - min_

            for grid in range(grid_size):
                grid_vectors = [] # vectors that could be good for this part of the PC
                grid_lymphs = []
                for vector, lymph in zip(vectors, lymphs):
                    if int((vector[idx_component] - min_) // (range_/grid_size)) == grid:
                        grid_vectors.append(vector)
                        grid_lymphs.append(lymph)
                popped = np.array([np.delete(i, idx_component) for i in grid_vectors])
                dists_from_PC = [np.sqrt(np.sum(np.square(i))) for i in popped]
                idx_min = dists_from_PC.index(min(dists_from_PC))
                to_plot = grid_lymphs[idx_min]
                plotted_points.append(grid_vectors[idx_min])
                plotter.subplot(idx_component, grid)
                #to_plot.surface_plot(plotter, uropod_align=True)
                to_plot.plotRecon_singleDeg(plotter, max_l = 2, uropod_align = True)
            plotted_points_all.append(plotted_points)

        plotter.show(cpos=[0, 1, 0])
        self._scatter_plotted_components(vectors, plotted_points_all)


    def plot_2D_embeddings(self, pca, components):
        """
        Plot the meshes at their embeddings for 2 components
        """
        lymphs = utils_general.list_all_lymphs(self)
        if pca:
            self._set_pca(n_components = 3)


        random.shuffle(lymphs)

        lymphs, vectors = lymphs[::7], vectors[::7]
        plotter = pv.Plotter()
        for lymph in lymphs:
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
