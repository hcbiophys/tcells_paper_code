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
import random
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob

from lymphocytes.lymph_serieses.pca_methods import PCA_Methods
from lymphocytes.lymph_serieses.single_cell_methods import Single_Cell_Methods
from lymphocytes.lymph_serieses.centroid_variable_methods import Centroid_Variable_Methods
from lymphocytes.lymph_snap.lymph_snap_class import Lymph_Snap


import lymphocytes.utils.voxels as utils_voxels
import lymphocytes.utils.plotting as utils_plotting


class Lymph_Serieses(Single_Cell_Methods, PCA_Methods, Centroid_Variable_Methods):
    """
    Class for all lymphocyte serieses.
    Mixins are:
    - PCA_Methods: methods without involving reduced-dimensionality representation (via PCA).
    - Single_Cell_Methods: methods suitable for a single cell series.
    """


    def __init__(self, stack_triplets, max_l):
        """
        Args:
        -- stack_triplets: (mat_filename, coeffPathFormat, zoomedVoxelsPathFormat) triplet, where...
        - mat_filename: filename of .mat file containing all initial info.
        - coeffPathFormat: start path (without frame number) of SPHARM coefficients.
        - zoomedVoxelsPathFormat: start path (without frame number) for zoomed voxels (for speed reasons).
        """

        self.lymph_serieses = []

        for (mat_filename, coeffPathFormat, zoomedVoxelsPathFormat) in stack_triplets:

            lymph_series = []

            f = h5py.File(mat_filename, 'r')
            frames = f['OUT/FRAME']

            max_frame = int(np.max(np.array(frames)))
            for frame in range(1, max_frame+1):
                if np.any(np.array(frames) == frame) and os.path.isfile(coeffPathFormat.format(frame)): # if it's within arena and SPHARM-PDM worked
                    snap = Lymph_Snap(mat_filename = mat_filename, frame = frame, coeffPathFormat = coeffPathFormat, zoomedVoxelsPathFormat = zoomedVoxelsPathFormat, max_l = max_l)
                    lymph_series.append(snap)

            self.lymph_serieses.append(lymph_series)

        self.num_serieses = len(self.lymph_serieses)


    def plot_volumes(self):
        """
        Plot volume changes as calculated from original voxel representations.
        Args:
        - zoom_factor: subsampling factor for the voxels.
        """
        for lymph_series in self.lymph_serieses:
            plt.plot([lymph.frame for lymph in lymph_series if lymph is not None], [lymph.volume for lymph in lymph_series if lymph is not None])
        plt.show()



    def plot_rotInvRep_mean_std(self):
        """
        Plot the mean and standard deviation of rotationally-invariant shape descriptor.
        Args:
        - maxl: truncations.
        """

        vectors = np.array([lymph.RI_vector for lymph_series in self.lymph_serieses for lymph in lymph_series if lymph])

        means = np.mean(vectors, axis = 0)
        stds = np.std(vectors, axis = 0)

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.bar(range(len(means)), means, color = 'red')
        ax = fig.add_subplot(1, 2, 2)
        ax.bar(range(len(stds)), stds, color = 'blue')

        plt.show()


    def plot_recons_increasing_l(self, maxl, l):
        """
        Plot a sample of reconstructions in order of ascending l.
        Args:
        - maxl: the truncation of the representation.
        - l: which l (energy) to order by (maximum l possible is the truncation, maxl).
        """
        lymphs = [lymph for lymph_series in self.lymph_serieses for lymph in lymph_series]
        vectors = [lymph.RI_vector for lymph in lymphs]

        ls = [vec[l] for vec in vectors]
        lymphs = [j for i,j in sorted(zip(ls, lymphs))][::int(len(ls)/8)]

        fig = plt.figure()
        for i in range(len(lymphs)):
            ax = fig.add_subplot(1, len(lymphs), i+1, projection = '3d')
            lymphs[i].plotRecon_singleDeg(ax)
        equal_axes(*fig1.axes)


    def scatter_first2_descriptors(self, pca):
        for lymph_series in self.lymph_serieses:
            color = np.random.rand(3,)
            if pca:
                self._set_pca(n_components=5,  removeSpeedNone = False, removeAngleNone = False)
                vectors = [lymph.pca[:2] for lymph in lymph_series]
            else:
                vectors = [lymph.RI_vector[:2] for lymph in lymph_series]
            plt.plot([v[0] for v in vectors], [v[1] for v in vectors], c = color)


    def plot_2Dmanifold(self, grid_size, pca, just_x = False, just_y = False):
        """
        Plot the manifold of the rotationally-invariant description projected to 2D via PCA.
        Args:
        - grid_size: resolution of the manifol, i.e. dimensions of grid to show images in.
        - max_l: truncation degree.
        - pca: whether PCA used. If False, 2D representation is the first two energies (associated with first two ls).
        - just_x: if True, shows 1D manifold along x.
        - just_y: if True, shows 1D manifold along y.
        """
        lymphs = [lymph for lymph_series in self.lymph_serieses for lymph in lymph_series]
        if pca:
            vectors = [lymph.pca for lymph in lymphs]
        else:
            vectors = [lymph.RI_vector for lymph in lymphs]

        # shuffle so covers many cells
        c = list(zip(vectors, lymphs))
        random.shuffle(c)
        vectors, lymphs = zip(*c)

        min1 = min([v[0] for v in vectors])
        max1 = max([v[0] for v in vectors])
        min2 = min([v[1] for v in vectors])
        max2 = max([v[1] for v in vectors])
        range1 = max1 - min1
        range2 = max2 - min2

        if just_x or just_y:
            grid_size = 10

        fig = plt.figure(figsize = (10, 10))
        grids_done = []
        for idx, vector in enumerate(vectors):

            grid1 = (vector[0] - min1) // (range1/grid_size)
            grid2 = (vector[1] - min2) // (range2/grid_size)

            if not [grid1, grid2] in grids_done:

                if not just_x and not just_y:
                    ax =  fig.add_subplot(grid_size+1, grid_size+1, (grid_size+1)*grid2 + grid1+1, projection = '3d')
                    lymphs[idx].plotRecon_singleDeg(ax)
                    grids_done.append([grid1, grid2])
                    ax.set_title(os.path.basename(lymphs[idx].zoomed_voxels_path))
                elif just_x:
                    if grid2 == (np.mean([v[1] for v in vectors]) - min2) // (range2/grid_size):
                        ax =  fig.add_subplot(1, grid_size+1, grid1+1, projection = '3d')
                        lymphs[idx].plotRecon_singleDeg(ax, max_l, 'thetas', elev = elev, azim = azim)
                        grids_done.append([grid1, grid2])
                        ax.set_title('x_' + os.path.basename(lymphs[idx].zoomed_voxels_path))
                elif just_y:
                    if grid1 == (np.mean([v[0] for v in vectors]) - min1) // (range1/grid_size):

                        ax =  fig.add_subplot(1, grid_size+1, grid2+1, projection = '3d')
                        lymphs[idx].plotRecon_singleDeg(ax, max_l, 'thetas', elev = elev, azim = azim)
                        grids_done.append([grid1, grid2])
                        ax.set_title('y_' + os.path.basename(lymphs[idx].zoomed_voxels_path))

        utils_plotting.equal_axes_3D(*fig.axes)
