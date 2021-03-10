import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
import os
import lymphocytes.utils.disk as utils_disk
import lymphocytes.utils.plotting as utils_plotting

class Raw_Methods:
    """
    Inherited by Lymph_Snap class.
    Contains methods without spherical harmonics.
    """

    def surface_plot(self, subsample_rate):
        """
        Plot the (subsampled) triangulation surface.
        """
        fig_surfacePlot = plt.figure()
        ax = fig_surfacePlot.add_subplot(111, projection='3d')

        ax.plot_trisurf(self.vertices[0, :], self.vertices[1, :], self.vertices[2, :], triangles = np.asarray(self.faces[:, ::subsample_rate]).T)
        ax.grid(False)

        utils_plotting.label_axes_3D(ax)
        utils_plotting.no_pane(ax)
        utils_plotting.equal_axes_3D(ax)


    def show_voxels(self):
        """
        Plot voxels, either orig(inal) or zoomed (in which case read from
        saved subsampled for speed reasons).
        """
        fig_showVoxels = plt.figure()
        ax = fig_showVoxels.add_subplot(111, projection = '3d')
        ax.voxels(self.zoomed_voxels,  edgecolors = 'white')
        label_axes_3D(ax)
