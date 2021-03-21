import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
import os
import pyvista as pv
import pickle

import lymphocytes.utils.disk as utils_disk
import lymphocytes.utils.plotting as utils_plotting
import lymphocytes.utils.general as utils_general

class Raw_Methods:
    """
    Inherited by Lymph_Snap class.
    Contains methods without spherical harmonics.
    """

    def surface_plot(self, plotter=None, uropod_allign=False, color = (1, 1, 1), opacity = 1):

        if uropod_allign:
            point_cloud = self.uropod_allign_vertices_origin()
            plotter.add_mesh(point_cloud)


        surf = pv.PolyData(self.vertices, self.faces)
        if plotter is None:
            surf.plot()
        else:
            plotter.add_mesh(surf, color = color, opacity = opacity)

    def _set_orig_centroid(self):
        if self.orig_centroid is None:
            x, y, z = np.argwhere(self.zoomed_voxels == 1).sum(0) / np.sum(self.zoomed_voxels)
            self.orig_centroid = np.array([x*5*0.103, y*5*0.103, z*5*0.211])


    def show_voxels(self):
        """
        Plot voxels, either orig(inal) or zoomed (in which case read from
        saved subsampled for speed reasons).
        """
        fig_showVoxels = plt.figure()
        ax = fig_showVoxels.add_subplot(111, projection = '3d')
        ax.voxels(self.zoomed_voxels,  edgecolors = 'white')
        utils_plotting.label_axes_3D(ax)
