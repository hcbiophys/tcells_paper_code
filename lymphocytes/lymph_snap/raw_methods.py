import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
import os
import pyvista as pv
import pickle
from scipy.spatial.transform import Rotation

import lymphocytes.utils.disk as utils_disk
import lymphocytes.utils.plotting as utils_plotting
import lymphocytes.utils.general as utils_general

class Raw_Methods:
    """
    Inherited by Lymph_Snap class.
    Contains methods without spherical harmonics.
    """

    def surface_plot(self, plotter=None, uropod_align=False, color = (1, 1, 1), opacity = 1):

        if uropod_align:
            uropod, centroid, vertices = self._uropod_align()
            plotter.add_mesh(uropod, color = (1, 0, 0))
        else:
            plotter.add_mesh(self.uropod, color = (1, 0, 0))


        surf = pv.PolyData(vertices, self.faces)
        if plotter is None:
            surf.plot()
        else:
            plotter.add_mesh(surf, color = color, opacity = opacity)


    def _uropod_align(self, axis = np.array([0, 0, 1])):
        """
        Shift centroid to origin and rotate to align ellipsoid with an axis.
        """

        if not self.uropod_aligned:
            vertices -= self.uropod
            centroid -= self.uropod
            uropod = np.array([0, 0, 0])

            rotation_matrix = utils_general.rotation_matrix_from_vectors(centroid, axis)
            R = Rotation.from_matrix(rotation_matrix)
            vertices = R.apply(vertices)
            centroid = R.apply(centroid)

        return uropod, centroid, vertices




    def _set_centroid(self):
        x, y, z = np.argwhere(self.zoomed_voxels == 1).sum(0) / np.sum(self.zoomed_voxels)
        self.centroid = np.array([x*5*0.103, y*5*0.103, z*5*0.211])


    def show_voxels(self):
        """
        Plot voxels, either orig(inal) or zoomed (in which case read from
        saved subsampled for speed reasons).
        """
        fig_showVoxels = plt.figure()
        ax = fig_showVoxels.add_subplot(111, projection = '3d')
        ax.voxels(self.zoomed_voxels,  edgecolors = 'white')
        utils_plotting.label_axes_3D(ax)
