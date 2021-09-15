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
    Inherited by Cell_Frame class.
    Contains methods without spherical harmonics.
    """

    def surface_plot(self, plotter=None, uropod_align=False, color = (1, 1, 1),  opacity = 1, scalars = None):
        """
        Plot the original cell mesh
        - plotter: plotter object to plot onto
        - uropod_align: whether or not to shift centroid to origin and rotate to align ellipsoid with an axis.
        - scalars: color each face differently (?)
        """

        if uropod_align:
            uropod, centroid, vertices = self._uropod_align()
            plotter.add_mesh(pv.Sphere(radius=1, center=uropod), color = (1, 0, 0))
            surf = pv.PolyData(vertices, self.faces)
        else:
            #plotter.add_mesh(pv.Sphere(radius=1, center=self.uropod), color = (1, 0, 0))
            surf = pv.PolyData(self.vertices, self.faces)


        if scalars is None:
            plotter.add_mesh(surf, color = color, opacity = opacity)
        else:
            plotter.add_mesh(surf, color = color, scalars = scalars, opacity = opacity)


    def _set_centroid(self):
        """
        Set the centroid attribute
        """
        x, y, z = np.argwhere(self.zoomed_voxels == 1).sum(0) / np.sum(self.zoomed_voxels)
        self.centroid = np.array([x*5*self.xyz_res[0], y*5*self.xyz_res[1], z*5*self.xyz_res[2]])


    def uropod_centroid_line_plot(self, plotter=None, color = (1, 1, 1)):
        """
        Plot the line joining the uropod and centroid
        """
        if plotter is None:
            surf.plot()
        else:
            plotter.add_lines(np.array([self.uropod, self.centroid]), color = color)



    def _uropod_align(self, axis = np.array([0, 0, -1])):
        """
        Shift centroid to origin and rotate to align ellipsoid with an axis.
        """


        vertices = self.vertices - self.uropod
        centroid = self.centroid - self.uropod
        print(np.mean(vertices))
        uropod = np.array([0, 0, 0])

        rotation_matrix = utils_general.rotation_matrix_from_vectors(centroid, axis)
        R = Rotation.from_matrix(rotation_matrix)
        vertices = R.apply(vertices)
        centroid = R.apply(centroid)

        return uropod, centroid, vertices


    def _uropod_and_horizontal_align(self):

        uropod, centroid, vertices = self._uropod_align()
        ranges = []
        for idx in range(2):
            coord_range = np.max(vertices[:, idx]) - np.min(vertices[:, idx])
            ranges.append(coord_range)

        horiz_vector = [0, 0, 0]
        horiz_vector[ranges.index(max(ranges))] = 1
        rotation_matrix = utils_general.rotation_matrix_from_vectors(horiz_vector, (-1, -1, 0))

        R = Rotation.from_matrix(rotation_matrix)
        vertices = R.apply(vertices)
        centroid = R.apply(centroid)

        return uropod, centroid, vertices





    def show_voxels(self):
        """
        Plot zoomed voxels (zoomed for delta_centroid)
        """
        fig_showVoxels = plt.figure()
        ax = fig_showVoxels.add_subplot(111, projection = '3d')
        ax.voxels(self.zoomed_voxels,  edgecolors = 'white')
        utils_plotting.label_axes_3D(ax)

        print(self.zoomed_voxels.shape)
        sys.exit()
