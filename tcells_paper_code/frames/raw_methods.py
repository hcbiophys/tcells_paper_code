import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import time
from scipy.spatial.transform import Rotation

import tcells_paper_code.utils.general as utils_general

class Raw_Methods:
    """
    Inherited by Frame class.
    Contains methods without spherical harmonics.
    """

    def surface_plot(self, plotter=None, uropod_align=False, color = (1, 1, 1),  opacity = 1, scalars = None, box = None, with_uropod = True):
        """
        Plot the original cell mesh
        - plotter: pyvista plotter object to plot onto
        - uropod_align: whether or not to shift centroid to origin and rotate to align ellipsoid with an axis.
        - scalars: color each face differently
        - box: box to add around the cell surface
        - with_uropod: whether to plot with a sphere at the location of the uropod label
        """

        if box is not None:
            plotter.add_mesh(box, style='wireframe', opacity = opacity)

        if self.vertices is None:
            return

        if uropod_align:
            uropod, centroid, vertices = self._uropod_align()
            plotter.add_mesh(pv.Sphere(radius=1, center=uropod), color = (1, 0, 0))
            surf = pv.PolyData(vertices, self.faces)
        else:
            if self.uropod is not None and with_uropod:

                plotter.add_mesh(pv.Sphere(radius=1, center=self.uropod), color = (1, 0, 0))
                plotter.add_mesh(pv.Sphere(radius=1, center=self.centroid), color = (0, 0, 0))

            surf = pv.PolyData(self.vertices, self.faces)

        if scalars is None:
            plotter.add_mesh(surf, color = color, opacity = opacity)
        else:
            plotter.add_mesh(surf, color = color, scalars = scalars, opacity = opacity)



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
        """
        Horizontal alignment of the mesh for better visualisation
        """

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
