import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
import os
import pyvista as pv
import time
import pickle
from scipy.spatial.transform import Rotation
from scipy.ndimage import measurements

import lymphocytes.utils.disk as utils_disk
import lymphocytes.utils.plotting as utils_plotting
import lymphocytes.utils.general as utils_general

class Raw_Methods:
    """
    Inherited by Cell_Frame class.
    Contains methods without spherical harmonics.
    """

    def surface_plot(self, plotter=None, uropod_align=False, color = (1, 1, 1),  opacity = 1, scalars = None, box = None, with_uropod = True):
        """
        Plot the original cell mesh
        - plotter: plotter object to plot onto
        - uropod_align: whether or not to shift centroid to origin and rotate to align ellipsoid with an axis.
        - scalars: color each face differently (?)
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
                pass
                #plotter.add_mesh(pv.Sphere(radius=1, center=self.uropod), color = (1, 0, 0))
                #plotter.add_mesh(pv.Sphere(radius=1, center=self.centroid), color = (0, 0, 0))

            surf = pv.PolyData(self.vertices, self.faces)




        if scalars is None:
            plotter.add_mesh(surf, color = color, opacity = opacity)
        else:
            plotter.add_mesh(surf, color = color, scalars = scalars, opacity = opacity)




        """
        plotter.add_lines(np.array([[-100, 0, 0], [100, 0, 0]]), color = (0, 0, 0))
        plotter.add_lines(np.array([[0, -100, 0], [0, 100, 0]]), color = (0, 0, 0))
        plotter.add_lines(np.array([[0, 0, -100], [0, 0, 100]]), color = (0, 0, 0))
        """


    def voxel_point_cloud(self, plotter):


        voxels = np.array(self.voxels)
        voxels = np.moveaxis(np.moveaxis(voxels, 0, -1), 0, 1)

        def remove_extra_edge(idx_coord, voxels):
            coords = [0, 1, 2]
            other_coords = [i for i in coords if i != idx_coord]
            for row in range(voxels.shape[other_coords[0]]):
                for col in range(voxels.shape[other_coords[1]]):
                    for go in range(voxels.shape[coords[idx_coord]]):
                        if idx_coord == 0:
                            if voxels[go, row, col] == 1:
                                voxels[go, row, col]  = 0
                                break

                        elif idx_coord == 1:
                            if voxels[row, go, col] == 1:
                                voxels[row, go, col] = 0
                                break

                        elif idx_coord == 2:
                            if voxels[row, col, go] == 1:
                                voxels[row, col, go] = 0
                                break

            return voxels
        """
        for idx_coord in [0, 1, 2]:
            print('idx_coord', idx_coord)
            voxels = remove_extra_edge(idx_coord, voxels)
        """


        lw, num = measurements.label(voxels)
        area = measurements.sum(voxels, lw, index=np.arange(lw.max() + 1))


        for idx, j in enumerate(list(np.unique(lw))[1:]):
            voxels_sub = np.zeros_like(lw)
            voxels_sub[lw == j] = 1

            #coordinates = np.argwhere(voxels_sub == 1)*np.array(self.xyz_res) + 0.5*np.array(self.xyz_res)
            point_cloud = pv.PolyData(coordinates)
            color = np.random.rand(3)
            plotter.add_mesh(point_cloud, color = color, opacity = 0.3)


        x, y, z = coordinates.sum(0) / np.sum(voxels)
        self.voxels_centroid = np.array([x, y, z])




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
