import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.special import sph_harm
import matplotlib.tri as mtri
import pyvista as pv
import pickle
from scipy.spatial.transform import Rotation
import math

import lymphocytes.utils.voxels as utils_voxels
import lymphocytes.utils.plotting as utils_plotting
import lymphocytes.utils.general as utils_general

class SH_Methods:
    """
    Inherited by Cell_Frame class.
    Contains methods with spherical harmonics.
    """

    def _set_spharm_coeffs(self, coeffs_txt_file):
        """
        Set the SPHARM coeffs as self.coeff_array attribute
        Args:
        - coeffs_txt_file: full path to load (including the frame)
        NOTE: Not initially in correct order, so reordering applied so columns are (x, y, z)
        """

        with open(coeffs_txt_file, "r") as file:
            num_lines = 0
            for line in file:
                num_lines += 1

        coeff_array = np.zeros((num_lines, 3))

        with open(coeffs_txt_file, "r") as file:

            on_line = 0

            for line in file:

                if on_line == 0:
                    list = line.split(',')
                    first = float(list[1][1:])
                    second = float(list[2])
                    third = float(list[3][:-1])
                elif on_line == num_lines-1:
                    list = line.split(',')
                    first = float(list[0][1:])
                    second = float(list[1])
                    third = float(list[2][:-2])
                else:
                    list = line.split(',')
                    list = list[:-1]
                    first = float(list[0][1:])
                    second = float(list[1])
                    third = float(list[2][:-1])

                coeff_array[on_line, 0] = first
                coeff_array[on_line, 1] = second
                coeff_array[on_line, 2] = third

                on_line += 1

        self.coeff_array = coeff_array

        if not self.idx_cell[:2] == 'zm':
            self.coeff_array[:, [0, 1, 2]] = self.coeff_array[:, [2, 1, 0]]




        # scale by voxel resolution
        self.coeff_array[:, 0] *= self.xyz_res[0]
        self.coeff_array[:, 1] *= self.xyz_res[1]
        self.coeff_array[:, 2] *= self.xyz_res[2]


    def _get_clm(self, dimension, l, m):
        """
        Get clm (spherical harmonic complex coefficient) from the self.coeff_array atttribute
        Args:
        - dimension: {x, y, or z}
        - l: 'energy' index
        - m: rotational index
        """
        if m == 0:
            a = self.coeff_array[l*l, dimension]
            b = 0
        else:
            a = self.coeff_array[l*l + 2*m - 1, dimension]
            b = self.coeff_array[l*l + 2*m, dimension]
        return complex(a, b)


    def _set_vector(self, plot = False, ax = None):
        """
        Sets the vector representation of (truncated) self.coeff_array
        """
        idx_trunc = self.max_l*self.max_l + 2*(self.max_l) + 1
        self.vector = np.concatenate([self.coeff_array[:idx_trunc, 0], self.coeff_array[:idx_trunc, 1], self.coeff_array[:idx_trunc, 2]], axis = 0)

    def _set_RIvector(self):
        """
        Sets the rotationally invariant descriptor
        """

        self.RI_vector = [(3/2)*np.linalg.norm(self.centroid - self.uropod)/(np.cbrt(self.volume))] # here is the first descriptor, distance between centroid and uropod
        for l in np.arange(1, self.max_l + 1):
            l_energy = 0
            for coord in [0, 1, 2]:
                for m in np.arange(0, l+1):
                    clm = self._get_clm(coord, l, m)
                    clm /= np.cbrt(self.volume) # from LBS particles and LBS fragments
                    l_energy += clm*np.conj(clm)

            if math.isnan(np.sqrt(l_energy.real)):
                print(self.frame, clm)

            self.RI_vector.append(np.sqrt(l_energy.real)) # imaginary component is zero


        self.RI_vector = np.array(self.RI_vector) # scale invariance



    def reconstruct_xyz_from_spharm_coeffs(self, l_start = 1, max_l = None):
        """
        Reconstruct {x, y, z} shape from (truncated) self.coeff_array attribute
        - l_start: determines whether it's reconstructed at the origin or not
        """
        if max_l is None:
            max_l = self.max_l
        thetas = np.linspace(0, np.pi, 50)
        phis = np.linspace(0, 2*np.pi, 50)
        thetas, phis = np.meshgrid(thetas, phis)
        thetas, phis = thetas.flatten(), phis.flatten()

        xs, ys, zs = [], [], []

        for coord_idx, list in zip([0, 1, 2], [xs, ys, zs]):
            for t, p in zip(thetas, phis):
                func_value = 0
                for l in np.arange(l_start, max_l + 1):
                    for m in np.arange(0, l+1):
                        clm = self._get_clm(coord_idx, l, m)
                        func_value += clm*sph_harm(m, l, p, t)

                list.append(func_value.real)


        if self.idx_cell[:2] == 'zm':

            xs = [x + np.min(self.vertices[:, 0]) for x in xs]
            ys = [y + np.min(self.vertices[:, 1]) for y in ys]
            zs = [z + np.min(self.vertices[:, 2]) for z in zs]

        return xs, ys, zs, phis, thetas



    def _get_vertices_faces_plotRecon_singleDeg(self, max_l = None, uropod_align = False, horizontal_align = False):

        xs, ys, zs, phis, thetas = self.reconstruct_xyz_from_spharm_coeffs(l_start = 0, max_l = max_l)
        [xs, ys, zs, phis, thetas] = [np.array(i) for i in [xs, ys, zs, phis, thetas]]

        vertices = np.array(np.concatenate([xs[..., np.newaxis] , ys[..., np.newaxis] , zs[..., np.newaxis] ], axis = 1))
        uropod = self.uropod

        if uropod_align:
            # don't change class attributes here
            print(np.mean(vertices), np.mean(self.uropod))
            vertices -= self.uropod
            centroid = self.centroid - self.uropod
            uropod = np.array([0, 0, 0])

            rotation_matrix = utils_general.rotation_matrix_from_vectors(centroid-uropod, np.array([0, 0, -1]))
            R = Rotation.from_matrix(rotation_matrix)
            vertices = R.apply(vertices)


        if horizontal_align:
            ranges = []
            for idx in range(2):
                coord_range = np.max(vertices[:, idx]) - np.min(vertices[:, idx])
                ranges.append(coord_range)

            horiz_vector = [0, 0, 0]
            horiz_vector[ranges.index(max(ranges))] = 1

            rotation_matrix = utils_general.rotation_matrix_from_vectors(vec1 = horiz_vector, vec2 = (-1, -1, 0))
            R = Rotation.from_matrix(rotation_matrix)
            vertices = R.apply(vertices)

        faces = utils_general.faces_from_phisThetas(phis, thetas)

        return vertices, faces, uropod




    def plotRecon_singleDeg(self, plotter, max_l = None, uropod_align = False, color = (1, 1, 1), opacity = 1):
        """
        Plot reconstruction at a single truncation degree
        """

        vertices, faces, uropod = self._get_vertices_faces_plotRecon_singleDeg(max_l = max_l, uropod_align = uropod_align)

        plotter.add_mesh(pv.Sphere(radius=1, center=uropod), color = (1, 0, 0))

        surf = pv.PolyData(vertices, faces)
        plotter.add_mesh(surf, color = color, opacity = opacity)


        #plotter.add_lines(np.array([[-10, 0, 0], [10, 0, 0]]), color = (0, 0, 0))
        #plotter.add_lines(np.array([[0, -10, 0], [0, 10, 0]]), color = (0, 0, 0))
        #plotter.add_lines(np.array([[0, 0, -10], [0, 0, 10]]), color = (0, 0, 0))
