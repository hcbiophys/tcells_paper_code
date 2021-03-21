import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.special import sph_harm
import matplotlib.tri as mtri
import pyvista as pv
from scipy.spatial.transform import Rotation
import pickle

import lymphocytes.utils.voxels as utils_voxels
import lymphocytes.utils.plotting as utils_plotting
import lymphocytes.utils.general as utils_general

class SH_Methods:
    """
    Inherited by Lymph_Snap class.
    Contains methods with spherical harmonics.
    """

    def _set_spharm_coeffs(self, coeffs_txt_file):
        """
        Set the SPHARM coeffs as self.coeff_array attribute.
        Args:
        - coeffs_txt_file: full path to load (including the frame).
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

        self.coeff_array[:, [0, 1, 2]] = self.coeff_array[:, [2, 1, 0]]

        # scale by voxel resolution
        self.coeff_array[:, 0] *= 5*0.103 # 5 since zoomed by 5x, 0.103 for voxel resolution
        self.coeff_array[:, 1] *= 5*0.103
        self.coeff_array[:, 2] *= 5*0.211


    def _get_clm(self, dimension, l, m):
        """
        Get clm (spherical harmonic complex coefficient) from the self.coeff_array atttribute.
        Args:
        - dimension: {x, y, or z}.
        - l: 'energy' index.
        - m: rotational index.
        """
        if m == 0:
            a = self.coeff_array[l*l, dimension]
            b = 0
        else:
            a = self.coeff_array[l*l + 2*m - 1, dimension]
            b = self.coeff_array[l*l + 2*m, dimension]

        return complex(a, b)


    def _set_vector(self):
        """
        Returns vector representation of (truncated) self.coeff_array.
        """

        idx_trunc = (self.max_l*self.max_l + 2*(self.max_l+1)) + 1
        self.vector = np.concatenate([self.coeff_array[:idx_trunc, 0], self.coeff_array[:idx_trunc, 1], self.coeff_array[:idx_trunc, 2]], axis = 0)

    def _set_RIvector(self):
        file = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/uropods/cell_{}.pickle'.format(self.idx_cell),"rb")
        uropods = pickle.load(file)
        uropod_coords = np.array(uropods[self.frame])


        self._set_orig_centroid()
        self.RI_vector = [np.linalg.norm(self.orig_centroid - uropod_coords)]
        for l in np.arange(1, self.max_l + 1):
            l_energy = 0
            for coord in [0, 1, 2]:
                for m in np.arange(0, l+1):
                    clm = self._get_clm(coord, l, m)
                    l_energy += clm*np.conj(clm)
            self.RI_vector.append(np.sqrt(l_energy.real)) # imaginary component is zero

        self.RI_vector = np.array([i/np.cbrt(self.volume) for i in self.RI_vector]) # scale invariance




    def reconstruct_xyz_from_spharm_coeffs(self, l_start = 0, max_l = None):
        """
        Reconstruct {x, y, z} shape from (truncated) self.coeff_array attribute.
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

        return xs, ys, zs, phis, thetas






    def ellipsoid_allign_vertices_origin(self, axis = np.array([0, 0, 1])):
        """
        Shift centroid to origin and rotate to align uropod with an axis.
        """
        self._set_orig_centroid()
        self.vertices -= self.orig_centroid

        # get rotation matrix
        xs, ys, zs, phis, thetas = self.reconstruct_xyz_from_spharm_coeffs(l_start = 1, max_l = 1)
        dists = [np.sqrt(xs[idx]**2 + ys[idx]**2 + zs[idx]**2) for idx in range(len(xs))]
        idx_max = dists.index(max(dists))
        peak_vec = np.array([xs[idx_max], ys[idx_max], zs[idx_max]])
        rotation_matrix = utils_general.rotation_matrix_from_vectors(peak_vec, axis)
        R = Rotation.from_matrix(rotation_matrix)

        self.vertices = R.apply(self.vertices)


    def uropod_allign_vertices_origin(self, axis = np.array([0, 0, 1])):
        """
        Shift centroid to origin and rotate to align ellipsoid with an axis.
        """
        self._set_orig_centroid()
        self.vertices -= self.orig_centroid

        # get rotation matrix
        file = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/uropods/cell_{}.pickle'.format(self.idx_cell),"rb")
        uropods = pickle.load(file)

        uropod_vec = uropods[self.frame] - self.orig_centroid
        rotation_matrix = utils_general.rotation_matrix_from_vectors(uropod_vec, axis)
        R = Rotation.from_matrix(rotation_matrix)
        self.vertices = R.apply(self.vertices)

        point_cloud = uropods[self.frame] - self.orig_centroid
        point_cloud = pv.PolyData(R.apply(point_cloud))

        return point_cloud


    def plotRecon_singleDeg(self, plotter, l_start = 0, max_l = None, ellipsoid_allign = False):
        """
        Plot reconstruction at a single truncation degree.
        Args:
        - ax: axis to plot onto.
        - max_l: max l value (i.e. truncation).
        - color_param: parameter to color by ("thetas" / "phis").
        - elev: elevation of view.
        - azim: azimuthal angle of view.
        """

        xs, ys, zs, phis, thetas = self.reconstruct_xyz_from_spharm_coeffs(l_start = l_start, max_l = max_l)
        #[xs, ys, zs, phis, thetas] = utils_general.subsample_lists(3, xs, ys, zs, phis, thetas)
        [xs, ys, zs, phis, thetas] = [np.array(i) for i in [xs, ys, zs, phis, thetas]]

        vertices = np.array(np.concatenate([xs[..., np.newaxis] , ys[..., np.newaxis] , zs[..., np.newaxis] ], axis = 1))
        if ellipsoid_allign:
            vertices = self.R.apply(vertices)

        faces = utils_general.faces_from_phisThetas(phis, thetas)
        surf = pv.PolyData(vertices, faces)
        plotter.add_mesh(surf)
        plotter.add_axes()
