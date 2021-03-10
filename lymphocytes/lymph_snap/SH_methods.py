import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.special import sph_harm
import matplotlib.tri as mtri

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


    def _set_vector(self, max_l):
        """
        Returns vector representation of (truncated) self.coeff_array.
        """

        idx_trunc = (max_l*max_l + 2*(max_l+1)) + 1
        self.vector = np.concatenate([self.coeff_array[:idx_trunc, 0], self.coeff_array[:idx_trunc, 1], self.coeff_array[:idx_trunc, 2]], axis = 0)



    def _set_rotInv_vector(self, max_l):
        """
        Returns vector representation of (truncated) rotationally invariant self.coeff_array.
        """

        vector = []
        for l in np.arange(0, max_l + 1):
            l_energy = 0
            for coord in [0, 1, 2]:
                for m in np.arange(0, l+1):
                    clm = self._get_clm(coord, l, m)
                    l_energy += clm*np.conj(clm)
            vector.append(l_energy)

        self.RI_vector = [i.real for i in vector] # imaginary parts are 0
        self.RI_vector = self.RI_vector[1:] # translation invariance
        self.RI_vector = [i/np.cbrt(self.volume) for i in self.RI_vector] # scale invariance


    def reconstruct_xyz_from_spharm_coeffs(self):
        """
        Reconstruct {x, y, z} shape from (truncated) self.coeff_array attribute.
        """

        volume = np.sum(self.zoomed_voxels)

        thetas = np.linspace(0, np.pi, 50)
        phis = np.linspace(0, 2*np.pi, 50)
        thetas, phis = np.meshgrid(thetas, phis)
        thetas, phis = thetas.flatten(), phis.flatten()

        xs, ys, zs = [], [], []

        for coord_idx, list in zip([0, 1, 2], [xs, ys, zs]):
            for t, p in zip(thetas, phis):
                func_value = 0
                for l in np.arange(0, self.max_l + 1):
                    for m in np.arange(0, l+1):
                        clm = self._get_clm(coord_idx, l, m)
                        func_value += clm*sph_harm(m, l, p, t)
                list.append(func_value)

        return xs, ys, zs, phis, thetas


    def plotRecon_singleDeg(self, ax, color_param = 'thetas'):
        """
        Plot reconstruction at a single truncation degree.
        Args:
        - ax: axis to plot onto.
        - max_l: max l value (i.e. truncation).
        - color_param: parameter to color by ("thetas" / "phis").
        - elev: elevation of view.
        - azim: azimuthal angle of view.
        """

        xs, ys, zs, phis, thetas = self.reconstruct_xyz_from_spharm_coeffs()
        [xs, ys, zs, phis, thetas] = utils_general.subsample_lists(4, xs, ys, zs, phis, thetas)
        tris = mtri.Triangulation(phis, thetas)
        collec = ax.plot_trisurf([i.real for i in xs], [i.real for i in ys], [i.real for i in zs], triangles = tris.triangles, edgecolor='none', linewidth = 0, antialiased = False)
        elev, azim = utils_voxels.find_optimal_3dview(self.zoomed_voxels)
        ax.view_init(elev, azim)

        if color_param == 'phis':
            colors = np.mean(phis[tris.triangles], axis = 1)
        elif color_param == 'thetas':
            colors = np.mean(thetas[tris.triangles], axis = 1)
        collec.set_array(colors)

        ax.grid(False)

        utils_plotting.label_axes_3D(ax)
        utils_plotting.equal_axes_3D(ax)
        utils_plotting.no_pane_3D(ax)
