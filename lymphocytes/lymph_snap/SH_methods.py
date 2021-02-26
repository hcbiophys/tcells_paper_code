import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.special import sph_harm
import matplotlib.tri as mtri

import lymphocytes.utils.voxels as utils_voxels
import lymphocytes.utils.plotting as utils_plotting


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
                    #list = line.split(',')
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
        self.coeff_array[:, 0] *= 0.103
        self.coeff_array[:, 1] *= 0.103
        self.coeff_array[:, 2] *= 0.211


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


    def get_vector(self, max_l):
        """
        Returns vector representation of (truncated) self.coeff_array.
        """

        idx_trunc = (max_l*max_l + 2*(max_l+1)) + 1
        vector = np.concatenate([self.coeff_array[:idx_trunc, 0], self.coeff_array[:idx_trunc, 1], self.coeff_array[:idx_trunc, 2]], axis = 0)

        return vector


    def get_rotInv_vector(self, max_l):
        """
        Returns vector representation of (truncated) rotationally invariant self.coeff_array.
        """

        # fo scale invariance
        # use pre-saved zoomed voxels for speed reasons
        volume = np.sum(self.zoomed_voxels)
        x_range, y_range, z_range = utils_voxels.find_voxel_ranges(self.zoomed_voxels)

        vector = []

        for l in np.arange(0, max_l + 1):
            l_energy = 0
            for coord, range in zip([0, 1, 2], [x_range, y_range, z_range]):
                for m in np.arange(0, l+1):
                    clm = self._get_clm(coord, l, m)
                    clm /= range/100
                    #clm /= np.cbrt(volume/5000)

                    l_energy += clm*np.conj(clm)

            vector.append(l_energy)

        vector = [i.real for i in vector] # imaginary parts are 0

        return vector


    def reconstruct_xyz_from_spharm_coeffs(self, max_l):
        """
        Reconstruct {x, y, z} shape from (truncated) self.coeff_array attribute.
        """

        volume = np.sum(self.zoomed_voxels)

        thetas = np.linspace(0, np.pi, 50)
        phis = np.linspace(0, 2*np.pi, 50)
        thetas, phis = np.meshgrid(thetas, phis)

        thetas = thetas.flatten()
        phis = phis.flatten()

        xs = []
        ys = []
        zs = []

        for coord_idx, list in zip([0, 1, 2], [xs, ys, zs]):
            for t, p in zip(thetas, phis):
                func_value = 0
                for l in np.arange(0, max_l + 1):
                    for m in np.arange(0, l+1):
                        clm = self._get_clm(coord_idx, l, m)
                        clm /= np.cbrt(volume/5000)
                        func_value += clm*sph_harm(m, l, p, t)

                list.append(func_value)

        return xs, ys, zs, phis, thetas



    def plotRecon_manyDegs(self, max_l_list, color_var):
        """
        Plot subplots with different truncation degrees.
        Args:
        - max_l_list: list of maximum l values (i.e. different truncations).
        - color_var: parameter to color by (theta or phi).
        """

        fig_plotRecon_manyDegs = plt.figure()

        for idx in range(len(max_l_list)):
            ax = fig.add_subplot(1, len(max_l_list)+1, idx+1, projection = '3d')
            xs, ys, zs, phis, thetas = self.reconstruct_xyz_from_spharm_coeffs(max_l_list[idx])
            if color_var == 'phis':
                #ax.scatter([i.real for i in xs], [i.real for i in ys], [i.real for i in zs], s = 100, alpha = 1, c = phis)
                #points = np.concatenate([np.expand_dims([i.real for i in xs], axis = 1), np.expand_dims([i.real for i in ys], axis = 1), np.expand_dims([i.real for i in zs], axis = 1) ], axis = 1)
                #cloud = pv.PolyData(points)
                #cloud.plot(point_size=15)
                tris = mtri.Triangulation(phis, thetas)
                ax.plot_trisurf([i.real for i in xs], [i.real for i in ys], [i.real for i in zs], triangles = tris.triangles)
                #surf = cloud.delaunay_2d()
                #surf.plot(show_edges=True)
            elif color_var == 'thetas':
                ax.scatter([i.real for i in xs], [i.real for i in ys], [i.real for i in zs], s = 100, alpha = 1, c = thetas)
            else:
                raise ValueError('Phi or Theta incorrect arg')


    def plotRecon_singleDeg(self, ax, max_l, color_param = 'thetas', elev = None, azim = None):
        """
        Plot reconstruction at a single truncation degree.
        Args:
        - ax: axis to plot onto.
        - max_l: max l value (i.e. truncation).
        - color_param: parameter to color by ("thetas" / "phis").
        - elev: elevation of view.
        - azim: azimuthal angle of view.
        """

        xs, ys, zs, phis, thetas = self.reconstruct_xyz_from_spharm_coeffs(max_l)

        xs = xs[::4]
        ys = ys[::4]
        zs = zs[::4]
        phis = phis[::4]
        thetas = thetas[::4]


        tris = mtri.Triangulation(phis, thetas)

        collec = ax.plot_trisurf([i.real for i in ys], [i.real for i in zs], [i.real for i in xs], triangles = tris.triangles, edgecolor='none', linewidth = 0, antialiased = False)
        #collec = ax.plot_trisurf([i.real for i in ys], [i.real for i in zs], [i.real for i in xs], triangles = tris.triangles, color=(0,0,0,0), edgecolor='none')
        ax.view_init(elev, azim)

        if color_param == 'phis':
            colors = np.mean(phis[tris.triangles], axis = 1)
            collec.set_array(colors)
        elif color_param == 'thetas':
            colors = np.mean(thetas[tris.triangles], axis = 1)
            collec.set_array(colors)


        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        utils_plotting.equal_axes_3D(ax)
        #remove_ticks(ax)
        #ax.set_axis_off()
        #plt.colorbar(collec)
