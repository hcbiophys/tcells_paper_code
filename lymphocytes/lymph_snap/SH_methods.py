import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.special import sph_harm

from lymphocytes.utils.voxels import Utils_Voxels
from lymphocytes.utils.plotting import Utils_Plotting


class SH_Methods:

    def set_spharm_coeffs(self, coeffs_txt_file):

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


    def get_clm(self, coeff_array, dimension, l, m):

        if m == 0:
            a = coeff_array[l*l, dimension]
            b = 0
        else:
            a = coeff_array[l*l + 2*m - 1, dimension]
            b = coeff_array[l*l + 2*m, dimension]

        return complex(a, b)


    def set_vector(self, max_l):


        idx_trunc = (max_l*max_l + 2*(max_l+1)) + 1


        vector = np.concatenate([self.coeff_array[:idx_trunc, 0], self.coeff_array[:idx_trunc, 1], self.coeff_array[:idx_trunc, 2]], axis = 0)

        return vector


    def set_rotInv_vector(self, max_l):

        volume = Utils_Voxels.voxel_volume(self.niigz)

        x_range, y_range, z_range = Utils_Voxels.find_voxel_ranges(self.niigz)

        vector = []

        for l in np.arange(0, max_l + 1):
            l_energy = 0
            for coord, range in zip([0, 1, 2], [x_range, y_range, z_range]):
                for m in np.arange(0, l+1):
                    clm = self.SH_get_clm(self.coeff_array, coord, l, m)
                    clm /= range/100
                    #clm /= np.cbrt(volume/5000)

                    l_energy += clm*np.conj(clm)

            vector.append(l_energy)

        vector = [i.real for i in vector] # imaginary parts are 0


        return vector




    def reconstruct_xyz_from_spharm_coeffs(self, coeff_array, max_l):
        """
        max max_l: 85
        """

        volume = Utils_Voxels.voxel_volume(self.niigz)

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
                        clm = self.SH_get_clm(coeff_array, coord_idx, l, m)
                        clm /= np.cbrt(volume/5000)
                        func_value += clm*sph_harm(m, l, p, t)

                list.append(func_value)

        return xs, ys, zs, phis, thetas



    def plotRecon_manyDegs(self, max_l_list, color_var):


        """
        pts = mlab.points3d([i.real for i in xs], [i.real for i in ys], [i.real for i in zs], phis)
        mesh = mlab.pipeline.delaunay2d(pts)
        pts.remove()
        surf = mlab.pipeline.surface(mesh)

        mlab.show()
        sys.exit()
        """

        fig_plotRecon_manyDegs = plt.figure()

        for idx in range(len(max_l_list)):
            ax = fig.add_subplot(1, len(max_l_list)+1, idx+1, projection = '3d')
            xs, ys, zs, phis, thetas = self.SH_reconstruct_xyz_from_spharm_coeffs(max_l_list[idx])
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


    def plotRecon_singleDeg(self, ax, max_l, color_param = 'thetas', elev = None, azim = None, normaliseScale = False):

        xs, ys, zs, phis, thetas = self.SH_reconstruct_xyz_from_spharm_coeffs(self.coeff_array, max_l)

        xs = xs
        ys = ys
        zs = zs
        phis = phis
        thetas = thetas


        tris = mtri.Triangulation(phis, thetas)

        #collec = ax.plot_trisurf([i.real for i in ys], [i.real for i in zs], [i.real for i in xs], triangles = tris.triangles, cmap=plt.cm.PiYG, edgecolor='none', linewidth = 0, antialiased = False)
        collec = ax.plot_trisurf([i.real for i in ys], [i.real for i in zs], [i.real for i in xs], triangles = tris.triangles, color=(0,0,0,0), edgecolor='Gray')
        ax.view_init(elev, azim)

        """
        if color_param == 'xs':
            real_xs = np.array([i.real for i in xs])
            colors = np.mean(real_xs[tris.triangles], axis = 1)
            collec.set_array(colors)
        elif color_param == 'ys':
            real_ys = np.array([i.real for i in ys])
            colors = np.mean(real_ys[tris.triangles], axis = 1)
            collec.set_array(colors)
        elif color_param == 'zs':
            real_zs = np.array([i.real for i in zs])
            colors = np.mean(real_zs[tris.triangles], axis = 1)
            collec.set_array(colors)
        elif color_param == 'phis':
            colors = np.mean(phis[tris.triangles], axis = 1)
            collec.set_array(colors)
        elif color_param == 'thetas':
            colors = np.mean(thetas[tris.triangles], axis = 1)
            collec.set_array(colors)
        """



        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        Utils_Plotting.equal_axes(ax)
        #remove_ticks(ax)
        #ax.set_axis_off()

        #plt.colorbar(collec)
