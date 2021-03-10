import matplotlib.pyplot as plt
import lymphocytes.utils.plotting as utils_plotting
import sys
import numpy as np

class Single_Cell_Methods:
    """
    Inherited by Lymph_Snap class.
    Contains methods for series of a single cell.
    """

    def plot_migratingCell(self, plot_every = 15):

        fig_sing = plt.figure()
        fig_mult = plt.figure()


        ax_sing = fig_sing.add_subplot(111, projection='3d')

        num = len(self.lymph_serieses[0][::plot_every])
        for idx, lymph in enumerate(self.lymph_serieses[0]):
            if lymph is not None:
                if idx%plot_every == 0:

                    ax_sing.plot_trisurf(lymph.vertices[0, :], lymph.vertices[1, :], lymph.vertices[2, :], triangles = np.asarray(lymph.faces[:, ::10]).T)

                    ax = fig_mult.add_subplot(2, num, (idx//plot_every)+1, projection='3d')
                    ax.plot_trisurf(lymph.vertices[0, :], lymph.vertices[1, :], lymph.vertices[2, :], triangles = np.asarray(lymph.faces[:, ::4]).T)
                    ax = fig_mult.add_subplot(2, num, num + (idx//plot_every)+1, projection='3d')

                    lymph.plotRecon_singleDeg(ax)




        for ax in fig_sing.axes + fig_mult.axes[::3]:
            #ax.grid(False)
            #utils_plotting.no_pane_3D(ax)
            ax.set_xlim([0, 0.103*900])
            ax.set_ylim([0, 0.103*512])
            ax.set_zlim([0, 0.211*125])

        #equal_axes_notSquare(*fig.axes)


    def plot_series_voxels(self, plot_every):

        lymph_series = self.lymph_serieses[0]

        voxels = [lymph.zoomed_voxels for lymph in lymph_series]

        voxels = voxels[::plot_every]

        num = len(voxels)
        num_cols = (num // 3) + 1

        fig = plt.figure()

        for idx_file, file in enumerate(voxels):

            ax = fig.add_subplot(3, num_cols, idx_file+1, projection = '3d')
            ax.voxels(voxels)


    def plot_recon_series(self, plot_every):
        fig = plt.figure()

        lymphs_plot = self.lymph_serieses[0][::plot_every]
        num_cols = (len(lymphs_plot) // 3) + 1

        for idx_plot, lymph in enumerate(lymphs_plot):
                ax = fig.add_subplot(3, num_cols, idx_plot+1, projection = '3d')
                lymph.plotRecon_singleDeg(ax)
        utils_plotting.equal_axes_3D(*fig.axes)

    def plot_series_bars(self, plot_every, rotInv):


        lymphs = self.lymph_serieses[0][::plot_every]
        num_cols = (len(lymphs)//3) + 1

        if rotInv:
            fig = plt.figure()
            for idx, lymph in enumerate(lymphs):
                ax = fig.add_subplot(3, num_cols, idx+1)
                ax.bar(range(len(lymph.RI_vector)), lymph.RI_vector)
        else:
            fig_x, fig_y, fig_z = plt.figure(), plt.figure(), plt.figure()
            for coord, fig in zip([0, 1, 2], [fig_x, fig_y, fig_z]):
                for idx, lymph in enumerate(lymphs):
                    ax = fig.add_subplot(3, num_cols, idx+1)
                    if idx == 0:
                        ax.set_title(str(coord))
                    count = 0
                    for l, color in zip([1, 2], ['red', 'blue']):
                        for m in np.arange(0, l+1):
                            clm = lymph._get_clm(coord, l, m)
                            #mag = clm*np.conj(clm)
                            mag = abs(clm)
                            ax.bar(count, mag, color = color)
                            count += 1

            #ax.set_ylim([0, 4])
