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

    def plot_rotInvRep_series_bars(self, maxl = 5, plot_every = 1, means_adjusted = False):


        fig = plt.figure()

        num_cols = (len(self.lymph_serieses[0])//3) + 1

        for lymph in self.lymph_serieses[0]:

            lymph.set_rotInv_vector(maxl)

            ax = fig.add_subplot(3, num_cols, idx+1)
            ax.bar(range(len(lymph.RI_vector)), lymph.RI_vector)
            ax.set_ylim([0, 4])
