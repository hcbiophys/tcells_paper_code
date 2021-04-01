import matplotlib.pyplot as plt
import lymphocytes.utils.plotting as utils_plotting
import sys
import numpy as np
import pyvista as pv
import pickle

class Single_Cell_Methods:
    """
    Inherited by Lymph_Snap class.
    Contains methods for series of a single cell.
    """

    def _uropod_callback(self, a, b):
        """
        Callback for when selecting uropods
        """
        point = np.array(a.points[b, :])
        print(point)
        self.uropod_coords[-1] = point

    def select_uropods(self, idx_cell, plot_every):
        """
        Select the uropods
        """
        self.frames = []
        self.uropod_coords = []

        lymphs = self.lymph_serieses[idx_cell][::plot_every]


        for idx_plot, lymph in enumerate(lymphs):
            self.frames.append(lymph.frame)
            self.uropod_coords.append(None)
            plotter = pv.Plotter()
            lymph.surface_plot(plotter=plotter, uropod_align=False)
            plotter.enable_point_picking(callback = self._uropod_callback, show_message=True,
                       color='pink', point_size=10,
                       use_mesh=True, show_point=True)
            #plotter.enable_cell_picking(through=False, callback = self._uropod_callback)
            plotter.show(cpos=[0, 1, 0])


        uropod_dict = {frame:coords for frame, coords in zip(self.frames, self.uropod_coords)}

        pickle_out = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/uropods/cell_{}.pickle'.format(idx_cell),'wb')
        pickle.dump(uropod_dict, pickle_out)
        print(uropod_dict)



    def plot_orig_series(self, idx_cell, uropod_align, plot_every):
        """
        Plot original mesh series, with point at the uropods
        """

        lymphs_plot = self.lymph_serieses[idx_cell][::plot_every]
        num_cols = (len(lymphs_plot) // 3) + 1
        plotter = pv.Plotter(shape=(3, num_cols))
        #lymphs_plot[0]._set_ellipsoid_rotation_matrix()
        #R = lymphs_plot[0].R
        for idx_plot, lymph in enumerate(lymphs_plot):
            plotter.subplot(idx_plot//num_cols, idx_plot%num_cols)
            lymphs_plot[idx_plot-1].surface_plot(plotter=plotter, uropod_align=uropod_align, color = (0.3, 0.3, 1), opacity = 0.5)
            lymph.surface_plot(plotter=plotter, uropod_align=uropod_align)


        plotter.show(cpos=[0, 1, 0])


    def plot_uropod_trajectory(self, idx_cell):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        uropods = [lymph.uropod for lymph in self.lymph_serieses[idx_cell]]
        ax.plot([i[0] for i in uropods], [i[1] for i in uropods], [i[2] for i in uropods])


    def plot_migratingCell(self, idx_cell, plot_every = 15):
        """
        Plot all meshes of a cell in one window
        """

        lymphs = self.lymph_serieses[idx_cell][::plot_every]
        plotter = pv.Plotter()
        for lymph in lymphs:
            color = np.random.rand(3,)
            surf = pv.PolyData(lymph.vertices, lymph.faces)
            plotter.add_mesh(surf, color = color)
        box = pv.Box(bounds=(0, 92.7, 0, 52.7, 0, 26.4))
        plotter.add_mesh(box, style='wireframe')
        plotter.add_axes()
        plotter.show(cpos=[0, 1, 0.5])


    def plot_series_PCs(self, idx_cell, plot_every):
        """
        Plot the PCs of each frame of a cell
        """
        fig = plt.figure()
        self._set_pca(n_components = 3)
        lymphs = self.lymph_serieses[idx_cell][::plot_every]
        pcas = np.array([i.pca for i in lymphs])
        min_ = np.min(np.min(pcas, axis = 0))
        max_ = np.max(np.max(pcas, axis = 0))
        num_cols = (len(lymphs)//3) + 1
        for idx, lymph in enumerate(lymphs):
            ax = fig.add_subplot(3, num_cols, idx+1)
            ax.bar(range(len(lymph.pca)), lymph.pca)
            ax.set_ylim([min_, max_])
        plt.show()


    def plot_recon_series(self, idx_cell, plot_every):
        """
        Plot reconstructed mesh series
        """
        lymphs_plot = self.lymph_serieses[idx_cell][::plot_every]
        num_cols = (len(lymphs_plot) // 3) + 1
        plotter = pv.Plotter(shape=(3, num_cols))
        for idx_plot, lymph in enumerate(lymphs_plot):
            plotter.subplot(idx_plot//num_cols, idx_plot%num_cols)
            lymph.plotRecon_singleDeg(plotter)
        plotter.show()
