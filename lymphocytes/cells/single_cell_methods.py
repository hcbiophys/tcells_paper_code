import matplotlib.pyplot as plt
import lymphocytes.utils.plotting as utils_plotting
import sys
import numpy as np
import pyvista as pv
import pickle
from scipy.interpolate import UnivariateSpline

import lymphocytes.utils.general as utils_general


class Single_Cell_Methods:
    """
    Inherited by Cell_Frame class.
    Contains methods for series of a single cell.
    """

    def _uropod_callback(self, a, b):
        """
        Callback for when selecting uropods
        """
        point = np.array(a.points[b, :])
        self.uropod_dict[self.frame_now] = point

    def select_uropods(self, idx_cell):
        """
        Select the uropods
        """
        self.uropod_dict = {}

        lymphs = self.cells[idx_cell]
        frames = [lymph.frame for lymph in lymphs]
        frames_done = []
        frames_none = []
        for idx_lymph, lymph in enumerate(lymphs):
            frame = lymph.frame
            if frame-1 in frames and frame+1 in frames and frame-1 in frames_done:
                self.uropod_dict[frame] = None
                frames_none.append(frame)

            else:
                print(frame)
                plotter = pv.Plotter()

                if idx_lymph == 0:
                    lymph.surface_plot(plotter=plotter, uropod_align=False, opacity = 0.1, scalars = None)
                else: # color face closest to prev uropod (adding point at that location makes point selection snap to that point)
                    print(self.uropod_dict[frames_done[-1]])
                    dists = [np.linalg.norm(self.uropod_dict[frames_done[-1]]-lymph.vertices[i, :]) for i in range(lymph.vertices.shape[0])]
                    idx_closest = dists.index(min(dists))
                    scalars = []
                    for idx in range(int(lymph.faces.shape[0]/4)):
                        if idx_closest in lymph.faces[idx*4:(idx+1)*4]:
                            scalars.append(0.5)
                        else:
                            scalars.append(0)
                    lymph.surface_plot(plotter=plotter, uropod_align=False, opacity = 0.5, scalars = scalars)

                self.frame_now = frame
                plotter.enable_point_picking(callback = self._uropod_callback, show_message=True,
                           color='pink', point_size=10,
                           use_mesh=True, show_point=True)
                #plotter.enable_cell_picking(through=False, callback = self._uropod_callback)
                plotter.show(cpos=[0, 1, 0])
                frames_done.append(frame)


        print('frames_done', frames_done)
        print('frames_none', frames_none)
        for frame in frames_none:
            self.uropod_dict[frame] = (self.uropod_dict[frame-1] + self.uropod_dict[frame+1])/2


        pickle_out = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/uropods/cell_{}.pickle'.format(idx_cell),'wb')
        pickle.dump(self.uropod_dict, pickle_out)



    def select_uropods_add_frames(self, idx_cell):
        """
        Select the uropods if only a few frames are missing (e.g. from getting new data)
        """

        self.uropod_dict = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/uropods/cell_{}.pickle'.format(idx_cell), "rb"))
        lymphs = self.cells[idx_cell]
        """
        for frame in range(116, 156):
            del self.uropod_dict[frame]
        """
        frames_done = list(self.uropod_dict.keys())

        for lymph in lymphs:
            if lymph.frame not in frames_done:
                plotter = pv.Plotter()
                frames_done_now = list(self.uropod_dict.keys())
                frame_dists = [abs(i - lymph.frame) for i in frames_done_now]
                closest_frame = frames_done_now[frame_dists.index(min(frame_dists))]
                print('closest_frame', closest_frame)
                print('self.uropod_dict[closest_frame]', self.uropod_dict[closest_frame])
                dists = [np.linalg.norm(self.uropod_dict[closest_frame]-lymph.vertices[i, :]) for i in range(lymph.vertices.shape[0])]
                idx_closest = dists.index(min(dists))
                print('idx', idx_closest)
                scalars = []
                for idx in range(int(lymph.faces.shape[0]/4)):
                    if idx_closest in lymph.faces[idx*4:(idx+1)*4]:
                        scalars.append(0.5)
                    else:
                        scalars.append(0)
                lymph.surface_plot(plotter=plotter, uropod_align=False, scalars = scalars, opacity = 0.5)

                self.frame_now =  lymph.frame
                plotter.enable_point_picking(callback = self._uropod_callback, show_message=True,
                           color='pink', point_size=10,
                           use_mesh=True, show_point=True)
                #plotter.enable_cell_picking(through=False, callback = self._uropod_callback)
                plotter.show(cpos=[0, 1, 0])

        pickle_out = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/uropods/cell_{}_updated.pickle'.format(idx_cell),'wb')
        pickle.dump(self.uropod_dict, pickle_out)






    def plot_orig_series(self, idx_cell, uropod_align, color_by = None, plot_every = 1):
        """
        Plot original mesh series, with point at the uropods
        """

        self._set_centroid_attributes('searching', time_either_side = 50)

        lymphs_plot = self.cells[idx_cell][::plot_every]
        num_cols=int(len(lymphs_plot)/3)+1
        plotter = pv.Plotter(shape=(3, num_cols), border=False)



        if color_by is not None:
            if color_by[:3] == 'pca':
                self._set_pca(n_components=3)
            elif color_by == 'delta_centroid' or color_by == 'delta_sensing_direction':
                self._set_centroid_attributes(color_by)
            elif color_by == 'morph_deriv':
                self._set_morph_derivs()
            vmin, vmax = utils_general.get_color_lims(self, color_by)

        for idx_plot, lymph in enumerate(lymphs_plot):
            plotter.subplot(idx_plot//num_cols, idx_plot%num_cols)


            plotter.add_text("{}".format(lymph.frame), font_size=10)
            #plotter.subplot(idx_plot, 0)

            color = (1, 1, 1)
            if color_by is not None:
                if getattr(lymph, color_by) is not None:
                    color = (1-(getattr(lymph, color_by)-vmin)/(vmax-vmin), 1, 1)

            mins, maxs = np.min(self.cells[idx_cell][0].vertices, axis = 0), np.max(self.cells[idx_cell][0].vertices, axis = 0)
            box = pv.Box(bounds=(mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]))
            lymph.surface_plot(plotter=plotter, uropod_align=uropod_align, color = color, box = box)

            if lymph.spin_vec is not None:
                plotter.add_lines(np.array([lymph.centroid, lymph.centroid+lymph.spin_vec*1000]), color = (0, 0, 0))
            print(lymph.spin_vec)



        plotter.show(cpos=[0, 1, 0])
        #plotter.show(cpos=[0, 0, 1])
        print('------')

    def plot_voxels_series(self, idx_cell, plot_every):

        lymphs_plot = self.cells[idx_cell][::plot_every]
        num_cols=int(len(lymphs_plot)/3)+1
        plotter = pv.Plotter(shape=(3, num_cols), border=False)

        for idx_plot, lymph in enumerate(lymphs_plot):
            plotter.subplot(idx_plot//num_cols, idx_plot%num_cols)

            voxels = np.array(lymph.voxels)
            voxels = np.moveaxis(np.moveaxis(voxels, 0, -1), 0, 1)
            #coordinates = np.argwhere(voxels == 1)*np.array(lymph.xyz_res) + 0.5*np.array(lymph.xyz_res)
            point_cloud = pv.PolyData(coordinates)
            plotter.add_mesh(point_cloud)

        plotter.show()


    def plot_uropod_centroid_line(self, idx_cell, plot_every):

        self._set_centroid_attributes('delta_sensing_direction')
        self._set_centroid_attributes('delta_centroid')

        lymphs_plot = self.cells[idx_cell][::plot_every]
        plotter = pv.Plotter()

        fig = plt.figure()
        ax_centroid  = fig.add_subplot(411)
        ax_uropod = fig.add_subplot(412)
        ax_deltas = fig.add_subplot(413)
        ax_actual_uropods = fig.add_subplot(414)

        frames = []
        mean_centroids = []
        mean_uropods = []
        actual_uropods = []
        delta_centroids = []
        delta_sensing_directions = []

        for idx_plot, lymph in enumerate(lymphs_plot):
            lymph.uropod_centroid_line_plot(plotter=plotter, color = (1, idx_plot/(len(lymphs_plot)-1), 1))

            plotter.add_mesh(pv.Sphere(radius=0.1, center=lymph.uropod), color = (1, 0, 0))
            plotter.add_mesh(pv.Sphere(radius=0.1, center=lymph.centroid), color = (0, 1, 0))

            frames.append(lymph.frame)
            actual_uropods.append(lymph.uropod)
            mean_centroids.append(lymph.mean_centroid)
            mean_uropods.append(lymph.mean_uropod)
            delta_centroids.append(lymph.delta_centroid)
            delta_sensing_directions.append(lymph.delta_sensing_direction)

        mean_centroids = [i if i is not None else np.array([np.nan, np.nan, np.nan]) for i in mean_centroids]
        mean_uropods = [i  if i is not None else np.array([np.nan, np.nan, np.nan]) for i in mean_uropods]
        actual_uropods -=  np.nanmean(np.array(actual_uropods), axis = 0)
        mean_centroids -= np.nanmean(np.array(mean_centroids), axis = 0)
        mean_uropods -=  np.nanmean(np.array(mean_uropods), axis = 0)

        for i in range(3):
            ax_centroid.plot(frames, [j[i] for j in mean_centroids])
        for i in range(3):
            ax_uropod.plot(frames, [j[i] for j in mean_uropods])

        ax_deltas.plot(frames, delta_centroids, color = 'red')
        ax_deltas.plot(frames, delta_sensing_directions, color = 'blue')
        ax_actual_uropods.plot(frames, actual_uropods)
        plt.show()

        plotter.show(cpos=[0, 1, 0])






    def plot_uropod_trajectory(self, idx_cell):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        uropods = [lymph.uropod for lymph in self.cells[idx_cell]]
        ax.plot([i[0] for i in uropods], [i[1] for i in uropods], [i[2] for i in uropods])


    def plot_migratingCell(self, idx_cell,  color_by = 'time', plot_every = 15):
        """
        Plot all meshes of a cell in one window



        lymphs = self.cells[idx_cell][::plot_every]
        plotter = pv.Plotter()

        if color_by != 'time':
            if color_by[:3] == 'pca':
                self._set_pca(n_components=3)
            elif color_by == 'delta_centroid' or color_by == 'delta_sensing_direction' or color_by == 'run':
                self._set_centroid_attributes(color_by, time_either_side = 2)
            elif color_by == 'morph_deriv':
                self._set_morph_derivs()

            vmin, vmax = utils_general.get_color_lims(self, color_by)
        """

        frames = [lymph.frame for lymph in self.cells[idx_cell]]


        fig = plt.figure()
        for idx, time_either_side in enumerate([5, 30, 55]):
            self._set_centroid_attributes('run', time_either_side = time_either_side)
            self._set_centroid_attributes('searching', time_either_side = time_either_side)

            ax = fig.add_subplot(2, 3, idx+1)
            runs = [lymph.run for lymph in self.cells[idx_cell]]
            ax.plot(frames, runs, c = 'blue')
            ax.plot(frames, [0 for _ in runs], c = 'black')
            ax.set_ylim([0, 0.02])

            searchings = [lymph.searching for lymph in self.cells[idx_cell]]
            ax.plot(frames, [i*10 if i is not None else None for i in searchings], c = 'red')
            ax.plot(frames, [0 for _ in searchings], c = 'black')

            ax = fig.add_subplot(2, 3, 3 + idx+1)
            ax.scatter(searchings, runs)

        plt.show()
        """
        # SEARCHING
        self._set_centroid_attributes('searching', time_either_side = 2, run_running_mean = True)
        plotter = pv.Plotter()
        dists =  [lymph.searching for lymph in self.cells[idx_cell]]
        dists = [dist if dist is not None else np.nan for dist in dists]
        plt.plot(frames, dists, c = 'blue')

        dists = np.convolve(dists, np.ones(5)/5, mode='valid')
        dists = [np.nan]*2 + list(dists) + [np.nan]*2
        plt.plot(frames, dists, c = 'red')
        #plt.ylim([0, 0.1])
        plt.show()
        """


        """
        spin_vecs =  [lymph.spin_vec for lymph in self.cells[idx_cell]]
        spin_vecs =  [spin_vec if spin_vec is not None else np.array([np.nan]*3) for spin_vec in spin_vecs]

        for idx, vec in enumerate(spin_vecs):
            if not np.isnan(vec[0]):
                color = (idx/(len(spin_vecs)-1), 0, 0)
                plotter.add_lines(np.array([[0, 0, 0], vec]), color = color)
                poly = pv.PolyData(vec)
                poly["My Labels"] = [str(idx)]
                plotter.add_point_labels(poly, "My Labels", point_size=20, font_size=36)
        plotter.show()
        """






        """
        for idx_lymph, lymph in enumerate(lymphs):
            surf = pv.PolyData(lymph.vertices, lymph.faces)

            color = (1, idx_lymph/(len(lymphs)-1), 1)

            if color_by != 'time':
                if getattr(lymph, color_by) is not None:
                    color = (1-(getattr(lymph, color_by)-vmin)/(vmax-vmin), 1, 1)

            if len(lymphs) == 1:
                plotter.add_mesh(surf, color = (1, 1, 1), opacity =  0.5)
            else:
                plotter.add_mesh(surf, color = color, opacity =  0.5)
            plotter.add_mesh(pv.Sphere(radius=1, center=lymph.uropod), color = (1, 0, 0))
        #box = pv.Box(bounds=(0, 92.7, 0, 52.7, 0, 26.4))
        #box = pv.Box(bounds=(0, 92.7, 0, 82.4, 0, 26.4))
        #plotter.add_mesh(box, style='wireframe')

        plotter.add_axes()
        plotter.show(cpos=[0, 1, 0.5])
        """

    def plot_attribute(self, idx_cell, attribute):

        if attribute[:3] == 'pca':
            self._set_pca(n_components=3)
        if attribute == 'delta_centroid' or attribute == 'delta_sensing_direction':
            self._set_centroid_attributes(attribute)

        lymphs = self.cells[idx_cell]
        frame_list = [lymph.frame for lymph in lymphs if getattr(lymph, attribute) is not None]
        attribute_list = [getattr(lymph, attribute) for lymph in lymphs if getattr(lymph, attribute)  is not None]

        plt.plot(frame_list, attribute_list)
        plt.show()


    def plot_series_PCs(self, idx_cell, plot_every):
        """
        Plot the PCs of each frame of a cell
        """
        fig = plt.figure(figsize = (1, 6))
        self._set_pca(n_components = 3)
        lymphs = self.cells[idx_cell][::plot_every]
        for idx, lymph in enumerate(lymphs):
            ax = fig.add_subplot(len(lymphs), 1, idx+1)
            ax.bar(range(len(lymph.pca_normalized)), lymph.pca_normalized)
            ax.set_ylim([-3.5, 3.5])
            ax.set_yticks([])
            ax.set_yticks([-3, 3])
            ax.set_xticks([])
        plt.tight_layout()
        plt.subplots_adjust(hspace = 0)
        plt.show()


    def plot_recon_series(self, idx_cell, plot_every, max_l = None, color_by = None):
        """
        Plot reconstructed mesh series
        """

        self._set_centroid_attributes('searching', time_either_side = 50)

        if color_by is not None:
            if color_by[:3] == 'pca':
                self._set_pca(n_components=3)
            else:
                self._set_centroid_attributes(color_by)
            vmin, vmax = utils_general.get_color_lims(self, color_by)

        lymphs_plot = self.cells[idx_cell][::plot_every]
        num_cols = (len(lymphs_plot) // 3) + 1
        plotter = pv.Plotter(shape=(3, num_cols))
        for idx_plot, lymph in enumerate(lymphs_plot):
            color = (1, 1, 1)
            if color_by is not None:
                if getattr(lymph, color_by) is not None:
                    color = (1-(getattr(lymph, color_by)-vmin)/(vmax-vmin), 1, 1)
            plotter.subplot(idx_plot//num_cols, idx_plot%num_cols)
            lymph.plotRecon_singleDeg(plotter, max_l = max_l, color = color)

            if lymph.spin_vec is not None:
                plotter.add_lines(np.array([lymph.centroid, lymph.centroid+lymph.spin_vec*1000]), color = (0, 0, 0))


        plotter.show(cpos=[0,0, 1])

    def plot_l_truncations(self, idx_cell):
        plotter = pv.Plotter(shape=(2, 4), border=False)
        for idx, l in enumerate([1, 2, 3, 4, 6, 9, 12, 15]):
            plotter.subplot(idx // 4, idx %4)
            self.cells[idx_cell][0].plotRecon_singleDeg(plotter=plotter, max_l = l, uropod_align = False)
            print('l', l)
        plotter.show()
