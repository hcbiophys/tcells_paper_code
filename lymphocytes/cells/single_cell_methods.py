import matplotlib.pyplot as plt
import lymphocytes.utils.plotting as utils_plotting
import sys
import numpy as np
import pyvista as pv
import pickle
from scipy.interpolate import UnivariateSpline
import os

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
                    lymph.surface_plot(plotter=plotter, uropod_align=False, opacity = 0.1, scalars = None, with_uropod = False)
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
                    lymph.surface_plot(plotter=plotter, uropod_align=False, opacity = 0.5, scalars = scalars , with_uropod = False)

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
        print('HERE')

        self.uropod_dict = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/uropods/cell_{}.pickle'.format(idx_cell), "rb"))
        lymphs = self.cells[idx_cell]

        for frame in range(95, 110):
            del self.uropod_dict[frame]

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
                lymph.surface_plot(plotter=plotter, uropod_align=False, scalars = scalars, opacity = 0.5, with_uropod = False)

                self.frame_now =  lymph.frame
                plotter.enable_point_picking(callback = self._uropod_callback, show_message=True,
                           color='pink', point_size=10,
                           use_mesh=True, show_point=True)
                #plotter.enable_cell_picking(through=False, callback = self._uropod_callback)
                plotter.show(cpos=[1, 0, 0])

        pickle_out = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/uropods/cell_{}_updated.pickle'.format(idx_cell),'wb')
        pickle.dump(self.uropod_dict, pickle_out)




    def show_video(self, idx_cell, color_by = None, save = False):
        extension = ''

        if not os.path.isdir('/Users/harry/Desktop/lymph_vids/{}{}/'.format(idx_cell, extension)):
            os.mkdir('/Users/harry/Desktop/lymph_vids/{}{}/'.format(idx_cell, extension))

        cpos = [0, 0, 1]
        if idx_cell == 'zm_3_3_3' or idx_cell == 'zm_3_6_0':
            cpos = [1, 0, 0]




        lymph_series = self.cells[idx_cell]
        mins, maxs = np.min(lymph_series[0].vertices, axis = 0), np.max(lymph_series[0].vertices, axis = 0)
        box = pv.Box(bounds=(mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]))

        if color_by is not None:
            vmin, vmax = utils_general.get_color_lims(self, color_by)
        for lymph in lymph_series:
            if lymph.vertices is not None:
                if save:
                    plotter = pv.Plotter(off_screen=True, notebook = False)
                else:
                    plotter = pv.Plotter()

                #lymph.surface_plot(plotter=plotter, uropod_align=False, box = box, with_uropod = True, opacity = 0.5)

                color = (1, 1, 1)
                if color_by is not None:
                    if getattr(lymph, color_by) is not None  and not np.isnan(getattr(lymph, color_by)):
                        if color_by == 'pca1':
                            color = (1-(getattr(lymph, color_by)-vmin)/(vmax-vmin), 1, 1)
                        elif color_by == 'pca2':
                            color = (1, 1, 1-(getattr(lymph, color_by)-vmin)/(vmax-vmin))

                surf = pv.PolyData(lymph.vertices, lymph.faces)
                plotter.add_mesh(surf, opacity = 0.5, color = color)

                plotter.add_mesh(box, style='wireframe', opacity = 0.5)
                plotter.add_mesh(pv.Sphere(radius=1, center=lymph.uropod), color = (1, 0, 0))
                plotter.add_mesh(pv.Sphere(radius=1, center=lymph.centroid), color = (0, 0, 0))

                if save:
                    plotter.show(screenshot='/Users/harry/Desktop/lymph_vids/{}{}/{}.png'.format(idx_cell, extension, lymph.frame), cpos=cpos)
                    #print('/Users/harry/Desktop/lymph_vids/{}/{}.png'.format(idx_cell, lymph.frame))
                else:
                    plotter.show(cpos=cpos)



                plotter.close()
                pv.close_all()


    def plot_orig_series(self, idx_cell, uropod_align, color_by = None, plot_every = 1, plot_flat = False):
        """
        Plot original mesh series, with point at the uropods
        """

        lymphs_plot = self.cells[idx_cell][::plot_every]
        #num_cols=int(len(lymphs_plot)/3)+1
        #plotter = pv.Plotter(shape=(3, num_cols), border=False)


        if color_by is not None:
            if color_by[:3] == 'pca':
                self._set_pca(n_components=3)
            elif color_by == 'delta_centroid':
                self._set_centroid_attributes(color_by)
            elif color_by == 'morph_deriv':
                self._set_morph_derivs()
            elif color_by[:3] == 'run':
                self._set_centroid_attributes('run', time_either_side = -1)
            elif color_by[:4] == 'spin' or color_by == 'angle':
                self._set_centroid_attributes('searching', time_either_side = -1)
            vmin, vmax = utils_general.get_color_lims(self, color_by)



        num_per = 200
        for idx_start in range(len(lymphs_plot)//num_per + 1):
            lymphs_plot_section = lymphs_plot[idx_start*num_per:idx_start*num_per + num_per]
            num_cols=int(len(lymphs_plot_section)/2)+1
            if plot_flat:
                plotter = pv.Plotter(shape=(1, len(lymphs_plot)), border=False)
            else:
                plotter = pv.Plotter(shape=(2, num_cols), border=False)

            for idx_plot, lymph in enumerate(lymphs_plot_section):

                if plot_flat:
                    plotter.subplot(0, idx_plot)
                else:
                    plotter.subplot(idx_plot//num_cols, idx_plot%num_cols)

                plotter.add_text("{}".format(round((lymph.frame-lymphs_plot[0].frame)*lymph.t_res)), font_size=10)
                #plotter.add_text("{}".format(lymph.frame), font_size=10)


                color = (1, 1, 1)
                if color_by is not None:
                    if getattr(lymph, color_by) is not None  and not np.isnan(getattr(lymph, color_by)):
                        color = (1-(getattr(lymph, color_by)-vmin)/(vmax-vmin), 1, 1)

                mins, maxs = np.min(self.cells[idx_cell][0].vertices, axis = 0), np.max(self.cells[idx_cell][0].vertices, axis = 0)
                box = pv.Box(bounds=(mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]))
                lymph.surface_plot(plotter=plotter, uropod_align=uropod_align, color = color, opacity = 0.5, box = box)


                #lymph.plotRecon_singleDeg(plotter=plotter, max_l = 1, opacity = 0.5)

                #if lymph.ellipsoid_vec_smoothed is not None:
                    #plotter.add_lines(np.array([lymph.centroid, lymph.centroid+10*lymph.ellipsoid_vec_smoothed]), color = (1, 0, 0))





            plotter.show(cpos=[1, 0, 0])
            #plotter.show(cpos=[0, 0, 1])






    def plot_voxels_series(self, idx_cell, plot_every):

        lymphs_plot = self.cells[idx_cell][::plot_every]
        num_cols=int(len(lymphs_plot)/3)+1
        plotter = pv.Plotter(shape=(3, num_cols), border=False)

        for idx_plot, lymph in enumerate(lymphs_plot):
            plotter.subplot(idx_plot//num_cols, idx_plot%num_cols)

            voxels = np.array(lymph.voxels)
            voxels = np.moveaxis(np.moveaxis(voxels, 0, -1), 0, 1)
            coordinates = np.argwhere(voxels == 1)*np.array(lymph.xyz_res) + 0.5*np.array(lymph.xyz_res)
            point_cloud = pv.PolyData(coordinates)
            plotter.add_mesh(point_cloud)

        plotter.show()





    def plot_uropod_centroid_line(self, idx_cell, plot_every):

        lymphs_plot = self.cells[idx_cell][::plot_every]
        plotter = pv.Plotter(shape=(1, 2))

        plotter.subplot(0, 0)
        for idx_plot, lymph in enumerate(lymphs_plot):
            plotter.add_lines(np.array([lymph.uropod, lymph.centroid]), color = (1, idx_plot/(len(lymphs_plot)-1), 1))

            plotter.add_mesh(pv.Sphere(radius=0.3, center=lymph.uropod), color = (1, 0, 0))
            plotter.add_mesh(pv.Sphere(radius=0.3, center=lymph.centroid), color = (0,  0, 1))

            if not len(lymphs_plot) < 5 and idx_plot % int(len(lymphs_plot)/5) == 0:
                lymph.surface_plot(plotter = plotter, opacity = 0.2, with_uropod = False)


        lymphs_plot = self.cells[idx_cell][::plot_every]
        plotter.subplot(0, 1)


        for idx_plot, lymph in enumerate(lymphs_plot):
            if lymph.mean_uropod is not None and lymph.mean_centroid is not None:
                print('h', lymph.frame)
                plotter.add_lines(np.array([lymph.mean_uropod, lymph.mean_centroid]), color = (1, idx_plot/(len(lymphs_plot)-1), 1))

                plotter.add_mesh(pv.Sphere(radius=0.3, center=lymph.mean_uropod), color = (1, 0, 0))
                plotter.add_mesh(pv.Sphere(radius=0.3, center=lymph.mean_centroid), color = (0,  0, 1))

                if not len(lymphs_plot) < 5 and idx_plot % int(len(lymphs_plot)/5) == 0:
                    lymph.surface_plot(plotter = plotter, opacity = 0.2, with_uropod = False)

        plotter.show(cpos=[0, 1, 0])

        self._set_run()
        plt.plot([i.frame for i in lymphs_plot], [i.run_uropod for i in lymphs_plot])
        plt.show()






    def plot_migratingCell(self, idx_cell,  color_by = 'time', plot_every = 15):



        lymphs = self.cells[idx_cell][::plot_every]
        plotter = pv.Plotter()

        if color_by != 'time':
            if color_by[:3] == 'pca':
                self._set_pca(n_components=3)
            elif color_by == 'delta_centroid' or color_by == 'run':
                self._set_centroid_attributes(color_by, time_either_side = 2)
            elif color_by == 'morph_deriv':
                self._set_morph_derivs()

            vmin, vmax = utils_general.get_color_lims(self, color_by)


        frames = [lymph.frame for lymph in self.cells[idx_cell]]

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
            plotter.add_mesh(pv.Sphere(radius=1, center=lymph.centroid), color = (0, 0, 0))
        #box = pv.Box(bounds=(0, 92.7, 0, 52.7, 0, 26.4))
        #box = pv.Box(bounds=(0, 92.7, 0, 82.4, 0, 26.4))
        #plotter.add_mesh(box, style='wireframe')

        #plotter.add_axes()
        plotter.show(cpos=[0, 1, 0.5])


    def plot_attribute(self, idx_cell, attribute):

        if attribute[:3] == 'pca':
            self._set_pca(n_components=3)
        if attribute == 'delta_centroid':
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
        self._set_pca(n_components = 3)
        lymphs = self.cells[idx_cell][::plot_every]

        num_cols=int(len(lymphs)/5)+1
        fig = plt.figure()

        for idx, lymph in enumerate(lymphs):
            ax = fig.add_subplot(5, num_cols, idx+1)
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
