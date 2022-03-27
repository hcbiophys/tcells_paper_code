import matplotlib.pyplot as plt
import tcells_paper_code.utils.plotting as utils_plotting
import sys
import numpy as np
import pyvista as pv
import pickle
from scipy.interpolate import UnivariateSpline
import os
import glob
from PIL import Image
from matplotlib.colors import ListedColormap

import tcells_paper_code.utils.general as utils_general


class Single_Cell_Methods:
    """
    Inherited by Videos class.
    Contains methods for series of a single cell.
    """

    def _uropod_callback(self, a, b):
        """
        Callback for when selecting uropods
        """
        point = np.array(a.points[b, :])
        self.uropod_dict[self.idx_frame_now] = point

    def select_uropods(self, idx_cell):
        """
        Select the uropods (alternating frames if it is surrounded by 2 consecutive frames, middle is linearly interpolated) for idx_cell
        note: the closest point to the reviously selected uropod is highlighted on the new mesh
        """
        self.uropod_dict = {}

        frames = self.cells[idx_cell]
        frames = [frame.idx_frame for frame in frames]
        frames_done = []
        frames_none = []
        for idx, frame in enumerate(frames):
            frame = frame.idx_frame

            if frame-1 in frames and frame+1 in frames and frame-1 in frames_done: # if it is surrounded by 2 consecutive frames, middle will be linearly interpolated
                self.uropod_dict[frame] = None
                frames_none.append(frame)

            else: # if not, select the uropod (aim for the center)
                print(frame)
                plotter = pv.Plotter()

                if idx == 0:
                    frame.surface_plot(plotter=plotter, uropod_align=False, opacity = 0.1, scalars = None, with_uropod = False)
                else: # color face closest to prev uropod
                    print(self.uropod_dict[frames_done[-1]])
                    dists = [np.linalg.norm(self.uropod_dict[frames_done[-1]]-frame.vertices[i, :]) for i in range(frame.vertices.shape[0])]
                    idx_closest = dists.index(min(dists))
                    scalars = []
                    for idx in range(int(frame.faces.shape[0]/4)):
                        if idx_closest in frame.faces[idx*4:(idx+1)*4]:
                            scalars.append(0.5)
                        else:
                            scalars.append(0)
                    frame.surface_plot(plotter=plotter, uropod_align=False, opacity = 0.5, scalars = scalars , with_uropod = False)

                self.idx_frame_now = frame
                plotter.enable_point_picking(callback = self._uropod_callback, show_message=True,
                           color='pink', point_size=10,
                           use_mesh=True, show_point=True)
                #plotter.enable_cell_picking(through=False, callback = self._uropod_callback)
                plotter.show(cpos=[0, 1, 0])

                frames_done.append(frame)


        for frame in frames_none: # linearly linterpolate
            self.uropod_dict[frame] = (self.uropod_dict[frame-1] + self.uropod_dict[frame+1])/2

        pickle_out = open('../data/uropods/{}.pickle'.format(idx_cell),'wb')
        pickle.dump(self.uropod_dict, pickle_out)




    def select_uropods_add_frames(self, idx_cell, frames_redo):
        """
        Select the uropods if only a few frames are missing (e.g. from getting new data or redoing bad selections)
        re-do for range within frames_redo, e.g. frames_redo=[67, 92]
        """

        self.uropod_dict = pickle.load(open('../data/uropods/cell_{}.pickle'.format(idx_cell), "rb"))
        frames = self.cells[idx_cell]

        for frame in range(frames_redo[0], frames_redo[1]): # deleted the bad frames, so these are now not in frames_done, below
            del self.uropod_dict[frame]

        frames_done = list(self.uropod_dict.keys())

        for frame in frames:
            if frame.idx_frame not in frames_done:
                plotter = pv.Plotter()

                # highlight the closest uropod label
                frames_done_now = list(self.uropod_dict.keys())
                frame_dists = [abs(i - frame.idx_frame) for i in frames_done_now]
                closest_frame = frames_done_now[frame_dists.index(min(frame_dists))]
                dists = [np.linalg.norm(self.uropod_dict[closest_frame]-frame.vertices[i, :]) for i in range(frame.vertices.shape[0])]
                idx_closest = dists.index(min(dists))
                scalars = []
                for idx in range(int(frame.faces.shape[0]/4)):
                    if idx_closest in frame.faces[idx*4:(idx+1)*4]:
                        scalars.append(0.5)
                    else:
                        scalars.append(0)
                frame.surface_plot(plotter=plotter, uropod_align=False, scalars = scalars, opacity = 0.5, with_uropod = False)

                self.idx_frame_now =  frame.idx_frame
                plotter.enable_point_picking(callback = self._uropod_callback, show_message=True,
                           color='pink', point_size=10,
                           use_mesh=True, show_point=True)
                #plotter.enable_cell_picking(through=False, callback = self._uropod_callback)
                plotter.show(cpos=[1, 0, 0])

        pickle_out = open('../data/uropods/{}_updated.pickle'.format(idx_cell),'wb')
        pickle.dump(self.uropod_dict, pickle_out)




    def show_video(self, idx_cell, color_by = None, save = False):
        """
        Show cell video
        Args:
        - idx_cell: the cell index (see main.py for the indices, or the data folder)
        - color_by: what to color the surfaces by, e.g. pca1 is PC 2 in the manuscript (since pca0 is PC 1, using computing counting)
        - save: whether to save into a folder, that can then be made into a video using e.g. the Quicktime app
        """
        extension = '' # optional extension for the filename to differentiate from previously saved ones

        if not os.path.isdir('/Users/harry/Desktop/lymph_vids/{}{}/'.format(idx_cell, extension)): # make folder to save the frames into
            os.mkdir('/Users/harry/Desktop/lymph_vids/{}{}/'.format(idx_cell, extension))

        cpos = [0, 0, 1] # camera orientation
        if idx_cell == 'zm_3_3_3' or idx_cell == 'zm_3_6_0':
            cpos = [1, 0, 0]


        video = self.cells[idx_cell]
        mins, maxs = np.min(video[0].vertices, axis = 0), np.max(video[0].vertices, axis = 0) # box coordinates
        box = pv.Box(bounds=(mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]))

        if color_by is not None:
            vmin, vmax = utils_general.get_color_lims(self, color_by) # color limits
        for frame in video:
            if frame.vertices is not None:
                if save:
                    plotter = pv.Plotter(off_screen=True, notebook = False)
                else:
                    plotter = pv.Plotter()

                color = (1, 1, 1)
                if color_by is not None:
                    if getattr(frame, color_by) is not None  and not np.isnan(getattr(frame, color_by)):
                        if color_by == 'pca1':
                            color = (1-(getattr(frame, color_by)-vmin)/(vmax-vmin), 1, 1)
                        elif color_by == 'pca2':
                            color = (1, 1, 1-(getattr(frame, color_by)-vmin)/(vmax-vmin))

                surf = pv.PolyData(frame.vertices, frame.faces)
                plotter.add_mesh(surf, opacity = 0.5, color = color)
                plotter.add_mesh(box, style='wireframe', opacity = 0.5)
                plotter.add_mesh(pv.Sphere(radius=1, center=frame.uropod), color = (1, 0, 0))
                plotter.add_mesh(pv.Sphere(radius=1, center=frame.centroid), color = (0, 0, 0))

                if save:
                    plotter.show(screenshot='/Users/harry/Desktop/lymph_vids/{}{}/{}.png'.format(idx_cell, extension, frame.idx_frame), cpos=cpos)
                    #print('/Users/harry/Desktop/lymph_vids/{}/{}.png'.format(idx_cell, frame.idx_frame))
                else:
                    plotter.show(cpos=cpos)

                plotter.close()
                pv.close_all()



    def add_colorbar_pic(self, idx_cell, old_frame_dir, new_frame_dir, pc012):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure

        if pc012 == 1:
            pcs = [i.pca1 for i in self.cells[idx_cell]]
            text = 'PC 2'
            rgb = [55, 212, 203]
        elif pc012 == 2:
            pcs = [i.pca2 for i in self.cells[idx_cell]]
            text = 'PC 3'
            rgb = [193, 192, 87]

        a = np.array([[min(pcs), max(pcs)]])
        fig = Figure()
        canvas = FigureCanvas(fig)

        N = 256
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(rgb[0]/256, 1, N)
        vals[:, 1] = np.linspace(rgb[1]/256, 1, N)
        vals[:, 2] = np.linspace(rgb[2]/256, 1, N)
        new_cmap = ListedColormap(vals[::-1, :])


        img = plt.imshow(a, cmap = new_cmap)
        plt.gca().set_visible(False)
        cax = plt.axes([0.1, 0.4, 0.8, 0.1])
        cbar = plt.colorbar(orientation="horizontal", cax=cax)
        #plt.show()
        #sys.exit()
        cbar.ax.tick_params(labelsize = 8)
        cbar.ax.set_xlabel(text, fontsize = 8)
        plt.savefig('/Users/harry/Desktop/colorbar_temp.png', dpi=80)

        colorbar_im = Image.open('/Users/harry/Desktop/colorbar_temp.png')
        colorbar_im = colorbar_im.crop((50, 190, 480, 270))




        for file in glob.glob(old_frame_dir + '/*'):
            im = Image.open(file)
            width, height = im.size
            #im.paste(colorbar_im.resize((500, 100)))
            im.paste(colorbar_im)

            im.save(new_frame_dir + os.path.basename(file))








    def plot_orig_series(self, idx_cell, uropod_align, color_by = None, plot_every = 1, plot_flat = False):
        """
        Plot original mesh series, with point at the uropods
        Args:
        - idx_cell: the cell index (see main.py for the indices, or the data folder)
        - uropod_align: whether to align based on uropod-centroid axis
        - color_by: e.g. pca0 is PC 1
        - plot_every: e.g. if this is 10 it plots every 10 frames
        - plot_flat: plot flat (only 1 row), or with rows and columns
        """

        frames_plot = self.cells[idx_cell][::plot_every]

        if color_by is not None:
            if color_by[:3] == 'pca':
                self._set_pca(n_components=3)
            elif color_by == 'morph_deriv':
                self._set_morph_derivs()
            elif color_by[:3] == 'speed':
                self._set_speed()
            elif color_by[:4] == 'spin' or color_by == 'angle':
                self._set_rotation(time_either_side = -1)
            vmin, vmax = utils_general.get_color_lims(self, color_by)



        num_per = 300 # change this number if want to e.g. view the frames in 50 frame batches
        for idx_start in range(len(frames_plot)//num_per + 1):
            frames_plot_section = frames_plot[idx_start*num_per:idx_start*num_per + num_per]
            num_cols=int(len(frames_plot_section)/2)+1
            if plot_flat:
                plotter = pv.Plotter(shape=(1, len(frames_plot)), border=False)
            else:
                plotter = pv.Plotter(shape=(2, num_cols), border=False)

            for idx_plot, frame in enumerate(frames_plot_section):

                if plot_flat:
                    plotter.subplot(0, idx_plot)
                else:
                    plotter.subplot(idx_plot//num_cols, idx_plot%num_cols)

                #plotter.add_text("{}".format(round((frame.idx_frame-frames_plot[0].idx_frame)*frame.t_res)), font_size=10) # label with the time
                #plotter.add_text("{}".format(frame.idx_frame), font_size=10)

                color = (1, 1, 1)
                if color_by is not None: # get color
                    if getattr(frame, color_by) is not None  and not np.isnan(getattr(frame, color_by)):
                        color = (1-(getattr(frame, color_by)-vmin)/(vmax-vmin), 1, 1)

                mins, maxs = np.min(self.cells[idx_cell][0].vertices, axis = 0), np.max(self.cells[idx_cell][0].vertices, axis = 0)
                box = pv.Box(bounds=(mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]))
                frame.surface_plot(plotter=plotter, uropod_align=uropod_align, color = color, opacity = 0.5, box = box)

                #frame.plotRecon_singleDeg(plotter=plotter, max_l = 1, opacity = 0.5) # can also add e.g. the ellipsoid component

            plotter.show(cpos=[0, 0, 1])



    def plot_voxels_series(self, idx_cell, plot_every):
        """
        Plot the voxels
        """

        frames_plot = self.cells[idx_cell][::plot_every]
        num_cols=int(len(frames_plot)/3)+1
        plotter = pv.Plotter(shape=(3, num_cols), border=False)

        for idx_plot, frame in enumerate(frames_plot):
            plotter.subplot(idx_plot//num_cols, idx_plot%num_cols)

            voxels = np.array(frame.voxels)
            voxels = np.moveaxis(np.moveaxis(voxels, 0, -1), 0, 1)
            coordinates = np.argwhere(voxels == 1)*np.array(frame.xyz_res) + 0.5*np.array(frame.xyz_res)
            point_cloud = pv.PolyData(coordinates)
            plotter.add_mesh(point_cloud)

        plotter.show()





    def plot_uropod_centroid_line(self, idx_cell, plot_every):
        """
        Plot how the uropod, centroid & uropod-centroid axis change in time in one subplot. One subplot shows with raw points, the other with smoothed points
        """

        frames_plot = self.cells[idx_cell][::plot_every]
        plotter = pv.Plotter(shape=(1, 2))

        plotter.subplot(0, 0) # with raw points
        for idx_plot, frame in enumerate(frames_plot):
            plotter.add_lines(np.array([frame.uropod, frame.centroid]), color = (1, idx_plot/(len(frames_plot)-1), 1))

            plotter.add_mesh(pv.Sphere(radius=0.3, center=frame.uropod), color = (1, 0, 0))
            plotter.add_mesh(pv.Sphere(radius=0.3, center=frame.centroid), color = (0,  0, 1))

            if not len(frames_plot) < 5 and idx_plot % int(len(frames_plot)/5) == 0:
                frame.surface_plot(plotter = plotter, opacity = 0.2, with_uropod = False)


        plotter.subplot(0, 1) # with smoothed points
        for idx_plot, frame in enumerate(frames_plot):
            if frame.mean_uropod is not None and frame.mean_centroid is not None:
                plotter.add_lines(np.array([frame.mean_uropod, frame.mean_centroid]), color = (1, idx_plot/(len(frames_plot)-1), 1))

                plotter.add_mesh(pv.Sphere(radius=0.3, center=frame.mean_uropod), color = (1, 0, 0))
                plotter.add_mesh(pv.Sphere(radius=0.3, center=frame.mean_centroid), color = (0,  0, 1))

                if not len(frames_plot) < 5 and idx_plot % int(len(frames_plot)/5) == 0:
                    frame.surface_plot(plotter = plotter, opacity = 0.2, with_uropod = False)

        plotter.show(cpos=[0, 1, 0])



    def plot_migratingCell(self, idx_cell, opacity = 0.5, plot_every = 15):
        """
        Plot some frames from a video in one subplot, colored by time
        """

        frames = self.cells[idx_cell][::plot_every]
        plotter = pv.Plotter()

        for idx, frame in enumerate(frames):
            surf = pv.PolyData(frame.vertices, frame.faces)

            color = (1, idx/(len(frames)-1), 1) # color by time

            plotter.add_mesh(surf, color = color, opacity =  opacity)
            plotter.add_mesh(pv.Sphere(radius=1, center=frame.uropod), color = (1, 0, 0))
            plotter.add_mesh(pv.Sphere(radius=1, center=frame.centroid), color = (0, 0, 0))
        #box = pv.Box(bounds=(0, 92.7, 0, 52.7, 0, 26.4))
        #box = pv.Box(bounds=(0, 92.7, 0, 82.4, 0, 26.4))
        #plotter.add_mesh(box, style='wireframe')

        #plotter.add_axes()
        plotter.show(cpos=[0, 1, 0.5])



    def plot_series_PCs(self, idx_cell, plot_every):
        """
        Plot the PCs of each frame of a cell
        """
        self._set_pca(n_components = 3)
        frames = self.cells[idx_cell][::plot_every]

        num_cols=int(len(frames)/5)+1
        fig = plt.figure()

        for idx, frame in enumerate(frames):
            ax = fig.add_subplot(5, num_cols, idx+1)
            ax.bar(range(len(frame.pca_normalized)), frame.pca_normalized)
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

        if color_by is not None:
            if color_by[:3] == 'pca':
                self._set_pca(n_components=3)

            vmin, vmax = utils_general.get_color_lims(self, color_by)

        frames_plot = self.cells[idx_cell][::plot_every]
        num_cols = (len(frames_plot) // 3) + 1
        plotter = pv.Plotter(shape=(3, num_cols))
        for idx_plot, frame in enumerate(frames_plot):
            color = (1, 1, 1)
            if color_by is not None:
                if getattr(frame, color_by) is not None:
                    color = (1-(getattr(frame, color_by)-vmin)/(vmax-vmin), 1, 1)
            plotter.subplot(idx_plot//num_cols, idx_plot%num_cols)
            frame.plotRecon_singleDeg(plotter, max_l = max_l, color = color)

            if frame.spin_vec is not None:
                plotter.add_lines(np.array([frame.centroid, frame.centroid+frame.spin_vec*1000]), color = (0, 0, 0))


        plotter.show(cpos=[0,0, 1])




    def plot_l_truncations(self, idx_cell):
        """
        Plot reconstructions with l truncated to show smoothing
        """
        plotter = pv.Plotter(shape=(2, 4), border=False)
        for idx, l in enumerate([1, 2, 3, 4, 6, 9, 12, 15]):
            plotter.subplot(idx // 4, idx %4)
            self.cells[idx_cell][0].plotRecon_singleDeg(plotter=plotter, max_l = l, uropod_align = False)
            print('l', l)
        plotter.show()
