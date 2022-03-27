import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import h5py # Hierarchical Data Format 5
import nibabel as nib
from scipy.ndimage import zoom
from scipy.special import sph_harm
from matplotlib import cm, colors
import matplotlib.tri as mtri
from mayavi import mlab
import pyvista as pv
import os
from sklearn.decomposition import PCA
import scipy.stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import pickle
import random
from pykdtree.kdtree import KDTree
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from pyvista import examples
import time

from tcells_paper_code.videos.pca_methods import PCA_Methods
from tcells_paper_code.videos.single_cell_methods import Single_Cell_Methods
from tcells_paper_code.videos.motion_methods import Motion_Methods
from tcells_paper_code.frames.frame_class import Frame
from tcells_paper_code.morphodynamics.consecutive_frames_class import Consecutive_Frames
from tcells_paper_code.videos.curvature_lists import all_lists
from tcells_paper_code.videos.uncertainties import save_PC_uncertainties, get_tau_sig, save_curvatures

import tcells_paper_code.utils.disk as utils_disk
import tcells_paper_code.utils.plotting as utils_plotting
import tcells_paper_code.utils.general as utils_general


class Videos(Single_Cell_Methods, PCA_Methods, Motion_Methods):
    """
    Class for all cells
    Mixins are:
    - Single_Cell_Methods: methods suitable for a single cell series
    - PCA_Methods: methods without involving reduced-dimensionality representation (via PCA)
    - Motion_Methods: methods to set attributes based on centroid and uropod
    """


    def __init__(self, stack_attributes, cells_model, uropods_bool, keep_every_random = 1):
        """
        - stack_attributes: (idx_cell, mat_filename, coeffPathFormat, xyz_res, color, t_res)
        - cells_model: indexes of the cells to model, e.g. ['3_1_0', '3_1_2']
        - uropods_bool: set to False when setting the uropods
        - keep_every_random: if e.g. 10, only 1/10 cells are included. Speeds up speeds for just checking things
        """

        stack_attributes_dict = {i[0]:i for i in stack_attributes}

        self.cells = {}

        for idx_cell in cells_model:



            (idx_cell, mat_filename, coeffPathFormat, xyz_res, color, t_res) = stack_attributes_dict[idx_cell]

            print('idx_cell: {}'.format(idx_cell))
            video = []


            self.uropods = uropods_bool
            if self.uropods:
                uropods = pickle.load(open('../data/uropods/{}.pickle'.format(idx_cell), "rb"))

            if int(idx_cell[4:]) > 15:
                frames_all, voxels_all, vertices_all, faces_all = utils_disk.get_attribute_from_mat(mat_filename=mat_filename, zeiss_type = 'zeiss', idx_cell = int(idx_cell[-1]), include_voxels = False)
            else:
                frames_all, voxels_all, vertices_all, faces_all = utils_disk.get_attribute_from_mat(mat_filename=mat_filename, zeiss_type = 'not_zeiss',  include_voxels = False)

            for idx_frame in range(int(max(frames_all)+1)):
                if os.path.isfile(coeffPathFormat.format(idx_frame)): # if it's within arena and SPHARM-PDM worked

                    if np.random.randint(0, keep_every_random) == 0:
                        idx = frames_all.index(idx_frame)

                        if self.uropods:
                            uropod = np.array(uropods[frames_all[idx]])
                        else:
                            uropod = None
                        frame = Frame(mat_filename = mat_filename, idx_frame = frames_all[idx], coeffPathFormat = coeffPathFormat, voxels = voxels_all[idx], xyz_res = xyz_res,  idx_cell = idx_cell, uropod = uropod, vertices = vertices_all[idx], faces = faces_all[idx])
                        frame.color = np.array(color)
                        frame.t_res = t_res

                        video.append(frame)



            self.cells[idx_cell] = video


        if self.uropods and keep_every_random == 1:
            self.interoplate_SPHARM() # interpolate if only 1 missing

            # set  tau_sig, i.e. tau_sig
            for idx_cell, video in self.cells.items():
                tau_sig = get_tau_sig(idx_cell, video)
                for frame in video:
                    frame.tau_sig = tau_sig

            for idx_cell in self.cells.keys():
                print(idx_cell, ' tau_sig: {}'.format(self.cells[idx_cell][0].tau_sig))
                self._set_mean_uropod_and_centroid(idx_cell = idx_cell, time_either_side = self.cells[idx_cell][0].tau_sig/2)


        self.pca_obj = None
        self.frame_pcs_set = False





    def interoplate_SPHARM(self):
        """
        Interpolate if only 1 frame missing
        """

        for idx_cell, video in self.cells.items():

            video_new = []
            dict = utils_general.get_frame_dict(video)

            idxs_frames = list(dict.keys())

            for i in range(int(idxs_frames[0]), int(max(idxs_frames))+1):
                if i in idxs_frames:
                    video_new.append(dict[i])

                elif i not in idxs_frames and i-1 in idxs_frames and i+1 in idxs_frames:
                    uropod_interpolated = (dict[i-1].uropod + dict[i+1].uropod)/2
                    frame = Frame(mat_filename = None, idx_frame = i, coeffPathFormat = None, voxels = None, xyz_res = None,  idx_cell = video[0].idx_cell, uropod = uropod_interpolated, vertices = None, faces = None)
                    frame.color = dict[i-1].color
                    frame.t_res = dict[i-1].t_res
                    frame.centroid = (dict[i-1].centroid + dict[i+1].centroid)/2
                    frame.volume = (dict[i-1].volume + dict[i+1].volume)/2
                    frame.RI_vector = (dict[i-1].RI_vector + dict[i+1].RI_vector)/2
                    frame.is_interpolation = True
                    video_new.append(frame)
            self.cells[idx_cell] = video_new



    def low_high_PC1_vecs(self):
        """
        Plot how spherical harmonic spectra change across PC 1
        """


        frames_low_PC1 = []
        frames_high_PC1 = []
        frames = utils_general.list_all_frames(self)
        for frame in frames:
            if frame.pca[0] < 0:
                frames_low_PC1.append(frame)
            if frame.pca[0] > 0:
                frames_high_PC1.append(frame)


        low_PC1_vecs = []
        for frame in frames_low_PC1:
            low_PC1_vecs.append(frame.RI_vector)
        num_low_PC1 = len(low_PC1_vecs)
        low_PC1_vecs = np.array(low_PC1_vecs)

        high_PC1_vecs = []
        for frame in frames_high_PC1:
            high_PC1_vecs.append(frame.RI_vector)
        num_high_PC1 = len(high_PC1_vecs)
        high_PC1_vecs = np.array(high_PC1_vecs)


        bins = 100
        num_descriptors = 10
        fig = plt.figure()
        plotted = 0
        for idx in range(num_descriptors):
            ax = fig.add_subplot(1, num_descriptors, idx+1)
            low_hist = np.histogram(low_PC1_vecs[:, idx], bins=bins, range=[0, 5])[0]
            high_hist = np.histogram(high_PC1_vecs[:, idx], bins=bins, range=[0, 5])[0]
            ax.barh(np.linspace(0, 5, bins), width = -low_hist, height = 5./bins, color = 'red')
            ax.barh(np.linspace(0, 5, bins), width = high_hist, height = 5./bins, color = 'green')
            ax.set_xticks([])
            if idx > 0:
                ax.set_yticks([])
        plt.show()


        """
        for idx in range(5):
            plt.scatter([idx for _ in range(num_low_PC1)], low_PC1_vecs[:, idx], color = 'red')
            plt.scatter([idx for _ in range(num_high_PC1)], high_PC1_vecs[:, idx], color = 'blue')
        plt.show()
        """





    def plot_cumulatives(self):
        """
        Plot cumulatives of speed_uropod
        """

        labeled = []

        fig = plt.figure(figsize = (30, 30))

        self._set_speed()

        for video in self.cells.values():
            #linestyle = random.choice(['--', '-.', '-', ':'])
            linestyle = random.choice(['-'])
            for frame in video:
                frame.linestyle = linestyle

        for video in self.cells.values():
            t_res = video[0].t_res

            all_lists = utils_general.split_by_consecutive_frames(video, attribute='speed_uropod', and_nan = True)

            for i in all_lists:

                speeds = [j.speed_uropod for j in i]
                speeds_sum = np.nancumsum(speeds)
                times = [t_res*i for i,j in enumerate(speeds_sum)]
                color = max([np.linalg.norm(j.centroid-i[0].centroid) for j in i])
                color /= (max(times)-min(times))
                color_lim = 0.07

                if video[0].idx_cell not in labeled:
                    plt.plot(times, speeds_sum, label = video[0].idx_cell, c = (0, min(1, color/color_lim), 0), linestyle = video[0].linestyle)
                    #plt.plot(times, speeds_sum, label = video[0].idx_cell, c = video[0].color, linestyle = video[0].linestyle)
                    labeled.append(video[0].idx_cell)
                else:
                    plt.plot(times, speeds_sum, c = (0, min(1, color/color_lim), 0), linestyle = video[0].linestyle)
                    #plt.plot(times, speeds_sum, c = video[0].color, linestyle = video[0].linestyle)

        plt.show()




    def alignments(self, min_length, min_time_either_side = 50):
        """
        Find which axis cells follow when both uropod & centroid move with similar vector
        """

        for idx_cell in self.cells.keys():
            #self._set_mean_uropod_and_centroid(idx_cell = idx_cell, time_either_side = max(min_time_either_side, self.cells[idx_cell][0].tau_sig/2))
            self._set_mean_uropod_and_centroid(idx_cell = idx_cell, time_either_side = 100) # 100 for long timecale behavior

        self._set_speed()
        self._set_rotation(time_either_side = -1)

        UC_uropod_angles = []
        ellipsoid_uropod_angles = []

        for frame in utils_general.list_all_frames(self):

            if frame.delta_uropod is not None and frame.delta_centroid is not None and frame.ellipsoid_vec is not None:

                if np.linalg.norm(frame.delta_uropod) > min_length: # if it's moving enough
                    max_diff = np.linalg.norm(frame.delta_uropod) /2
                    if np.linalg.norm(frame.delta_uropod - frame.delta_centroid) < max_diff: # if uropod & centroid are moving in same direction

                        print(frame.idx_cell)

                        # find UC - velocity angle
                        vec1 = frame.delta_uropod
                        vec2 = frame.mean_centroid - frame.mean_uropod
                        cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                        if cos_angle < 0:
                            vec1 = - frame.delta_uropod
                            cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                        UC_uropod_angle = (360/6.283)*np.arccos(cos_angle)
                        UC_uropod_angles.append(UC_uropod_angle)

                        # find UC - ellipsoid angle

                        vec1 = frame.delta_uropod
                        vec2 = frame.ellipsoid_vec
                        cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                        if cos_angle < 0:
                            vec1 = - frame.delta_uropod
                            cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                        ellipsoid_uropod_angle = (360/6.283)*np.arccos(cos_angle)
                        ellipsoid_uropod_angles.append(ellipsoid_uropod_angle)

                    """
                    mins, maxs = np.min(utils_general.list_all_frames(self)[0].vertices, axis = 0), np.max(utils_general.list_all_frames(self)[0].vertices, axis = 0)
                    box = pv.Box(bounds=(mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]))
                    plotter = pv.Plotter()
                    #frame.surface_plot(plotter = plotter, opacity = 0.5, box = box)
                    plotter.add_lines(np.array([[0, 0, 0], frame.delta_uropod]), color = (1, 0, 0))
                    plotter.add_lines(np.array([[0, 0, 0], frame.delta_centroid]), color = (0, 1, 0))
                    plotter.add_mesh(pv.Sphere(radius=max_diff, center=frame.delta_uropod), color = (1, 0, 0), opacity = 0.5)
                    plotter.show(cpos=[1, 0, 0])
                    """

        plt.hist([UC_uropod_angles, ellipsoid_uropod_angles], bins=20, color = ['red', 'blue'])
        plt.show()



    def bimodality_emergence(self):
        """
        Plot speed_uropod with diff running mean windows to see when stop-and-run bimodality emerges
        """

        self._set_speed()

        fig_scat = plt.figure(figsize = (20, 20))
        axes = [fig_scat.add_subplot(3, 1, i+1) for i in range(3)]
        width_points = [[] for _ in range(3)]
        for video in self.cells.values():
            print(video[0].idx_cell)
            color = video[0].color
            for idx_width, width in enumerate([-1, 50, 100]): # different time windows

                self._set_speed_uropod_running_means(idx_cell = video[0].idx_cell, time_either_side = width)

                speed_uropod_running_means = [frame.speed_uropod_running_mean if frame.speed_uropod_running_mean is not None else np.nan for frame in video]
                times = [frame.idx_frame*frame.t_res for frame in video]
                print('speed_uropod_running_means', speed_uropod_running_means)
                axes[idx_width].scatter(times, speed_uropod_running_means, s = 1, c = color)
                #axes[idx_width].set_ylim(bottom=0)
                width_points[idx_width] += speed_uropod_running_means


        fig_hist = plt.figure()
        for idx, i in enumerate(width_points):
            i = [j for j in i if not np.isnan(j)]
            ax = fig_hist.add_subplot(3, 1, idx+1)
            ax.hist(i, bins = 15, orientation = 'horizontal', color = 'black')
            ax.set_yticks([])
            #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        plt.show()




    def plot_attributes(self, attributes):
        """
        Plot time series and histograms of cell attributes (e.g. volume, principal components etc. )
        """

        fig_lines, fig_hists = plt.figure(figsize = (2, 7)), plt.figure(figsize = (2, 7))

        axes_line = [fig_lines.add_subplot(len(attributes), 1, i+1) for i in range(len(attributes))]
        axes_hist = [fig_hists.add_subplot(len(attributes), 1, i+1) for i in range(len(attributes))]
        all_attributes = [[] for i in range(len(attributes))]

        for idx_attribute, attribute in enumerate(attributes):
            if attribute[:3] == 'pca':
                self._set_pca(n_components=3)

            if attribute == 'morph_deriv':
                self._set_morph_derivs()

        for video in self.cells.values():
            color = np.random.rand(3,)
            for idx_attribute, attribute in enumerate(attributes):

                framesNested = utils_general.get_nestedList_connectedframes(video)
                for frames in framesNested:
                    frame_list = [frame.idx_frame for frame in frames if getattr(frame, attribute) is not None]
                    attribute_list = [getattr(frame, attribute) for frame in frames if getattr(frame, attribute)  is not None]
                    axes_line[idx_attribute].plot([frames[0].t_res*i for i in frame_list], attribute_list, color = frames[0].color, label = frames[0].idx_cell)
                    all_attributes[idx_attribute] += attribute_list
                    if idx_attribute != len(attributes)-1:
                        axes_line[idx_attribute].set_xticks([])
                    axes_line[idx_attribute].set_yticks([])

        for idx_attribute in range(len(attributes)):
            axes_hist[idx_attribute].hist(all_attributes[idx_attribute], bins = 7, orientation = 'horizontal', color = 'darkblue')
            axes_hist[idx_attribute].invert_xaxis()
            if idx_attribute != len(attributes)-1:
                axes_hist[idx_attribute].set_xticks([])

        #fig_lines.legend()
        for fig in [fig_lines, fig_hists]:
            fig.tight_layout()
            fig.subplots_adjust(hspace = 0)
            for ax in fig.axes:
                ax.set_ylim(bottom=0)
        plt.show()



    def speeds_histogram(self):
        """
        Histogram comparing speed_uropod and speed_centroid
        """

        def filter(list):
            return [i for i in list if i < 0 and i > -0.04]


        for idx_cell in self.cells.keys():
            self._set_mean_uropod_and_centroid(idx_cell = idx_cell, time_either_side = self.cells[idx_cell][0].tau_sig/2)
        self._set_speed()


        frames = utils_general.list_all_frames(self)

        speed_uropods = [frame.speed_uropod for frame in frames if frame.speed_uropod is not None]
        speed_centroids = [frame.speed_centroid for frame in frames if frame.speed_centroid is not None]


        fig = plt.figure(figsize = (10, 10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.hist([speed_uropods, speed_centroids], bins=30, color = ['red', 'blue'])
        ax2.hist([filter(speed_uropods), filter(speed_centroids)], bins=30, color = ['red', 'blue'])
        plt.show()

        print('speed_uropods, std:{}, var:{}'.format(np.std(speed_uropods), np.var(speed_uropods)))
        print('speed_centroids, std:{}, var:{}'.format(np.std(speed_centroids), np.var(speed_centroids)))






    def correlation(self,  attributes):
        """
        Get pearson correlation coefficient between different attributes
        """

        self._set_pca(n_components=3)
        self._set_morph_derivs()
        self._set_speed()
        self._set_speed_uropod_running_means(time_either_side = 75) # for a full time window of 150 (i.e. where the bimodality emerges)


        fig_scatt, fig_r, fig_p = plt.figure(figsize = (20, 20)), plt.figure(), plt.figure()
        r_values = np.empty((len(attributes), len(attributes)))
        p_values = np.empty((len(attributes), len(attributes)))
        r_values[:], p_values[:] = np.nan, np.nan
        for idx_row, dependent in enumerate(attributes):
            for idx_col, independent in enumerate(attributes):
                if dependent != independent and dependent[:3] != 'pca' and idx_col < idx_row:
                    if independent[:3] != 'speed':

                        print(independent, dependent)

                        ax = fig_scatt.add_subplot(len(attributes), len(attributes), idx_row*len(attributes)+idx_col+1)
                        #ax.set_xlabel(independent)
                        #ax.set_ylabel(dependent)

                        plot_frames = [frame for video in self.cells.values() for frame in video if getattr(frame, independent) is not None and  getattr(frame, dependent) is not None]
                        xs = [getattr(frame, independent) for frame in plot_frames]
                        ys = [getattr(frame, dependent)  for frame in plot_frames]


                        colors = [frame.color  for frame in plot_frames]
                        result = scipy.stats.linregress(np.array(xs), np.array(ys))
                        ax.scatter(xs, ys, s=1, c = colors)

                        model_xs = np.linspace(min(list(xs)), max(list(xs)), 50)
                        #ax.plot(model_xs, [result.slope*i+result.intercept for i in model_xs], c = 'red')
                        ax.tick_params(axis="both",direction="in")
                        if idx_row != len(attributes)-1:
                            ax.set_xticks([])
                        if idx_col != 0:
                            ax.set_yticks([])

                        r_values[idx_row, idx_col] = result.rvalue
                        p_values[idx_row, idx_col] = result.pvalue
        fig_scatt.subplots_adjust(hspace=0, wspace=0)
        ax = fig_r.add_subplot(111)
        r_extreme = np.nanmax(abs(r_values))
        r = ax.imshow(r_values, cmap = 'PiYG', vmin = -r_extreme, vmax = r_extreme)
        matplotlib.cm.Blues.set_bad(color='white')
        fig_r.colorbar(r, ax=ax, orientation='horizontal')
        ax = fig_p.add_subplot(111)
        p = ax.imshow(p_values, cmap = 'Reds')
        matplotlib.cm.Reds.set_bad(color='white')
        fig_p.colorbar(p, ax=ax, orientation='horizontal')

        plt.show()


    def scatter_annotate(self,  independent, dependent):
        """
        Scatter 2 attributes, with hover annotation for debugging (e.g. seeing which cells and frames are at different locations)
        """

        fig = plt.figure()


        plot_frames = [frame for video in self.cells.values() for frame in video if getattr(frame, independent) is not None and  getattr(frame, dependent) is not None]
        xs = [getattr(frame, independent) for frame in plot_frames]
        ys = [getattr(frame, dependent)  for frame in plot_frames]
        colors = [frame.color  for frame in plot_frames]

        names = [frame.idx_cell + '-{}'.format(frame.idx_frame) for frame in plot_frames]

        ax = fig.add_subplot(111)
        sc = ax.scatter(xs, ys, s = 0.5, c = colors)

        ax.set_xlabel(independent)
        ax.set_ylabel(dependent)

        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = "{}".format(" ".join([names[n] for n in ind["ind"]]))
            annot.set_text(text)
            #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
        annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)



        fig.canvas.mpl_connect("motion_notify_event", hover)
        #fig2 = plt.figure()
        #i = [j for j in i if not np.isnan(j)]
        #ax = fig2.add_subplot(1, 1, 1)
        #ax.hist(i, bins = 10, orientation = 'horizontal')

        plt.show()




    def plot_rotations(self,  time_either_side):
        """
        Plot ellipsoid major axes and 'spin' vectors corresponding to rotations in these
        """
        self._set_rotation(time_either_side = time_either_side)


        plotter = pv.Plotter(shape=(2, 4), border=False)


        for idx, frames_plot in enumerate(self.cells.values()):
            plotter.subplot(0, idx)
            for idx_plot, frame in enumerate(frames_plot):


                vec = frame.spin_vec
                if vec is not None:
                    plotter.add_lines(np.array([np.array([0, 0, 0]), vec]), color = (1, idx_plot/(len(frames_plot)-1), 1))
                    plotter.add_lines(np.array([[0, 0, 0], [0.005, 0, 0]]), color = (0.9, 0.9, 0.9))
                    plotter.add_lines(np.array([[0, 0, 0], [0, 0.005, 0]]), color = (0.9, 0.9, 0.9))
                    plotter.add_lines(np.array([[0, 0, 0], [0, 0, 0.005]]), color = (0.9, 0.9, 0.9))

            plotter.subplot(1, idx)
            for idx_plot, frame in enumerate(frames_plot):
                vec = frame.ellipsoid_vec_smoothed
                if vec is not None:
                    plotter.add_lines(np.array([-vec, vec]), color = (1, idx_plot/(len(frames_plot)-1), 1))

        plotter.show()






    def gather_time_series(self, save_name):
        """
        Collect all cells into 'consecutive_frames_class' object ready for continuous wavelet transform (cwt) and similar time series analyses
        """


        self._set_pca(n_components=3)
        self._set_speed()
        self._set_speed_uropod_running_means(time_either_side = 80)
        self._set_rotation(time_either_side = 75)


        all_consecutive_frames = []

        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        for idx_cell, video in self.cells.items():
            print('Gathering',  idx_cell)

            count = 0
            consecutive_frames = Consecutive_Frames(name = str(idx_cell)+alphabet[count], t_res_initial = video[0].t_res)
            prev_frame = None
            for idx, frame in enumerate(video):
                if idx == 0 or frame.idx_frame-prev_frame == 1:
                    consecutive_frames.add(frame.idx_frame, frame.pca[0], frame.pca[1], frame.pca[2],  frame.speed_uropod, frame.speed_uropod_running_mean, frame.turning)

                else:
                    consecutive_frames.interpolate()
                    all_consecutive_frames.append(consecutive_frames)
                    count += 1
                    consecutive_frames = Consecutive_Frames(name = str(idx_cell)+alphabet[count], t_res_initial = video[0].t_res)
                    consecutive_frames.add(frame.idx_frame, frame.pca[0], frame.pca[1], frame.pca[2], frame.speed_uropod, frame.speed_uropod_running_mean, frame.turning)
                prev_frame = frame.idx_frame

            consecutive_frames.interpolate()
            all_consecutive_frames.append(consecutive_frames)

        pickle_out = open('../data/time_series/{}.pickle'.format(save_name),'wb')
        pickle.dump(all_consecutive_frames, pickle_out)
