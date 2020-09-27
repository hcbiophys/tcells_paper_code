import numpy as np
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
import random
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lymphocyte_snap_class import *
from utils import *




class LymphocyteSerieses():

    def __init__(self, matFilenames_coeffPathStarts_niigzDirs_QUADS):

        x_ranges = []
        y_ranges = []
        z_ranges = []

        self.lymph_serieses = []

        for (mat_filename, coeffPathStart, niigzDir, exit_idxs) in matFilenames_coeffPathStarts_niigzDirs_QUADS:

            lymph_series = []

            f = h5py.File(mat_filename, 'r')
            dataset = f['DataOut/Surf']
            dataset_snaps = dataset.shape[1]

            niigzList = glob.glob(niigzDir + '*')
            niigz_sorted, idxs = self.sort_niigzList(niigzList)

            for idx in range(dataset_snaps):

                if idx in exit_idxs:
                    exited = True
                else:
                    exited = False
                if os.path.isfile(coeffPathStart + '{}_pp_surf_SPHARM_ellalign.txt'.format(idx)): # if SPHARM worked and found coefficients
                    lymph_series.append(LymphocyteSnap(mat_filename, coeffPathStart, idx, niigz_sorted[idx], None, None, exited))
                else:
                    lymph_series.append(LymphocyteSnap(mat_filename, None, idx, niigz_sorted[idx], None, None, exited))


            self.lymph_serieses.append(lymph_series)
            print('One cell series initialised')

        self.num_serieses = len(self.lymph_serieses)

        self.SH_extremes = None


    def plot_migratingCell(self, max_l = 5, idx_cell = 0, plot_every = 15):

        self.lymph_serieses = del_whereNone(self.lymph_serieses, 'coeff_array')

        fig_sing = plt.figure()
        fig_mult = plt.figure()


        ax_sing = fig_sing.add_subplot(111, projection='3d')

        num = len(self.lymph_serieses[idx_cell][::plot_every])
        for idx, lymph in enumerate(self.lymph_serieses[idx_cell]):
            if idx%plot_every == 0:
                print('idx: ', idx)
                ax_sing.plot_trisurf(lymph.vertices[0, :], lymph.vertices[1, :], lymph.vertices[2, :], triangles = np.asarray(lymph.faces[:, ::4]).T)

                ax = fig_mult.add_subplot(3, num, (idx//plot_every)+1, projection='3d')
                ax.plot_trisurf(lymph.vertices[0, :], lymph.vertices[1, :], lymph.vertices[2, :], triangles = np.asarray(lymph.faces[:, ::4]).T)

                ax = fig_mult.add_subplot(3, num, num + (idx//plot_every)+1, projection='3d')
                elev, azim = find_optimal_3dview(lymph.niigz)
                lymph.SH_plotRecon_singleDeg(ax, max_l, color_param = 'thetas', elev = elev, azim = azim, normaliseScale = False)

                azim += 90
                ax = fig_mult.add_subplot(3, num, (2*num) + (idx//plot_every)+1, projection='3d')
                lymph.SH_plotRecon_singleDeg(ax, max_l, color_param = 'thetas', elev = elev, azim = azim, normaliseScale = False)

        for ax in fig_sing.axes + fig_mult.axes[::3]:
            ax.grid(False)
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.set_xlim([0, 0.103*900])
            ax.set_ylim([0, 0.103*512])
            ax.set_zlim([0, 0.211*125])

        #equal_axes_notSquare(*fig.axes)



    def sort_niigzList(self, niigzList):
        if len(os.path.basename(niigzList[0]).split("_")) == 2:
            idxs = [int(os.path.basename(i).split("_")[1][:-7]) for i in niigzList]
        else:
            idxs = [int(os.path.basename(i).split("_")[4][:-7]) for i in niigzList]
        niigz_sorted = [niigz for idx, niigz in sorted(zip(idxs, niigzList))]
        idxs = [idx for idx, niigz in sorted(zip(idxs, niigzList))]
        return niigz_sorted, idxs



    def set_speedsAndTurnings(self):

        for series in self.lymph_serieses:

            idxs = [lypmh.idx for lypmh in series]
            dict = {}
            for idx, lymph in zip(idxs, series):
                dict[idx] = lymph
            for lymph in series:

                idx = lymph.idx
                # ANGLES
                if idx-1 in idxs and idx+1 in idxs:
                    vecs = []
                    for idx_ in [idx, idx+1]:

                        voxels_A = read_niigz(dict[idx_].niigz)
                        x_center_A, y_center_A, z_center_A = np.argwhere(voxels_A == 1).sum(0) / np.sum(voxels_A)
                        voxels_B = read_niigz(dict[idx_ - 1].niigz)
                        x_center_B, y_center_B, z_center_B = np.argwhere(voxels_B == 1).sum(0) / np.sum(voxels_B)

                        vecs.append( np.array([x_center_A-x_center_B, y_center_A-y_center_B, z_center_A-z_center_B]) )

                    angle = np.pi - np.arccos(np.dot(vecs[0], vecs[1])/(np.linalg.norm(vecs[0])*np.linalg.norm(vecs[1])))
                    lymph.angle = angle


                # SPEEDS
                if idx-2 in idxs and idx-1 in idxs and idx+1 in idxs and idx+2 in idxs:
                     to_avg = []
                     for idx_ in [idx-1, idx, idx+1, idx+2]:

                         voxels_A = read_niigz(dict[idx_].niigz)
                         x_center_A, y_center_A, z_center_A = np.argwhere(voxels_A == 1).sum(0) / np.sum(voxels_A)
                         voxels_B = read_niigz(dict[idx_ - 1].niigz)
                         x_center_B, y_center_B, z_center_B = np.argwhere(voxels_B == 1).sum(0) / np.sum(voxels_B)

                         speed = np.sqrt((x_center_A-x_center_B)**2 + (y_center_A-y_center_B)**2 + (z_center_A-z_center_B)**2)
                         to_avg.append(speed)
                     lymph.speed = np.mean(to_avg)


    def find_exiting_idxs(self):

        for idx in range(self.num_serieses):
            print('-'*100)
            print(idx)
            print('-'*20)
            for lymph in self.lymph_serieses[idx]:

                voxels = lymph.voxels
                if 1 in voxels[0, :, :] or 1 in voxels[-1, :, :] or 1 in voxels[:, 0, :] or 1 in voxels[:, -1, :] or 1 in voxels[:, :, 0] or 1 in voxels[:, :, -1]:
                    print(lymph.idx, 'EXITED')
                else:
                    print(lymph.idx)



    def plot_raw_volumes_series(self, zoom_factor = 1):

        for lymph_series in self.lymph_serieses:
            volumes = []
            for lymph in lymph_series:
                volumes.append(voxel_volume(lymph.niigz))
                print('idx', idx)

            plt.plot([i for i in range(len(volumes))], volumes)

        plt.ylim([0, 1.1*max(volumes)])

        plt.show()


    def plot_cofms(self, colorBy = 'speed'):

        self.set_speedsAndTurnings()

        cmap = plt.cm.viridis

        #if colorBy == 'speed' or colorBy == 'angle':
        #    self.lymph_serieses = del_whereNone(self.lymph_serieses, colorBy)

        #self.lymph_serieses = del_whereNone(self.lymph_serieses, 'coeff_array')

        fig = plt.figure()

        if colorBy == 'speed':
            speeds = [lymph.speed for sublist in self.lymph_serieses for lymph in sublist if lymph.speed is not None and lymph.coeff_array is not None]
            vmin, vmax = min(speeds), max(speeds)
        elif colorBy == 'angle':
            angles = [lymph.angle for sublist in self.lymph_serieses for lymph in sublist if lymph.angle is not None and lymph.coeff_array is not None]
            vmin, vmax = min(angles), max(angles)
        print('vmin', vmin, 'vmax', vmax)
        norm = plt.Normalize(vmin, vmax)


        for idx_ax, lymph_series in enumerate(self.lymph_serieses):
            print('new ax')
            ax = fig.add_subplot(2, (self.num_serieses//2)+2, idx_ax+1, projection = '3d')
            ax.grid(False)
            if colorBy == 'idx':
                ax.set_title(os.path.basename(self.lymph_serieses[idx_ax][0].mat_filename)[:-4] + '_' + str(min(colors_)) + '_' + str(max(colors_)))
            else:
                ax.set_title(os.path.basename(self.lymph_serieses[idx_ax][0].mat_filename)[:-4])
            for idx in range(len(lymph_series)-1):
                voxels_0 = read_niigz(lymph_series[idx].niigz)
                voxels_1 = read_niigz(lymph_series[idx+1].niigz)
                x_center_0, y_center_0, z_center_0 = np.argwhere(voxels_0 == 1).sum(0) / np.sum(voxels_0)
                x_center_1, y_center_1, z_center_1 = np.argwhere(voxels_1 == 1).sum(0) / np.sum(voxels_1)
                if colorBy == 'speed':
                    if lymph_series[idx].speed is None:
                        color = 'black'
                    else:
                        color = cmap(norm(lymph_series[idx].speed))
                elif colorBy == 'angle':
                    if lymph_series[idx].angle is None:
                        color = 'black'
                    else:
                        color = cmap(norm(lymph_series[idx].angle))
                if lymph_series[idx].coeff_array is None:
                    color = 'red'

                if lymph_series[idx].exited:
                    color = 'magenta'

                if color == 'black' or color == 'red' or color == 'magenta':
                    linewidth = 2
                else:
                    linewidth = 4
                ax.plot([x_center_0, x_center_1], [y_center_0, y_center_1], [z_center_0, z_center_1], c = color, linewidth = linewidth)


        ax = fig.add_subplot(2, (self.num_serieses//2)+2,self.num_serieses+2, projection = '3d')
        voxels = read_niigz(self.lymph_serieses[0][0].niigz)
        ax.voxels(voxels, edgecolors = 'white')


        """
        for idx, cofms_list in enumerate(cofms_lists):

            if colorBy == 'idx':
                colors_ = [lymph.idx for lymph in self.lymph_serieses[idx]]
                colors = cmap(colors_)
            elif colorBy == 'speed':
                colors_ = [lymph.speed for lymph in self.lymph_serieses[idx]]
                colors = cmap(colors_)
            elif colorBy == 'angle':
                colors_ = [lymph.angle for lymph in self.lymph_serieses[idx]]
                colors = cmap(colors_)
        """

        #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        #fig.colorbar(im, cax=cbar_ax)


        equal_axes_notSquare(*fig.axes)
        plt.show()



    def plot_individAndHists(self, variable):

        if variable == 'speed' or variable == 'angle':
            self.set_speedsAndTurnings()

        """
        For SPHARMs, variable is eg 'SH2'.
        """

        if variable[:-1] == 'PC':
            pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = 2, max_l = 5, rotInv = True)

        cells = []
        colors = []
        for idx_ax, lymph_series in enumerate(self.lymph_serieses):

            cell = []
            color = []
            idx_pc = 0
            for lymph in lymph_series:
                if lymph.exited:
                    cell.append(0)
                    color.append('magenta')
                elif variable == 'volume':
                    cell.append(voxel_volume(lymph.niigz))
                    color.append('blue')
                elif variable == 'speed':
                    if lymph.speed is None:
                        cell.append(0)
                        color.append('black')
                    else:
                        cell.append(lymph.speed)
                        color.append('blue')
                elif variable[:-1] == 'SH':
                    if lymph.coeff_array is None:
                        cell.append(0)
                        color.append('red')
                    else:
                        vector = lymph.SH_set_rotInv_vector(5)
                        cell.append(vector[int(variable[2:])])
                        color.append('blue')
                elif variable[:-1] == 'PC':
                    if lymph.coeff_array is None:
                        cell.append(0)
                        color.append('red')
                    else:
                        vector = lowDimRepSplit[idx_ax][idx_pc]
                        cell.append(vector[int(variable[2:])])
                        color.append('blue')
                        idx_pc += 1

            cells.append(cell)
            colors.append(color)

        fig = plt.figure(figsize = (30, 5))
        ax = fig.add_subplot(1, 1+self.num_serieses, 1)
        ax.hist([item for sublist in cells for item in sublist if item != 0], bins = 20, orientation = "horizontal")
        ax.set_ylabel(variable)
        for idx in range(self.num_serieses):
            ax = fig.add_subplot(1, 1+self.num_serieses, idx+2)
            ax.scatter(list(range(len(cells[idx]))), cells[idx], c = colors[idx], s = 1)
            ax.set_title(os.path.basename(self.lymph_serieses[idx][0].mat_filename)[:-4])

            ys = []
            xs = []
            for idx2 in range(len(cells[idx])):
                if cells[idx][idx2] != 0:
                    ys.append(cells[idx][idx2])
                    xs.append(idx2)
                elif cells[idx][idx2] == 0:
                    ax.plot(xs, ys, c = 'blue')
                    xs, ys = [], []
                ax.plot(xs, ys, c = 'blue')

        y_mins, y_maxs = [], []

        for ax in fig.axes:
            y_min, y_max = ax.get_ylim()
            y_mins.append(y_min)
            y_maxs.append(y_max)

        for ax in fig.axes:
            ax.set_ylim(min(y_mins), max(y_maxs))
        for ax in fig.axes[1:]:
            ax.set_xlim(0, 300)




    def C_plot_series_niigz(self, plot_every):

        lymph_series = self.lymph_serieses[0]

        niigzs = [lymph.niigz for lymph in lymph_series]

        niigzs = niigzs[::plot_every]

        num = len(niigzs)
        num_cols = (num // 3) + 1

        fig = plt.figure()

        for idx_file, file in enumerate(niigzs):
            voxels = read_niigz(file)

            ax = fig.add_subplot(3, num_cols, idx_file+1, projection = '3d')
            ax.voxels(voxels)


    def C_plot_recon_series(self, max_l, plot_every, color_param = 'phis'):

        lymph_series = []
        for i in self.lymph_serieses:
            lymph_series += i

        lymph_series = lymph_series[::plot_every]

        figRecons = plt.figure()

        num_cols = (len(lymph_series) // 3) + 1

        for idx_plot, lymph in enumerate(lymph_series):

            ax = figRecons.add_subplot(3, num_cols, idx_plot+1, projection = '3d')
            lymph.SH_plotRecon_singleDeg(ax, max_l, color_param)


        ax_list = figRecons.axes
        equal_axes(*ax_list)
        #remove_ticks(*ax_list)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        plt.show()


    def plot_rotInv_mean_std(self, maxl):

        vectors = []
        for lymph_series in self.lymph_serieses:
            for lymph in lymph_series:
                vector = lymph.SH_set_rotInv_vector(maxl)
                vectors.append(vector)

        vectors = np.array(vectors)
        means = np.mean(vectors, axis = 0)
        stds = np.std(vectors, axis = 0)

        fig= plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.bar(range(maxl+1), np.log10(means), color = 'red')
        ax = fig.add_subplot(1, 2, 2)
        ax.bar(range(maxl+1), np.log10(stds), color = 'blue')

        plt.show()


    def C_plot_rotInv_series_bars(self, maxl = 5, plot_every = 1, means_adjusted = False):

        pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = 2, max_l = maxl, rotInv = True)
        pc_idx = 1

        fig = plt.figure()
        vectors = []

        volumes = []
        speeds = []
        angles = []
        pc_vals = []

        num_cols = (len(self.lymph_serieses[0])//3) + 1

        for idx in range(len(self.lymph_serieses[0])):

            lymph = self.lymph_serieses[0][idx]
            vector = lymph.SH_set_rotInv_vector(maxl)
            vectors.append(vector)

            ax = fig.add_subplot(3, num_cols, idx+1)
            ax.bar(range(len(vectors[idx])), np.log10(vectors[idx]))
            ax.set_xlabel(str(idx), fontsize = 3.5)
            ax.set_ylim([0, 4])
            ax.set_xticks([])
            ax.set_yticks([])


    def plot_recons_increasing_l(self, lmax, l):

        vectors = []
        lymphs = []
        for lymph_series in self.lymph_serieses:
            for lymph in lymph_series:
                if not lymph.coeffPathStart is None:
                    vector = lymph.SH_set_rotInv_vector(lmax)
                    vectors.append(vector)
                    lymphs.append(lymph)

        ls = [vec[l] for vec in vectors]

        lymphs = [j for i,j in sorted(zip(ls, lymphs))][::int(len(ls)/8)]

        volumes = [voxel_volume(i.niigz) for i in lymphs]

        fig1 = plt.figure()
        for i in range(len(lymphs)):
            vol = volumes[i]

            ax = fig1.add_subplot(1, len(lymphs), i+1, projection = '3d')
            lymphs[i].SH_plotRecon_singleDeg(ax, lmax, 'phi', normaliseScale = True)
            ax.set_title(str(np.round(vol, 4)))

            ax.view_init(azim=0, elev=90)

        equal_axes(*fig1.axes)

        plt.show()



    def get_pca_objs(self, n_components, max_l, rotInv = True, removeSpeedNone = False, removeAngleNone = False, permAlterSeries = False):

        if permAlterSeries:
            self.lymph_serieses = del_whereNone(self.lymph_serieses, 'coeff_array')
            if removeSpeedNone:
                self.lymph_serieses = del_whereNone(self.lymph_serieses, 'speed')
            if removeAngleNone:
                self.lymph_serieses = del_whereNone(self.lymph_serieses, 'angle')
            lymph_serieses = self.lymph_serieses
        else:
            lymph_serieses = del_whereNone(self.lymph_serieses, 'coeff_array')
            if removeSpeedNone:
                lymph_serieses = del_whereNone(lymph_serieses, 'speed')
            if removeAngleNone:
                lymph_serieses = del_whereNone(lymph_serieses, 'angle')

        vectors = []
        idxs_newCell = [0]
        idx = 0
        for lymph_series in lymph_serieses:
            for lymph in lymph_series:
                if rotInv == True:
                    vector = lymph.SH_set_rotInv_vector(max_l)
                elif rotInv == False:
                    vector = lymph.SH_set_vector(max_l)
                vectors.append(vector)
                idx += 1
            idxs_newCell.append(idx)

        vectorsArray = np.array(vectors)

        print('Got SH Features')

        self.SH_extremes = np.zeros((15, 2))

        for l in range(vectorsArray[0, :].shape[0]-1):
            self.SH_extremes[l, 0], self.SH_extremes[l, 1] = np.min(vectorsArray[:, l+1]), np.max(vectorsArray[:, l+1])

        pca_obj = PCA(n_components = n_components)
        lowDimRepTogeth = pca_obj.fit_transform(vectorsArray)
        print('Done PCA')

        print('EXPLAINED VARIANCE RATIO: ', pca_obj.explained_variance_ratio_)

        lowDimRepSplit = []
        for idx in range(len(idxs_newCell)-1):
            split = []
            for idx_ in np.arange(idxs_newCell[idx], idxs_newCell[idx+1]):
                split.append(lowDimRepTogeth[idx_, :])

            lowDimRepSplit.append(split)


        return pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit


    def plot_rotInv_2Dmanifold(self, grid_size, max_l, pca, just_x = False, just_y = False):

        if pca:

            vectors = []
            lymphs = []
            pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = 2, max_l = max_l, rotInv = True, permAlterSeries = True)
            for idx_series, lymph_series in enumerate(self.lymph_serieses):
                for idx_cell, lymph in enumerate(lymph_series):
                    vectors.append(lowDimRepSplit[idx_series][idx_cell])
                    lymphs.append(lymph)

            c = list(zip(vectors, lymphs))
            random.shuffle(c)
            vectors, lymphs = zip(*c)


        else:
            vectors = []
            lymphs = []
            for lymph_series in self.lymph_serieses:
                for lymph in lymph_series:
                    if lymph.coeff_array is not None:
                        vector = lymph.SH_set_rotInv_vector(max_l)
                        vectors.append(vector)
                        lymphs.append(lymph)
            for idx in range(len(vectors)):
                vectors[idx] = np.array([vectors[idx][1], np.sum(vectors[idx][2:])])

            c = list(zip(vectors, lymphs))
            random.shuffle(c)
            vectors, lymphs = zip(*c)


        min1, max1 = min([v[0] for v in vectors]), max([v[0] for v in vectors])
        range1 = max1-min1
        min2, max2 = min([v[1] for v in vectors]), max([v[1] for v in vectors])
        range2 = max2-min2


        if just_x or just_y:
            grid_size = 10

        fig3D_long = plt.figure(figsize = (10, 10))
        fig3D_short = plt.figure(figsize = (10, 10))
        grids_done = []
        for idx, vector in enumerate(vectors):
            grid1 = (vector[0] - min1) // (range1/grid_size)
            grid2 = (vector[1] - min2) // (range2/grid_size)
            if not [grid1, grid2] in grids_done:

                if not just_x and not just_y:
                    ax =  fig3D_long.add_subplot(grid_size+1, grid_size+1, (grid_size+1)*grid2 + grid1+1, projection = '3d')
                    elev, azim = find_optimal_3dview(lymphs[idx].niigz)
                    lymphs[idx].SH_plotRecon_singleDeg(ax, max_l, 'phi', elev = elev, azim = azim, normaliseScale = True)

                    ax =  fig3D_short.add_subplot(grid_size+1, grid_size+1, (grid_size+1)*grid2 + grid1+1, projection = '3d')
                    azim += 90
                    lymphs[idx].SH_plotRecon_singleDeg(ax, max_l, 'phi', elev = elev, azim = azim, normaliseScale = True)

                    grids_done.append([grid1, grid2])
                    print('plotting at {},{}: '.format(grid1, grid2), lymphs[idx].niigz)
                    ax.set_title(os.path.basename(lymphs[idx].niigz)[:-7])
                elif just_x:
                    if grid2 == (np.mean([v[1] for v in vectors]) - min2) // (range2/grid_size):

                        ax =  fig3D_long.add_subplot(1, grid_size+1, grid1+1, projection = '3d')
                        elev, azim = find_optimal_3dview(lymphs[idx].niigz)
                        lymphs[idx].SH_plotRecon_singleDeg(ax, max_l, 'phi', elev = elev, azim = azim, normaliseScale = True)

                        ax =  fig3D_short.add_subplot(1, grid_size+1, grid1+1, projection = '3d')
                        azim += 90
                        lymphs[idx].SH_plotRecon_singleDeg(ax, max_l, 'phi', elev = elev, azim = azim, normaliseScale = True)

                        grids_done.append([grid1, grid2])
                        print('plotting at {},{}: '.format(grid1, grid2), lymphs[idx].niigz)
                        ax.set_title('x_' + os.path.basename(lymphs[idx].niigz)[:-7])
                elif just_y:
                    if grid1 == (np.mean([v[0] for v in vectors]) - min1) // (range1/grid_size):

                        ax =  fig3D_long.add_subplot(1, grid_size+1, grid2+1, projection = '3d')
                        elev, azim = find_optimal_3dview(lymphs[idx].niigz)
                        lymphs[idx].SH_plotRecon_singleDeg(ax, max_l, 'phi', elev = elev, azim = azim, normaliseScale = True)

                        ax =  fig3D_short.add_subplot(1, grid_size+1, grid2+1, projection = '3d')
                        azim += 90
                        lymphs[idx].SH_plotRecon_singleDeg(ax, max_l, 'phi', elev = elev, azim = azim, normaliseScale = True)

                        grids_done.append([grid1, grid2])
                        print('plotting at {},{}: '.format(grid1, grid2), lymphs[idx].niigz)
                        ax.set_title('y_' + os.path.basename(lymphs[idx].niigz)[:-7])



        equal_axes(*fig3D_long.axes)
        equal_axes(*fig3D_short.axes)

        figScatt = plt.figure()
        axScatt = figScatt.add_subplot()
        axScatt.scatter([v[0] for v in vectors], [v[1] for v in vectors], s = 2)





    def pca_plot_sampling(self, max_l, num_samples, color_param = None, rotInv = True, std = False):



        pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = 2, max_l = max_l, rotInv = rotInv)
        mean = np.mean(lowDimRepTogeth, axis = 0)
        dim_lims = []
        for dim in range(lowDimRepTogeth.shape[1]):
            min = np.min(lowDimRepTogeth[:, dim])
            max = np.max(lowDimRepTogeth[:, dim])
            dim_lims.append( (min, max) )

        """
        sigmas = []
        for dim in range(lowDimRepTogeth.shape[1]):
            mean = np.mean(lowDimRepTogeth[:, dim])
            std = np.std(lowDimRepTogeth[:, dim])
            sigmas.append( ( mean-2*std, mean-std, mean, mean+std, mean+2*std ) )
        """

        if rotInv == True:
            figSamples = plt.figure()

            pca0_points = []
            pca1_points = []
            pca_points_lists = [pca0_points, pca1_points]


            # normalise
            expansions = []
            for cell in range(lowDimRepTogeth.shape[0]):
                expansion = pca_obj.inverse_transform(lowDimRepTogeth[cell, :])
                expansions.append(expansion)

            mean_vector = np.mean(np.array(expansions), axis = 0)
            std_vector = np.std(np.array(expansions), axis = 0)
            print('mean', mean_vector)


            for dim in range(lowDimRepTogeth.shape[1]):
                for idx_sample in range(num_samples):

                    sample = np.mean(lowDimRepTogeth, axis = 0)

                    #sample[dim] = sigmas[dim][idx_sample]
                    sample[dim] = dim_lims[dim][0] + idx_sample*( dim_lims[dim][1]-dim_lims[dim][0] )/num_samples

                    expansion = pca_obj.inverse_transform(sample)
                    for l in range(len(expansion)-1):
                        expansion[l+1] = (expansion[l+1]-self.SH_extremes[l, 0])/(self.SH_extremes[l, 1]-self.SH_extremes[l, 0])


                    pca_points_lists[dim].append(sample)

                    ax = figSamples.add_subplot(num_samples+1, lowDimRepTogeth.shape[1], (idx_sample*lowDimRepTogeth.shape[1]) + (dim+1))

                    ax.bar([l for l in range(expansion.shape[0])], [i for i in expansion], color = 'magenta')

                    ax.set_ylim([-1.2, 1.2])

                    xmin, xmax = ax.get_xlim()
                    plt.plot([xmin, xmax], [0, 0], c = 'black', linewidth = 0.5)

                    ax.set_xticks([1, 2, 3, 4, 5])
                    if idx_sample != 3:
                        ax.set_xticks([])


            #ax = figSamples.add_subplot(num_samples+1, lowDimRepTogeth.shape[1], (idx_sample*lowDimRepTogeth.shape[1]) + (dim+1) + 1)
            #ax.bar([l for l in range(sample_expansion.shape[0])], [np.log10(i) for i in mean_vector], color = 'red')

            ax.set_xticks([1, 2, 3, 4, 5])


        if rotInv == False:


            fig3D = plt.figure()

            pca0_points = []
            pca1_points = []
            pca_points_lists = [pca0_points, pca1_points]

            for dim in range(lowDimRepTogeth.shape[1]):
                for idx_sample in range(num_samples):

                    sample = np.mean(lowDimRepTogeth, axis = 0)
                    sample[dim] = sigmas[dim][idx_sample]
                    sample_expansion = pca_obj.inverse_transform(sample)
                    pca_points_lists[dim].append(sample)

                    ax = fig3D.add_subplot(num_samples, lowDimRepTogeth.shape[1], (idx_sample*lowDimRepTogeth.shape[1]) + (dim+1), projection = '3d')

                    xcoeffs, ycoeffs, zcoeffs = np.split(sample_expansion, 3)
                    coeff_array_recon = np.concatenate([np.expand_dims(xcoeffs, 1), np.expand_dims(ycoeffs, 1), np.expand_dims(zcoeffs, 1)], axis = 1)
                    xs, ys, zs, phis, thetas = self.lymphSnaps_dict[0].SH_reconstruct_xyz_from_spharm_coeffs(coeff_array_recon, max_l)

                    tris = mtri.Triangulation(phis, thetas)
                    collec = ax.plot_trisurf([i.real for i in ys], [i.real for i in zs], [i.real for i in xs], triangles = tris.triangles, cmap=plt.cm.CMRmap, edgecolor='none', linewidth = 0, antialiased = False)
                    if color_param == 'phis':
                        colors = np.mean(phis[tris.triangles], axis = 1)
                        collec.set_array(colors)
                    elif color_param == 'thetas':
                        colors = np.mean(thetas[tris.triangles], axis = 1)
                        collec.set_array(colors)

            if rotInv == False:
                equal_axes(*fig3D.axes)

                for ax in fig3D.axes:
                    ax.grid(False)
                    ax.set_axis_off()
                    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))



    def pca_plot_shape_trajectories(self, max_l, rotInv = True, colorBy = 'time'):

        if colorBy == 'speed' or colorBy == 'angle':
            self.set_speedsAndTurnings()

        if colorBy == 'speed':
            pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = 2, max_l = max_l, rotInv = rotInv, removeSpeedNone = True, permAlterSeries = True)
        elif colorBy == 'angle':
            pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = 2, max_l = max_l, rotInv = rotInv, removeAngleNone = True, permAlterSeries = True)
        elif colorBy ==  'time':
            pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = 2, max_l = max_l, rotInv = rotInv)
        else:
            pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = 2, max_l = max_l, rotInv = rotInv, removeSpeedNone = False, permAlterSeries = True)

        if colorBy == 'time':
            list = [0, 1]
        elif colorBy == 'volume':
            list = [voxel_volume(lymph.niigz) for sublist in self.lymph_serieses for lymph in sublist]
        elif colorBy == 'speed':
            list = [lymph.speed for sublist in self.lymph_serieses for lymph in sublist]
        elif colorBy == 'angle':
            list = [lymph.angle for sublist in self.lymph_serieses for lymph in sublist]
        vmin, vmax = min(list), max(list)

        fig2D_sing = plt.figure()
        num_cols = (self.num_serieses // 3) +1
        fig2D_mult = plt.figure()

        if not colorBy == 'time':
            for idx_series, cmap in zip(range(self.num_serieses), [plt.cm.Blues_r for i in range(self.num_serieses)]):

                lowDimReps = lowDimRepSplit[idx_series]


                if colorBy == 'volume':
                    colors = [voxel_volume(i.niigz) for i in self.lymph_serieses[idx_series]]
                elif colorBy == 'speed':
                    colors = [i.speed for i in self.lymph_serieses[idx_series]]
                elif colorBy == 'angle':
                    colors = [i.angle for i in self.lymph_serieses[idx_series]]

                ax = fig2D_sing.add_subplot(111)
                im = ax.scatter([i[0] for i in lowDimReps], [i[1] for i in lowDimReps], c = colors, s = 8, vmin = vmin, vmax = vmax)

                ax = fig2D_mult.add_subplot(3, num_cols, idx_series+1)
                im = ax.scatter([i[0] for i in lowDimReps], [i[1] for i in lowDimReps], c = colors, vmin = vmin, vmax = vmax)
                ax.set_title(os.path.basename(self.lymph_serieses[idx_series][0].mat_filename)[:-4])

                ax.set_xlim([1.2*lowDimRepTogeth[:, 0].min(), 1.2*lowDimRepTogeth[:, 0].max()])
                ax.set_ylim([1.2*lowDimRepTogeth[:, 1].min(), 1.2*lowDimRepTogeth[:, 1].max()])

        else:

            cmaps = [plt.cm.Blues_r, plt.cm.Greens_r, plt.cm.Greys_r, plt.cm.Reds_r, plt.cm.Purples_r]*40

            for idx_series, series in enumerate(self.lymph_serieses):
                ax = fig2D_mult.add_subplot(3, num_cols, idx_series+1)

                ys = []
                xs = []

                count_withPC = 0
                idx_cmap = 0
                for lymph in series:
                    if lymph.coeff_array is not None:
                        xs.append(lowDimRepSplit[idx_series][count_withPC][0])
                        ys.append(lowDimRepSplit[idx_series][count_withPC][1])
                        count_withPC += 1
                    elif lymph.coeff_array is None:
                        colors = cmaps[idx_cmap]([i/len(ys) for i in range(len(ys))])
                        for idx in range(len(xs)-1):
                            ax.plot([xs[idx], xs[idx+1]], [ys[idx], ys[idx+1]], c = colors[idx])

                        ax.scatter(xs, ys, c = colors, s = 7)
                        xs, ys = [], []
                        idx_cmap += 1
                    colors = cmaps[idx_cmap]([i/len(ys) for i in range(len(ys))])
                    for idx in range(len(xs)-1):
                        ax.plot([xs[idx], xs[idx+1]], [ys[idx], ys[idx+1]], c = colors[idx])
                    ax.scatter(xs, ys, c = colors, s = 7)
                    ax.set_title(os.path.basename(self.lymph_serieses[idx_series][0].mat_filename)[:-4])


        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes('right', size='5%', pad=0.05)
        #fig2D_mult.colorbar(im, cax=cax, orientation='vertical')

        #cbar_ax = fig2D_sing.add_axes([0.85, 0.15, 0.05, 0.7])
        #fig2D_sing.colorbar(im, cax=cbar_ax)

        equal_axes_notSquare_2D(*fig2D_mult.axes)



    def plot_speeds_angles(self):

        self.set_speedsAndTurnings()

        for idx_series in range(self.num_serieses):
            speeds_cell = self.all_speeds[idx_series][1:-1]
            angles_cell = self.all_angles[idx_series][1:-1]

            colors = []
            for idx in range(len(speeds_cell)):

                if speeds_cell[idx] < 0.75 and angles_cell[idx] < 1.75:
                    colors.append('red')
                elif speeds_cell[idx] < 0.75 and angles_cell[idx] > 1.75:
                    colors.append('blue')
                else:
                    colors.append('green')

            plt.scatter(speeds_cell, angles_cell, c = colors)

        plt.xlabel('Speeds')
        plt.ylabel('Angles')
        plt.show()


    def correlate_with_speedAngle(self, max_l, rotInv, n_components, pca = False):

        self.set_speedsAndTurnings()

        self.lymph_serieses = del_whereNone(self.lymph_serieses, 'coeff_array')
        self.lymph_serieses = del_whereNone(self.lymph_serieses, 'speed')

        if pca:
            pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = n_components, max_l = max_l, rotInv = rotInv, permAlterSeries = True)
        else:
            lowDimRepSplit = []
            for idx_series in range(self.num_serieses):
                split = []
                for lymph in self.lymph_serieses[idx_series]:
                    split.append(lymph.SH_set_rotInv_vector(max_l))
                lowDimRepSplit.append(split)



        fig = plt.figure()
        speeds = []
        lowDimReps = []

        for idx, series in enumerate(self.lymph_serieses):
            for lymph in series:
                speeds.append(lymph.speed)
            lowDimReps_cell = lowDimRepSplit[idx]
            for lowDimRep in lowDimReps_cell:
                lowDimReps.append(lowDimRep)

        varNames = ['speeds']
        for idx_pc in range(n_components):
            pcs = [i[idx_pc] for i in lowDimReps]
            for idx_var, varList in enumerate([speeds]):
                ax = fig.add_subplot(2, n_components, (idx_var*n_components)+idx_pc+1)
                ax.scatter(pcs, varList, s = 1)
                if pca:
                    ax.set_xlabel('PC {}'.format(idx_pc))
                else:
                    ax.set_xlabel('Energy {}'.format(idx_pc))
                ax.set_ylabel(varNames[idx_var])

                corr, _ = pearsonr(pcs, varList)
                ax.set_title('pearson_corr: {}'.format(np.round(corr, 2)))





    def plot_pca_recons(self, n_pca_components, max_l, plot_every):

        pca_obj, max_l, lowDimRep = self.get_pca_objs(n_pca_components, max_l, rotInv = True)
        recon = pca_obj.inverse_transform(lowDimRep)

        figPCARecons = plt.figure()
        num_to_plot = self.num_snaps // plot_every + 1

        for snap_idx in range(0, self.num_snaps, plot_every):
            xcoeffs, ycoeffs, zcoeffs = np.split(recon[snap_idx, :], 3)
            coeff_array_recon = np.concatenate([np.expand_dims(xcoeffs, 1), np.expand_dims(ycoeffs, 1), np.expand_dims(zcoeffs, 1)], axis = 1)
            xs, ys, zs, phis, thetas = self.lymphSnaps_dict[snap_idx].SH_reconstruct_xyz_from_spharm_coeffs(coeff_array_recon, max_l)

            idx_plot = (snap_idx // plot_every) + 1
            ax = figPCARecons.add_subplot(5, (num_to_plot // 5) + 1, idx_plot, projection = '3d')

            tris = mtri.Triangulation(phis, thetas)
            ax.plot_trisurf([i.real for i in xs], [i.real for i in ys], [i.real for i in zs], triangles = tris.triangles)

        plt.show()
