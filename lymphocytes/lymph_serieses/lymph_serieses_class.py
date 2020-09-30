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
import glob

from lymphocytes.lymph_serieses.pca_methods import PCA_Methods
from lymphocytes.lymph_serieses.single_cell_methods import Single_Cell_Methods

from lymphocytes.lymph_snap.lymph_snap_class import Lymph_Snap


class Lymph_Serieses(PCA_Methods, Single_Cell_Methods):

    def __init__(self, stack_triplets):

        x_ranges = []
        y_ranges = []
        z_ranges = []

        self.lymph_serieses = []

        for (mat_filename, coeffPathStart, zoomedVoxelsPathStart) in stack_triplets:

            lymph_series = []

            f = h5py.File(mat_filename, 'r')
            frames = f['OUT/FRAME']

            for frame in np.array(frames).flatten():
                    lymph_series.append(Lymph_Snap(mat_filename = mat_filename, frame = frame, coeffPathStart = coeffPathStart, zoomed_voxels_path = zoomedVoxelsPathStart, speed = None, angle = None))

            self.lymph_serieses.append(lymph_series)
            print('One cell series initialised')

        self.num_serieses = len(self.lymph_serieses)

    """
    def sort_niigzList(self, niigzList):
        if len(os.path.basename(niigzList[0]).split("_")) == 2:
            idxs = [int(os.path.basename(i).split("_")[1][:-7]) for i in niigzList]
        else:
            idxs = [int(os.path.basename(i).split("_")[4][:-7]) for i in niigzList]
        niigz_sorted = [niigz for idx, niigz in sorted(zip(idxs, niigzList))]
        idxs = [idx for idx, niigz in sorted(zip(idxs, niigzList))]
        return niigz_sorted, idxs
    """


    """
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
    """

    """
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
    """



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
