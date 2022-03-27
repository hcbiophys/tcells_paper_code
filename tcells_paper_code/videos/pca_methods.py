import numpy as np
import matplotlib.pyplot as plt
import tcells_paper_code.utils.general as utils_general
from sklearn.decomposition import PCA
import sys
from sklearn.preprocessing import StandardScaler
import copy
import pyvista as pv
import random
import tcells_paper_code.utils.general as utils_general
import pickle
import tcells_paper_code.utils.disk as utils_disk
import os
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from tcells_paper_code.frames.frame_class import Frame

np.set_printoptions(threshold=sys.maxsize)


class PCA_Methods:
    """
    Inherited by Videos class.
    Contains methods that involve principal component analysis (PCA).
    """

    def _set_pca(self, n_components):
        """
        Set pca attribute of all frames, either from pre-loaded PCA, or re-compute it
        """

        if self.frame_pcs_set: # if this function has already been done (so frames already have PCs set)
            return

        frames = utils_general.list_all_frames(self)

        # re-compute if PCA not alrady loaded
        if self.pca_obj is None:
            RI_vectors = np.array([frame.RI_vector for frame in frames])
            self.pca_obj = PCA(n_components = n_components)
            self.pca_obj.fit_transform(RI_vectors)
            print('setting')

        # set pca attributes for all frames
        for frame in frames:
            frame.pca = self.pca_obj.transform(frame.RI_vector.reshape(1, -1))
            frame.pca = np.squeeze(frame.pca, axis = 0)
            frame.pca0 = frame.pca[0]
            frame.pca1 = frame.pca[1]
            frame.pca2 = frame.pca[2]

        # set normalized PCs
        pcas = np.array([frame.pca for frame in frames])
        means = np.mean(pcas, axis = 0)
        stds = np.std(pcas, axis = 0)
        all_pcas_normalized = (pcas - means)/stds
        for idx, frame in enumerate(frames):
            frame.pca_normalized = all_pcas_normalized[idx, :3]


        print('Explained variance: ', self.pca_obj.explained_variance_)
        print('Explained variance ratio: ', self.pca_obj.explained_variance_ratio_)
        print('components', self.pca_obj.components_[:3, :])

        """
        fig = plt.figure()
        ax = fig.add_subplot(311)
        ax.bar(list(range(5)), self.pca_obj.components_[0, :5], color = 'black')
        ax = fig.add_subplot(312)
        ax.bar(list(range(5)), self.pca_obj.components_[1, :5], color = 'black')
        ax = fig.add_subplot(313)
        ax.bar(list(range(5)), self.pca_obj.components_[2, :5], color = 'black')
        for ax in fig.axes:
            ax.set_ylim([-1, 1])
            ax.set_xticks([])
        plt.show()
        sys.exit()
        """


        #pickle.dump(self.pca_obj, open('/Users/harry/OneDrive - Imperial College London/lymphocytes/pca_obj.pickle', 'wb'))

        self.frame_pcs_set = True

    def expl_var_bar_plot(self):
        """
        Plot how dimensionality varies along PC 1
        """

        frames_low_PC1 = []
        frames_high_PC1 = []
        frames = utils_general.list_all_frames(self)
        for frame in frames:
            if frame.pca[0] < 0:
                frames_low_PC1.append(frame)
            if frame.pca[0] > 0:
                frames_high_PC1.append(frame)


        pca_low = PCA(n_components = 5)
        pca_low.fit_transform(np.array([frame.RI_vector for frame in frames_low_PC1]))
        pca_high = PCA(n_components = 5)
        pca_high.fit_transform(np.array([frame.RI_vector for frame in frames_high_PC1]))

        for which_pc, shift, color in zip([pca_low,  pca_high], [-0.2,  0.2], ['red',  'green']):
            plt.barh([i+shift for i in [0, 1, 2, 3, 4]], getattr(which_pc, 'explained_variance_ratio_')[:5][::-1], height=0.4, color = color)
        plt.yticks([])
        plt.show()





    def PC_arrows(self):
        """
        Visualise the PC arrows in representation space
        """

        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
                FancyArrowPatch.draw(self, renderer)

        #self._set_pca(n_components=3)
        #components = self.pca_obj.components_[:, :3]
        components = np.array([[ 0.8072487,   0.45532269,  0.34706196,],
        [-0.14969418, -0.45074785,  0.80496108,],
        [-0.56602573,  0.75203689,  0.23602912]])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in self.cells['2_8']:
            ax.scatter(*i.RI_vector[:3], c = 'red')
        for i in self.cells['zm_3_4_0']:
            ax.scatter(*i.RI_vector[:3], c = 'blue')

        ax.set_xlabel('RI0')
        ax.set_ylabel('RI1')
        ax.set_zlabel('RI2')
        ax.set_xlim([1, 3])
        ax.set_ylim([3, 4])
        ax.set_zlim([0, 1])

        mean = np.mean([frame.RI_vector for frame in utils_general.list_all_frames(self)], axis = 0)

        for row, color in zip(range(components.shape[0]), ['r', 'b', 'g']):
            a = Arrow3D([mean[0], mean[0]+components[row, 0]*4], [mean[1], mean[1]+components[row, 1]*4],
                    [mean[2], mean[2]+components[row, 2]*4], mutation_scale=20,
                    lw=3, arrowstyle="-|>", color=color)
            ax.add_artist(a)

        plt.show()



    def PC_sampling(self, n_components = 3):
        """
        For the min, mean and max of each PC (others set to 0), invert to find SPHARM rep
        """

        self._set_pca(n_components)

        frames = utils_general.list_all_frames(self)
        PCs = np.array([frame.pca for frame in frames])
        mins = np.min(PCs, axis = 0)
        maxs = np.max(PCs, axis = 0)
        mean = np.mean(PCs, axis = 0)

        fig_sampling = plt.figure()

        for idx_PC in range(3):


            min_copy = copy.deepcopy(mean)
            min_copy[idx_PC] = mins[idx_PC]
            max_copy = copy.deepcopy(mean)
            max_copy[idx_PC] = maxs[idx_PC]

            for idx_sample, sample in enumerate([min_copy, mean, max_copy]):
                colors = ['red']*len(self.pca_obj.inverse_transform(max_copy))
                for i,j in enumerate(list(self.pca_obj.inverse_transform(max_copy)-self.pca_obj.inverse_transform(min_copy))):
                    if j > 0:
                        colors[i] = 'blue'

                inverted = self.pca_obj.inverse_transform(sample)
                ax = fig_sampling.add_subplot(n_components, 3, 3*idx_PC+idx_sample+1)
                ax.bar(range(len(inverted[:5])), inverted[:5], color = colors)
                ax.set_ylim([0, 4.2])
                ax.set_yticks([0, 4])
                if idx_sample != 0:
                    ax.set_yticks([])
                if idx_PC != 2:
                    ax.set_xticks([])

        plt.subplots_adjust(hspace = 0.1, wspace = 0)
        plt.show()



    def plot_component_frames(self, bin_size, pca, plot_original, max_l):
        """
        Plot seperate sampling of each of the 3 components (meshes and scatter), and also the corresponding (normalised PCs),
        and a coloured scatter of where they are in PC space
        Args
        - bin_size: number of bins
        - pca: if True, samples PCs, if not, samples SPHARM
        - plot_original: if True, plots meshes, if False plots max_l recontructions
        """
        fig_bars = plt.figure()
        plotter = pv.Plotter(shape=(3, bin_size), border=False)
        plotted_points_all = []

        frames = utils_general.list_all_frames(self)
        random.shuffle(frames)

        if pca:
            self._set_pca(n_components = 3)
            vectors = [frame.pca for frame in frames]
        else:
            vectors = [frame.RI_vector for frame in frames]

        for idx_component in range(3):
            print('idx_component', idx_component)

            color = ['dodgerblue']*3
            color[idx_component] = 'black'
            plotted_points = []

            min_ = min([v[idx_component] for v in vectors])
            max_ = max([v[idx_component] for v in vectors])
            range_ = max_ - min_

            for bin in range(bin_size):

                bin_vectors = [] # vectors that could be good for this part of the PC
                bin_frames = []
                for vector, frame in zip(vectors, frames):

                    if round((vector[idx_component] - min_) // (range_/bin_size)) == bin:
                        bin_vectors.append(vector)
                        bin_frames.append(frame)

                # find distances from the PC axis, and choose frame with the minimum distance
                popped = np.array([np.delete(i, idx_component) for i in bin_vectors])
                dists_from_PC = [np.sqrt(np.sum(np.square(i))) for i in popped]
                idx_min = dists_from_PC.index(min(dists_from_PC))
                to_plot = bin_frames[idx_min]
                plotted_points.append(bin_vectors[idx_min])
                plotter.subplot(idx_component, bin)


                if plot_original:
                    to_plot.surface_plot(plotter, uropod_align=True)
                else:
                    to_plot.plotRecon_singleDeg(plotter, max_l = max_l, uropod_align = True)

                ax = fig_bars.add_subplot(3, bin_size, (idx_component*bin_size)+bin+1)

                ax.bar(range(3), to_plot.pca_normalized, color = color)
                ax.set_ylim([-4, 4])
                ax.set_yticks([-3, 3])
                if bin != 0:
                    ax.set_yticks([])
                ax.set_xticks([])

            plt.subplots_adjust(hspace = 0.1, wspace = 0)
            plotted_points_all.append(plotted_points)

        plotter.show(cpos=[0, 1, 0])
        self._scatter_plotted_components(vectors, plotted_points_all)


    def _scatter_plotted_components(self, vectors, plotted_points_all):
        """
        Scatter points of the plotted meshes
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        for i in vectors:
            ax1.scatter(i[0], i[1], i[2], s = 1, c = 'lightskyblue')
        for i, color in zip(plotted_points_all, ['red', 'green', 'black']):
            for j in i:
                ax1.scatter(j[0], j[1], j[2], s = 6, c = color)
                ax2.scatter(j[0], j[1], j[2], s = 6, c = color)
        plt.show()




    def plot_PC_space(self, plot_original, max_l):
        """
        Plot a sample of cells in the 3D PCA space
        """

        self._set_pca(n_components = 3)
        frames = utils_general.list_all_frames(self)
        random.shuffle(frames)

        plotter = pv.Plotter()
        scale_factor = 40

        coords_plotted = [] # track coordinates of plotted so far
        for frame in frames:
            if len([i for i in coords_plotted if np.linalg.norm(frame.pca_normalized-i) < 0.75]) == 0 and not frame.is_interpolation: # if it's not too close to one already plotted

                vertices, faces, uropod = frame._get_vertices_faces_plotRecon_singleDeg(max_l = max_l, uropod_align = True, horizontal_align = False)
                if plot_original:
                    #uropod, centroid, vertices = frame._uropod_align(axis = np.array([0, 0, -1]))
                    uropod, centroid, vertices = frame._uropod_and_horizontal_align()
                    faces = frame.faces
                vertices /= scale_factor
                vertices += frame.pca_normalized
                uropod = np.float64(uropod) + frame.pca_normalized

                surf = pv.PolyData(vertices, faces)

                surf = surf.decimate(0.98)

                #color = (1, 1, 1)
                vmin, vmax = utils_general.get_color_lims(self, color_by = 'pca1')
                color = (1-(frame.pca1-vmin)/(vmax-vmin), 1, 1)


                plotter.add_mesh(surf, color = color)


                plotter.add_mesh(pv.Sphere(radius=0.5/scale_factor, center=uropod), color = (1, 0, 0))
                plotter.add_lines(np.array([[-3, 0, 0], [3, 0, 0]]), color = (0, 0, 0)) # axes
                plotter.add_lines(np.array([[0, -3, 0], [0, 3, 0]]), color = (0, 0, 0))
                plotter.add_lines(np.array([[0, 0, -3], [0, 0, 3]]), color = (0, 0, 0))

                #poly = pv.PolyData(np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]]))
                #poly["My Labels"] = ['PC 1', 'PC 2', 'PC 3']
                #plotter.add_point_labels(poly, "My Labels", point_size=2, font_size=25)

                coords_plotted.append(frame.pca_normalized)

        plotter.show(cpos = (1, -1, 0.5))
