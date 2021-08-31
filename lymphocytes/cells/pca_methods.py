import numpy as np
import matplotlib.pyplot as plt
import lymphocytes.utils.general as utils_general
from sklearn.decomposition import PCA
import sys
from sklearn.preprocessing import StandardScaler
import copy
import pyvista as pv

import lymphocytes.utils.general as utils_general



class PCA_Methods:
    """
    Inherited by Lymph_Seriese class.
    Contains methods that involve PCA.
    """




    def _set_pca(self, n_components):
        """
        Args:
        - n_components: dimensionailty of low dimensional representation.
        - removedelta_centroidNone: whether to remove frames with delta_centroid None.
        - removeAngleNone: whether to remove frames with angle None.
        """

        if not self.pca_set:
            lymphs = utils_general.list_all_lymphs(self)
            RI_vectors = np.array([lymph.RI_vector for lymph in lymphs])

            pca_obj = PCA(n_components = n_components)
            pca_obj.fit_transform(RI_vectors)
            print('EXPLAINED VARIANCE RATIO: ', pca_obj.explained_variance_ratio_)

            for lymph in lymphs:
                lymph.pca = pca_obj.transform(lymph.RI_vector.reshape(1, -1))
                lymph.pca = np.squeeze(lymph.pca, axis = 0)
                lymph.pca0 = lymph.pca[0]
                lymph.pca1 = lymph.pca[1]
                lymph.pca2 = lymph.pca[2]

            self.pca_set = True

            return pca_obj

    def set_pca_normalized(self):

        lymphs = utils_general.list_all_lymphs(self)
        pcas = np.array([lymph.pca for lymph in lymphs])
        means = np.mean(pcas, axis = 0)
        stds = np.std(pcas, axis = 0)
        all_pcas_normalized = (pcas - means)/stds
        for idx, lymph in enumerate(lymphs):
            lymph.pca_normalized = all_pcas_normalized[idx, :]








    def PC_sampling(self, n_components):


        pca_obj = self._set_pca(n_components)
        lymphs = utils_general.list_all_lymphs(self)
        PCs = np.array([lymph.pca for lymph in lymphs])
        mins = np.min(PCs, axis = 0)
        maxs = np.max(PCs, axis = 0)
        mean = np.mean(PCs, axis = 0)
        print('PC mean', mean)

        samples = [-2, -1, 0, 1, 2]

        fig_sampling = plt.figure()

        for idx_PC in range(PCs.shape[1]):


            min_copy = copy.deepcopy(mean)
            min_copy[idx_PC] = mins[idx_PC]
            max_copy = copy.deepcopy(mean)
            max_copy[idx_PC] = maxs[idx_PC]

            for idx_sample, sample in enumerate([min_copy, mean, max_copy]):
                colors = ['red']*len(pca_obj.inverse_transform(max_copy))
                for i,j in enumerate(list(pca_obj.inverse_transform(max_copy)-pca_obj.inverse_transform(min_copy))):
                    if j > 0:
                        colors[i] = 'blue'

                inverted = pca_obj.inverse_transform(sample)
                ax = fig_sampling.add_subplot(n_components, 3, 3*idx_PC+idx_sample+1)
                ax.bar(range(len(inverted)), inverted, color = colors)
                ax.set_ylim([0, 4.2])
                ax.set_yticks([0, 4])
                if idx_sample != 0:
                    ax.set_yticks([])
                if idx_PC != 2:
                    ax.set_xticks([])

        plt.subplots_adjust(hspace = 0.1, wspace = 0)



    def plot_component_lymphs(self, grid_size, pca, plot_original):
        """
        Plot seperate sampling of each of the 3 components (meshes and scatter)
        """
        fig_bars = plt.figure()
        plotter = pv.Plotter(shape=(3, grid_size), border=False)
        plotted_points_all = []

        lymphs = utils_general.list_all_lymphs(self)
        random.shuffle(lymphs)

        if pca:
            self._set_pca(n_components = 3)
            self.set_pca_normalized()
            vectors = [lymph.pca for lymph in lymphs]
        else:
            vectors = [lymph.RI_vector for lymph in lymphs]


        for idx_component in range(3):
            color = ['pink']*3
            color[idx_component] = 'green'
            plotted_points = []

            min_ = min([v[idx_component] for v in vectors])
            max_ = max([v[idx_component] for v in vectors])
            range_ = max_ - min_

            for grid in range(grid_size):
                grid_vectors = [] # vectors that could be good for this part of the PC
                grid_lymphs = []
                for vector, lymph in zip(vectors, lymphs):
                    if int((vector[idx_component] - min_) // (range_/grid_size)) == grid:
                        grid_vectors.append(vector)
                        grid_lymphs.append(lymph)

                popped = np.array([np.delete(i, idx_component) for i in grid_vectors])
                dists_from_PC = [np.sqrt(np.sum(np.square(i))) for i in popped]
                idx_min = dists_from_PC.index(min(dists_from_PC))
                to_plot = grid_lymphs[idx_min]
                plotted_points.append(grid_vectors[idx_min])
                plotter.subplot(idx_component, grid)
                if plot_original:
                    to_plot.surface_plot(plotter, uropod_align=True)
                else:
                    to_plot.plotRecon_singleDeg(plotter, max_l = 2, uropod_align = True)

                ax = fig_bars.add_subplot(3, grid_size, (idx_component*grid_size)+grid+1)

                ax.bar(range(3), to_plot.pca_normalized, color = color)
                ax.set_ylim([-4, 4])
                ax.set_yticks([-3, 3])
                if grid != 0:
                    ax.set_yticks([])
                ax.set_xticks([])


            plt.subplots_adjust(hspace = 0.1, wspace = 0)
            plotted_points_all.append(plotted_points)

        plotter.show(cpos=[0, 1, 0])
        self._scatter_plotted_components(vectors, plotted_points_all)
