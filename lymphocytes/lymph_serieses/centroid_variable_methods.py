import numpy as np
import matplotlib.pyplot as plt
import lymphocytes.utils.plotting as utils_plotting
import lymphocytes.utils.general as utils_general

class Centroid_Variable_Methods:

    def _set_centroids(self):

        for lymph in [lymph for lymph_series in self.lymph_serieses for lymph in lymph_series]:
            voxels = lymph.zoomed_voxels
            x, y, z = np.argwhere(voxels == 1).sum(0) / np.sum(voxels)
            lymph.centroid = (0.103*92.7*(x/180), 0.103*52.7*(y/102), 0.211*26.4*(z/25))
            lymph.centroid = np.array([x, y, z])

    def plot_centroids(self, color_by = 'speed'):
        """
        Plots the centre of mass trajectories of each cell.
        Args:
        - colorBy: attribute to color by ("speed" / "angle").
        """

        self._set_speeds()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        vmin, vmax = utils_general.get_color_lims(self.lymph_serieses, color_by)


        for idx_ax, lymph_series in enumerate(self.lymph_serieses):


            for idx in range(len(lymph_series)-1):
                (x_center_0, y_center_0, z_center_0) = lymph_series[idx].centroid
                (x_center_1, y_center_1, z_center_1) = lymph_series[idx+1].centroid

                color = utils_general.get_color(lymph_series[idx], color_by = color_by, vmin = vmin, vmax = vmax)

                ax.plot([x_center_0, x_center_1], [y_center_0, y_center_1], [z_center_0, z_center_1], c = color)
                ax.grid(False)
        utils_plotting.set_limits_3D(*fig.axes)
        utils_plotting.label_axes_3D(*fig.axes)
        utils_plotting.equal_axes_notSquare_3D(*fig.axes)
        plt.show()


    def _set_speeds(self):
        self._set_centroids()

        for lymph_series in self.lymph_serieses:
            frames = [lypmh.frame for lypmh in lymph_series]
            dict = {}
            for frame, lymph in zip(frames, lymph_series):
                dict[frame] = lymph
            for lymph in lymph_series:
                frame = lymph.frame
                if frame-2 in frames and frame-1 in frames and frame+1 in frames and frame+2 in frames:
                     speeds = []
                     vectors = []
                     for frame_ in [frame-1, frame, frame+1, frame+2]:

                         (x_center_A, y_center_A, z_center_A) = dict[frame_].centroid
                         (x_center_B, y_center_B, z_center_B) = dict[frame_-1].centroid

                         speed = np.sqrt((x_center_A-x_center_B)**2 + (y_center_A-y_center_B)**2 + (z_center_A-z_center_B)**2)
                         speeds.append(speed)

                         vector = np.array([x_center_A-x_center_B, y_center_A-y_center_B, z_center_A-z_center_B])
                         vectors.append(vector)

                     lymph.speed = np.mean(speeds)
                     angles = [np.arccos(np.dot(vectors[idx], vectors[idx+1])/(np.linalg.norm(vectors[idx])*np.linalg.norm(vectors[idx+1]))) for idx in range(len(vectors)-1)]
                     lymph.angle = np.mean(angles)


    def correlate_shape_with_speedAngle(self, max_l, n_components, pca = False):
        """
        Correlates a representation with speed and angle.
        Args:
        - max_l: truncation degree.
        - n_components: number of representation components to correlate with.
        - pca: whether to correlate with dimensioally-reduced (via PCA) representation.
        """

        self.set_speedsAndTurnings()

        self.lymph_serieses = del_whereNone(self.lymph_serieses, 'coeff_array')
        self.lymph_serieses = del_whereNone(self.lymph_serieses, 'speed')

        if pca:
            pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = n_components, max_l = max_l)
        else:
            lowDimRepSplit = []
            for idx_series in range(self.num_serieses):
                split = []
                for lymph in self.lymph_serieses[idx_series]:
                    split.append(lymph.set_RIvector(max_l))
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
