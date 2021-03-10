

class Centroid_Variable_Methods:

    def plot_cofms(self, colorBy = 'speed'):
        """
        Plots the centre of mass trajectories of each cell.
        Args:
        - colorBy: attribute to color by ("speed" / "angle").
        """

        self.set_speedsAndTurnings()

        cmap = plt.cm.viridis

        fig = plt.figure()

        if colorBy == 'speed':
            speeds = [lymph.speed for sublist in self.lymph_serieses for lymph in sublist if lymph.speed is not None and lymph.coeff_array is not None]
            vmin, vmax = min(speeds), max(speeds)
        elif colorBy == 'angle':
            angles = [lymph.angle for sublist in self.lymph_serieses for lymph in sublist if lymph.angle is not None and lymph.coeff_array is not None]
            vmin, vmax = min(angles), max(angles)
        norm = plt.Normalize(vmin, vmax)


        for idx_ax, lymph_series in enumerate(self.lymph_serieses):
            ax = fig.add_subplot(2, (self.num_serieses//2)+2, idx_ax+1, projection = '3d')

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
                ax.grid(False)


        ax = fig.add_subplot(2, (self.num_serieses//2)+2,self.num_serieses+2, projection = '3d')
        voxels = read_niigz(self.lymph_serieses[0][0].niigz)
        ax.voxels(voxels, edgecolors = 'white')

        equal_axes_notSquare(*fig.axes)
        plt.show()





    def set_speeds(self):

        for series in self.lymph_serieses:

            idxs = [lypmh.idx for lypmh in series]
            dict = {}
            for idx, lymph in zip(idxs, series):
                dict[idx] = lymph
            for lymph in series:

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



    def set_angles(self):

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
                    split.append(lymph.set_rotInv_vector(max_l))
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
