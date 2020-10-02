
class PCA_Methods:
    """
    Inherited by Lymph_Seriese class.
    Contains methods that involve PCA.
    """

    def _get_pca_objs(self, n_components, max_l, removeSpeedNone = False, removeAngleNone = False, permAlterSeries = False):
        """

        Args:
        - n_components: dimensionailty of low dimensional representation.
        - max_l: truncation of original representation.
        - removeSpeedNone: whether to remove frames with speed None.
        - removeAngleNone: whether to remove frames with angle None.
        - permAlterSeries: whether to permantly delete these frames.
        Returns:
        - pca_obj: object returned by PCA(n_components = n_components).
        - max_l: max_l of original representation.
        - lowDimRepTogeth: all representations together in one array, shape (num_frames, rep_size).
        - lowDimRepSplit: list of series lists with separate low-dimensional representations.
        """


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
                vector = lymph.SH_set_rotInv_vector(max_l)
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


    def pca_plot_sampling(self, max_l, num_samples, color_param = None,  std = False):

        pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = 2, max_l = max_l)
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



    def pca_plot_shape_trajectories(self, max_l, colorBy = 'time'):

        if colorBy == 'speed' or colorBy == 'angle':
            self.set_speedsAndTurnings()

        if colorBy == 'speed':
            pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = 2, max_l = max_l, removeSpeedNone = True, permAlterSeries = True)
        elif colorBy == 'angle':
            pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = 2, max_l = max_l, removeAngleNone = True, permAlterSeries = True)
        elif colorBy ==  'time':
            pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = 2, max_l = max_l)
        else:
            pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = 2, max_l = max_l, removeSpeedNone = False, permAlterSeries = True)

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
        num_cols = (self.num_serieses // 3) + 1
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





    def plot_pca_recons(self, n_pca_components, max_l, plot_every):

        pca_obj, max_l, lowDimRep = self.get_pca_objs(n_pca_components, max_l)
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
