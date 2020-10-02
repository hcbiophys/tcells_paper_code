

class Single_Cell_Methods:
    """
    Inherited by Lymph_Snap class.
    Contains methods for series of a single cell.
    """

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


    def plot_series_niigz(self, plot_every):

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


    def plot_recon_series(self, max_l, plot_every, color_param = 'phis'):

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




        def plot_rotInvRep_series_bars(self, maxl = 5, plot_every = 1, means_adjusted = False):

            pca_obj, max_l, lowDimRepTogeth, lowDimRepSplit = self.get_pca_objs(n_components = 2, max_l = maxl)
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
