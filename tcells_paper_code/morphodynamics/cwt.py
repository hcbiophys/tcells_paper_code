import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import pywt
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from sympy import *
import pyvista as pv
pv.set_plot_theme("document")
import matplotlib.gridspec as gridspec
import copy
import cv2
import itertools

import tcells_paper_code.utils.utils_cwt as utils_cwt
import tcells_paper_code.utils.general as utils_general
from tcells_paper_code.dataloader.all import stack_attributes_all
from tcells_paper_code.videos.videos_class import Videos



class CWT():
    """
    Class for the continuous wavelet transform (CWT) for analysing T cell behaviour (i.e. the organisation of local morphodynamics)
    """

    def __init__(self, filename, load_or_save_or_run, idx_cfs= 'all', min_length = 15, chop = 15):
        """
        Args:
        - filename: the filename to load or save. Note: filename = 'None' means t-SNE isn't computed (used when just visualising the series or spectrograms).
        Options for loading are (as stored in */data/cwt_saved*): *150* (all PCs), *150_PC1_run* (marginal PC1 dynamics for run mode),
        *150_PC2_run* (marginal PC2 dynamics for run mode), *150_PC3_run* (marginal PC3 dynamics for run mode).
        - load_or_save_or_run: whether to load pre-saved data, save new ones, or simply run without saving.
        - idx_cfs: cfs stands for 'continuous frame section'. This is essentially the index of the cell, but letters are appended for each continuous time series section
        (since a few have big gaps in the time series). For example if cell 'CELL1' has two continuous sections with a gap, these have indices 'CELL1a' and 'CELL1b'.
        If the time series has no gaps, it will be 'CELL1a'
        - min_length: minimum time series length
        - chop: how much to chop off each side of the time series (determined by the wavelet region of influence) to mitigate edge effects
        """

        self.filename = filename
        self.load_or_save_or_run = load_or_save_or_run
        self.gaus1_scales = [0.4*i for i in np.linspace(2, 27, 6)]
        #self.gaus1_scales = [0.4*i for i in np.linspace(2, 22, 6)] # ultimate representation for PC2 morphodynamics uses this slightly reduced maximum wavelet width

        self.mexh_scales = [0.5*i for i in np.linspace(2, 12, 6)]
        self.chop = chop
        scales_per_wave = len(self.mexh_scales)
        self.inserts = [scales_per_wave+(scales_per_wave+1)*i for i in range(scales_per_wave)]
        self.time_either_side = 75
        self.min_length = 15


        if self.filename[-2:] == 'run':
            self.all_consecutive_frames = pickle.load(open('../data/time_series/shape_series_go.pickle',"rb"))
        elif self.filename[-4:] == 'stop':
            self.all_consecutive_frames = pickle.load(open('../data/time_series/shape_series_stop.pickle',"rb"))
        else:
            self.all_consecutive_frames = pickle.load(open('../data/time_series/shape_series.pickle',"rb"))

        if not idx_cfs== 'all':
            self.all_consecutive_frames = [i for i in self.all_consecutive_frames if i.name == idx_cfs]

        self.all_consecutive_frames = [j for i,j in enumerate(self.all_consecutive_frames) if len(j.pca0_list) > min_length]
        PC_uncertainties = pickle.load(open('../data/PC_uncertainties.pickle', 'rb'))


        idx_cells = ['CELL1', 'CELL2', 'CELL3', 'CELL4', 'CELL5', 'CELL6', 'CELL7', 'CELL8', 'CELL9', 'CELL11', 'CELL12', 'CELL13', 'CELL14', 'CELL15', 'CELL16', 'CELL17', 'CELL18', 'CELL19', 'CELL20', 'CELL21', 'CELL22', 'CELL23', 'CELL24', 'CELL25', 'CELL26', 'CELL27',  'CELL29', 'CELL30', 'CELL31']
        colors = [np.array([0.67739055, 0.82755935, 0.77116142]), np.array([0.0952304 , 0.09013385, 0.98938936]), np.array([0.05764147, 0.98641696, 0.75908016]), np.array([0.97425911, 0.48333032, 0.17135435]), np.array([0.43114909, 0.2235878 , 0.8842425 ]), np.array([0.32933019, 0.5921141 , 0.61633489]), np.array([0.07315546, 0.44819796, 0.16833376]), np.array([0.01532791, 0.73857975, 0.69280004]), np.array([0.67843096, 0.6826372 , 0.08518478]), np.array([0.08110285, 0.79746762, 0.908427  ]), np.array([0.30928829, 0.32599009, 0.42407218]), np.array([0.60985161, 0.36160205, 0.35521415]), np.array([0.47062361, 0.25963724, 0.91398498]), np.array([0.00744883, 0.07700202, 0.16986398]), np.array([0.87592732, 0.75720082, 0.17710782]), np.array([0.59714551, 0.40399573, 0.12145515]), np.array([0.26211748, 0.57891925, 0.28847181]), np.array([0.47409021, 0.04009612, 0.37440976]), np.array([0.01394242, 0.40145539, 0.70053317]), np.array([0.28150027, 0.31116461, 0.84870038]), np.array([0.10455617, 0.91580071, 0.53926957]), np.array([0.79352826, 0.12960295, 0.81574088]), np.array([0.46107105, 0.02359315, 0.45115123]), np.array([0.87501311, 0.29718405, 0.75983003]), np.array([0.54075337, 0.33526137, 0.71694272]), np.array([0.75402239, 0.83224114, 0.72639337]), np.array([0.30155334, 0.83126122, 0.14805019]), np.array([0.99656294, 0.70101507, 0.83437361]), np.array([0.99656294, 0.70101507, 0])]
        self.colors_dict = {i:j for i,j in zip(idx_cells, colors)}

        for cfs in self.all_consecutive_frames:
            cfs.PC_uncertainties = PC_uncertainties[cfs.name[:-1]]
            cfs.names_list = [cfs.name + '-' + str(i) for i in cfs.closest_frames]
            cfs.color = self.colors_dict[cfs.name[:-1]]

        self.spectrograms = None
        self.all_embeddings = None
        self.all_consecutive_frames_dict = {cfs.name: cfs for cfs in self.all_consecutive_frames}

        self._set_spectrograms()
        if filename is not 'None':
            print('here')
            self._set_tsne_embeddings()
            self.set_kde(plot = False)

    def _names_to_colors(self, names):
        """
        Get the colour given a cell's code
        """
        names = [i.split('-')[0][:-1] for i in names]
        colors = []
        for name in names:
            colors.append(self.colors_dict[name])
        return colors


    def plot_series(self):
        """
        Plot a few of the longer time series (PCs and retraction speeds)
        """

        fig = plt.figure()
        count = 0
        for cfs in self.all_consecutive_frames:
            if len(cfs.pca0_list) > 100 and count < 4:
                print(cfs.name)
                ax = fig.add_subplot(4, 1, count+1)
                for var_list, color in zip([cfs.pca0_list, cfs.pca1_list, cfs.pca2_list, cfs.speed_uropod_list], ['red', 'blue', 'green', 'black']):
                    var_list = utils_cwt._interpolate_list(var_list)
                    if color == 'black':
                        var_list = [i*100 for i in var_list] # scale for visualisation
                    ax.plot([i*5 for i,j in enumerate(var_list)], var_list, c = color)
                    ax.set_title(cfs.name)
                    #ax.set_ylim([-1, 1])
                count += 1

        plt.show()
        plt.subplots_adjust(hspace = 0)
        sys.exit()



    def COI(self):
        """
        Visualise the cone of influence to see where edge effects begin
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        fake_series = [100 for i in range(40)]
        coef, _ = pywt.cwt(fake_series, mexh_scales, 'mexh')

        ax1.imshow(coef)
        ax1.set_title('mexh')
        coef, _ = pywt.cwt(fake_series, self.gaus1_scales, 'gaus1')
        ax2.imshow(coef)
        ax2.set_title('gaus1')
        plt.show()
        sys.exit()


    def plot_wavelets(self, wavelet, scales):
        """
        Visualise the wavelets
        """

        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        for idx_scale, scale in enumerate(scales):
            t_ = Symbol('t_')
            if wavelet[:4] == 'mexh':
                func = (1-(((t_)/scale)**2))*exp(-0.5*(((t_)/scale)**2))
            elif wavelet[:4] == 'gaus':
                func = exp(-(t_**2))
                for _ in range(int(wavelet[4])):
                    func = func.diff(t_)
                func = func.subs(t_, (t_)/scale)


            p_real = plot(re(func), (t_, -40, 40), show = False, xlim = [-40, 40], ylim = [-2, 7])
            p_im = plot(im(func), (t_, -40, 40), show = False, xlim = [-40, 40], ylim = [-2, 7])
            p_real[0].line_color = 'blue'
            p_im[0].line_color = 'red'
            utils_cwt.move_sympyplot_to_axes(p_real, ax)
            utils_cwt.move_sympyplot_to_axes(p_im, ax)

        plt.show()


    def _set_spectrograms(self):
        """
        Set the spectrogram attribute
        """

        features = ['pca0_list', 'pca1_list', 'pca2_list']

        for consecutive_frames in self.all_consecutive_frames:
            spectrogram = []
            for idx_attribute, attribute in enumerate(features):

                coef, _ = pywt.cwt(getattr(consecutive_frames, attribute), self.mexh_scales, 'mexh')
                spectrogram.append(coef)
                coef, _ = pywt.cwt(getattr(consecutive_frames, attribute), self.gaus1_scales, 'gaus1')
                spectrogram.append(coef)

            spectrogram = np.concatenate(spectrogram, axis = 0)

            if self.chop is not None:
                spectrogram = spectrogram[:, self.chop :-self.chop]

                for attribute in ['closest_frames', 'pca0_list', 'pca1_list', 'pca2_list', 'speed_uropod_list', 'speed_uropod_running_mean_list', 'names_list']:
                    setattr(consecutive_frames, attribute, getattr(consecutive_frames, attribute)[self.chop :-self.chop])

                #consecutive_frames.closest_frames = consecutive_frames.closest_frames[self.chop :-self.chop]
                #consecutive_frames.pca0_list = consecutive_frames.pca0_list[self.chop :-self.chop]
                #consecutive_frames.pca1_list = consecutive_frames.pca1_list[self.chop :-self.chop]
                #consecutive_frames.pca2_list = consecutive_frames.pca2_list[self.chop :-self.chop]
                #consecutive_frames.speed_uropod_list = consecutive_frames.speed_uropod_list[self.chop :-self.chop]
                #consecutive_frames.speed_uropod_running_mean_list = consecutive_frames.speed_uropod_running_mean_list[self.chop :-self.chop]
                #consecutive_frames.names_list = consecutive_frames.names_list[self.chop :-self.chop]

            consecutive_frames.spectrogram = spectrogram

    def plot_spectrograms(self):
        """
        Visualise the spectrograms
        """
        for cfs in self.all_consecutive_frames:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(cfs.spectrogram, vmin = -0.25, vmax = 0.25)
            for ins in self.inserts:
                cfs.spectrogram = np.insert(cfs.spectrogram, ins, np.zeros(shape = (cfs.spectrogram.shape[1],)), 0)
            ax.imshow(cfs.spectrogram, cmap = 'PiYG', vmin = -0.25, vmax = 0.25)
            ax.axis('off')
            plt.show()


    def _set_tsne_embeddings(self):
        """
        Set the t-SNE embeddings
        """

        if self.load_or_save_or_run == 'load':

            data = pickle.load(open('../data/cwt_saved/{}_dots.pickle'.format(self.filename), 'rb'))
            xs, ys, names = data['xs'], data['ys'], data['names']
            self.all_embeddings = np.array(list(zip(xs, ys)))


        elif self.load_or_save_or_run == 'save' or self.load_or_save_or_run == 'run':
            concat = np.concatenate([i.spectrogram for i in self.all_consecutive_frames], axis = 1).T

            per_PC = int(concat.shape[1]/3)
            if '_' in self.filename:
                if self.filename.split('_')[1] == 'PC1':
                    concat = concat[:, :per_PC]
                elif self.filename.split('_')[1] == 'PC2':
                    concat = concat[:, per_PC:2*per_PC]
                elif '_' in self.filename and self.filename.split('_')[1] == 'PC3':
                    concat = concat[:, 2*per_PC:]


            self.all_embeddings = TSNE(n_components=2).fit_transform(concat)


        idxs_cells = np.cumsum([0] + [i.spectrogram.shape[1] for i in self.all_consecutive_frames])
        for idx, consecutive_frame in enumerate(self.all_consecutive_frames):
            consecutive_frame.embeddings = self.all_embeddings[idxs_cells[idx]:idxs_cells[idx+1], :]


    def _set_embedding_colors(self, xs, ys):
        """
        Set the colours of the t-SNE embeddings
        """


        colors_pc = []
        colors_speed_uropod = []
        colors_mode = []
        for cfs in self.all_consecutive_frames:

            colors_speed_uropod += list(cfs.speed_uropod_list)
            colors_pc += list(cfs.pca0_list)

            new_colors_mode = []
            for i in cfs.speed_uropod_running_mean_list:
                if i > 0.005:
                    new_colors_mode.append('red')
                elif i < 0.002:
                    new_colors_mode.append('blue')
                else:
                    new_colors_mode.append('grey')
            colors_mode += new_colors_mode

        colors_2d = colors_speed_uropod
        colors_3d = None


        return xs, ys, colors_2d, colors_3d

    def _plot_path_of(self, xs, ys, names, path_of, num_per_section = 40):
        """
        Plot the path of a cell through the morphodynamic space
        Args:
        - xs, ys: coordinates of the trajectory
        - names: names of each point on the trajectory (each name is the cell index and frame number)
        - path_of: index of the cell to plot
        - num_per_section: number of points to show before generating a new figure (useful if it's a really long timeseries)
        """

        cfs_xs, cfs_ys = [], []
        closest_frames = self.all_consecutive_frames_dict[path_of].closest_frames

        for idx in range(len(xs)):
            print(names[idx].split('-')[0], path_of)
            if names[idx].split('-')[0] == path_of:

                cfs_xs.append(xs[idx])
                cfs_ys.append(ys[idx])
        plt.plot(cfs_xs, cfs_ys, '-o', zorder = 0, color = self.all_consecutive_frames_dict[path_of].color)

        contours = []
        for i in np.linspace(np.min(self.all_embeddings[:, 0])-10, np.max(self.all_embeddings[:, 0])+10, 6)[1:-1]:
            for j in np.linspace(np.min(self.all_embeddings[:, 1])-10, np.max(self.all_embeddings[:, 1])+10, 6)[1:-1]:
                contours.append(np.array([i, j]))

        #plt.scatter([c[0] for c in contours], [c[1] for c in contours], s =50, c = 'black', zorder = 1)

        for idx in range(len(cfs_xs)):
            if idx % 5 == 0:
                plt.text(cfs_xs[idx], cfs_ys[idx], str(self.chop*5 + 5*idx), fontsize = 12)


        plt.xlim(np.min(self.all_embeddings[:, 0])-10, np.max(self.all_embeddings[:, 0])+10)
        plt.ylim(np.min(self.all_embeddings[:, 1])-10, np.max(self.all_embeddings[:, 1])+10)



    def plot_embeddings(self, path_of = None):
        """
        Plot the embeddings in the morphodynamic space
        Args:
        - path_of = None shows all embeddings, coloured by cell. Changing this argument to a cell index (e.g. 'CELL21a') plots the trajectory of that section
        """

        if self.load_or_save_or_run == 'load':
            data = pickle.load(open('../data/cwt_saved/{}_dots.pickle'.format(self.filename), 'rb'))
            xs = data['xs']
            ys = data['ys']
            names = data['names']


        elif self.load_or_save_or_run == 'save' or self.load_or_save_or_run == 'run':
            data = {}
            xs, ys, zs, names = [], [],  [], []
            for consecutive_frames in self.all_consecutive_frames:
                xs += list(consecutive_frames.embeddings[:, 0])
                ys += list(consecutive_frames.embeddings[:, 1])
                names +=  list(consecutive_frames.names_list)

            data['xs'] = xs
            data['ys'] = ys
            data['names'] = names
            if self.load_or_save_or_run == 'save':
                pickle.dump(data, open('../data/cwt_saved/{}_dots.pickle'.format(self.filename), 'wb'))

        xs, ys, colors_2d, colors_3d = self._set_embedding_colors(xs, ys)
        colors_2d = self._names_to_colors(names)


        if path_of is not None:
            self._plot_path_of(xs, ys, names, path_of, num_per_section = 200)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            sc = ax.scatter(xs, ys, c = colors_2d, vmax = 0.015, cmap = 'Blues')


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

        plt.xlim(np.min(self.all_embeddings[:, 0])-10, np.max(self.all_embeddings[:, 0])+10)
        plt.ylim(np.min(self.all_embeddings[:, 1])-10, np.max(self.all_embeddings[:, 1])+10)
        plt.show()



    def set_kde(self, plot = True):
        """
        Kernel density estimate to get the probability density function (PDF) over embeddings in the morphodynamic space (to see which regions correspond to stereotyped behaviours)
        """

        if self.load_or_save_or_run == 'load':
            pdf_array = pickle.load(open('../data/cwt_saved/{}_kde.pickle'.format(self.filename), 'rb'))
        elif self.load_or_save_or_run == 'save' or self.load_or_save_or_run == 'run':
            xs = np.linspace(np.min(self.all_embeddings[:, 0])-10, np.max(self.all_embeddings[:, 0])+10, 50)
            ys = np.linspace(np.min(self.all_embeddings[:, 1])-10, np.max(self.all_embeddings[:, 1])+10, 50)

            xx, yy = np.meshgrid(xs, ys)
            positions = np.vstack([xx.ravel(), yy.ravel()]).T

            kernel = KernelDensity(bandwidth = 5)
            kernel.fit(self.all_embeddings)


            pdf_array = np.exp(kernel.score_samples(positions))
            pdf_array = np.reshape(pdf_array, xx.shape)
            if self.load_or_save_or_run == 'save':
                pickle.dump(pdf_array, open('../data/cwt_saved/{}_kde.pickle'.format(self.filename), 'wb'))

        self.pdf_array = pdf_array
        if plot:
            plt.imshow(self.pdf_array[::-1, :], cmap = 'Reds')
            plt.show()
        return pdf_array


    def _get_contours(self, pdf, s, b):
        """
        Get the contours around the stereotyped behaviours in the morphodynamic space. s & b are the thresholding parameters
        """

        if s is None and b is None:
            plt.imshow(pdf[::-1, :])
            plt.show()
            s_list = [ 15, 17, 19, 21, 23, 25] # ROWS
            b_list = [ 35, 40, 45, 50, 55] # COLUMNS
            b_list = [b*(np.nanmax(pdf)/255) for b in b_list]

            fig_kde = plt.figure()
            for idx_s, s in enumerate(s_list):
                for idx_b, b in enumerate(b_list):

                    ax = fig_kde.add_subplot(6, 5, idx_s*len(b_list) + idx_b + 1)
                    pdf_new = np.zeros_like(pdf)

                    for row in range(pdf.shape[0]):
                        for col in range(pdf.shape[1]):
                            surrounding = pdf[int(row-(s/2)):int(row+(s/2)), int(col-(s/2)):int(col+(s/2))]
                            if pdf[row, col] > np.nanmean(surrounding) + b:
                                pdf_new[row, col] = 255
                            else:
                                pdf_new[row, col] = 0
                    contours, hierarchy = cv2.findContours(pdf_new.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    ax.imshow(pdf[::-1, :], cmap = 'Reds')
                    contours = self._clean_contours(contours)


                    for contour in contours:
                        ax.plot(contour[:, 0].flatten(), 49-contour[:,1].flatten(), c = 'black', linewidth = 3)

                    ax.set_xlabel('b')
                    ax.set_ylabel('s')
            plt.show()
            sys.exit()



        pdf_borders = np.zeros_like(pdf)

        b *= (np.nanmax(pdf)/255)
        for row in range(pdf.shape[0]):
            for col in range(pdf.shape[1]):
                surrounding = pdf[int(row-(s/2)):int(row+(s/2)), int(col-(s/2)):int(col+(s/2))]
                if pdf[row, col] > np.nanmean(surrounding) + b:
                    pdf_borders[row, col] = 255
                else:
                    pdf_borders[row, col] = 0

        contours, hierarchy = cv2.findContours(pdf_borders.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = self._clean_contours(contours)



        return contours


    def _clean_contours(self, contours):
        """
        Some internal processing to make the contour objects simpler to manipulate
        """

        contours = [np.squeeze(i) for i in contours]
        contours_new = []
        for idx, i in enumerate(contours):
            if len(i.shape) == 1:
                i = np.expand_dims(i, axis=0)
            i = np.vstack([i, i[0, :]])
            contours_new.append(i)

        contours_new = [c for c in contours_new if cv2.contourArea(c) > 5]

        return contours_new



    def transition_matrix(self, grid, plot = True):
        """
        Generate the transition probability matrix for transitions between stereotyped behaviours
        """

        pdf = copy.deepcopy(self.pdf_array)

        pdf[self.pdf_array<3e-5] = np.nan

        if plot:
            fig_both = plt.figure()
            ax = fig_both.add_subplot(1, 3, 1)
            ax.imshow(pdf[::-1, :], cmap = 'Reds')

        if grid:
            contours = []
            for i in np.linspace(0, 50, 6)[1:-1]:
                for j in np.linspace(0, 50, 6)[1:-1]:
                    contours.append(np.array([[i, j]]))
        else:
            thresh_params_dict = { '150': (7, 20), '150_run':(17, 35)}
            contours = self._get_contours(pdf, s = thresh_params_dict[filename][0], b = thresh_params_dict[filename][1])
            colors = ['black', 'magenta', 'darkgreen', 'red', 'blue', 'orange', 'blueviolet', 'grey', 'aqua', 'maroon', 'lime']
            for idx, contour in enumerate(contours):
                #ax.plot(contour[:, 0].flatten(), 49-contour[:,1].flatten(), c = 'black', linewidth = 3)
                if plot:
                    ax.plot(contour[:, 0].flatten(), 49-contour[:,1].flatten(), c = colors[idx], linewidth = 3)

        sequences = []
        for cfs in self.all_consecutive_frames:
            xs = list(cfs.embeddings[:, 0])
            ys = list(cfs.embeddings[:, 1])

            sequence = utils_cwt.get_idx_contours(contours, list(self.all_embeddings[:, 0]), list(self.all_embeddings[:, 1]), xs, ys)
            sequences.append(sequence)

        T1 = utils_cwt.get_tm(contours, sequences)
        _ = utils_cwt.entropy(T1)

        sequences_no_duplicates = []
        for sequence in sequences:
            sequences_no_duplicates.append([key for key, grp in itertools.groupby(sequence)])
        T2 = utils_cwt.get_tm(contours, sequences_no_duplicates)

        entropy_save = utils_cwt.entropy(T2)
        print('entropy_save', entropy_save)

        if plot:
            ax2 = fig_both.add_subplot(1, 3, 2)
            ax3 = fig_both.add_subplot(1, 3, 2)
            ax2.imshow(T1, cmap = 'Blues')
            ax3.imshow(T2, cmap = 'Blues')
            plt.show()


    def bar_entropy(self):
        """
        Plot the bar chart comparing entropy of the marginal dynamics
        """
        all = [1.54, 1.36, 1.45, 1.26, 1.35]
        pc1 = [1.17, 1.06, 1.15, 1.09, 1.19]
        pc2 = [0.86, 0.92, 1.04, 0.9, 0.93]
        pc3 = [1.06, 1.16, 0.96, 0.97, 1.06]

        plt.bar(range(4),  [np.mean(i) for i in [all, pc1, pc2, pc3]], color = 'red', zorder = 0)
        plt.scatter([0 for i in all], all, c = 'black', s = 2, zorder = 1)
        plt.scatter([1 for i in pc1], pc1, c = 'black', s = 2, zorder = 1)
        plt.scatter([2 for i in pc2], pc2, c = 'black', s = 2, zorder = 1)
        plt.scatter([3 for i in pc3], pc3, c = 'black', s = 2, zorder = 1)
        plt.ylim([0.6, 1.5])
        plt.xticks([])
        plt.show()




def show_cell_series_clustered(codes):
    """
    Show the cell surfaces of
    Args:
    - codes: is a list of codes, where each code is a cell index and frame, e.g. ['CELL21a-72']
    To get the codes, first run the 'plot_embeddings' function, and then you can hover over embeddings to get their indices
    """
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    cwt = CWT(filename = 'None', load_or_save_or_run = None, chop = None)

    all_pc0_series = []
    all_pc1_series = []
    all_pc2_series = []

    for code in codes:
        idx_cfs= code.split('-')[0]
        center_frame = int(code.split('-')[1])

        idx_cell, letter_keep = idx_cfs[:-1], idx_cfs[-1]
        cells = Videos(stack_attributes_all, cells_model = [idx_cell], uropods_bool = True)
        video = cells.cells[idx_cell]
        lymph_t_res = video[0].t_res

        keep = []

        for frame in video:
            if  abs((frame.idx_frame-center_frame)*lymph_t_res) < cwt.time_either_side:
                keep.append(frame)

        cells.cells[idx_cell] = keep
        plot_every = int(len(keep)/4)
        cells.plot_orig_series(idx_cell=idx_cell, uropod_align = False, color_by = None, plot_every = plot_every, plot_flat = True)

        cfs = [i for i in cwt.all_consecutive_frames if i.name == idx_cfs][0]

        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2)
        ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        idxs_plot = [i for i,j in enumerate(cfs.closest_frames) if abs((j-center_frame)*lymph_t_res) < cwt.time_either_side]

        pc0s = [j for i,j in enumerate(cfs.pca0_list) if i in idxs_plot]
        pc1s = [j for i,j in enumerate(cfs.pca1_list) if i in idxs_plot]
        pc2s = [j for i,j in enumerate(cfs.pca2_list) if i in idxs_plot]


        ax1.plot([5*i for i in list(range(len(pc0s)))], pc0s, color = 'red')
        ax1.plot([5*i for i in list(range(len(pc1s)))], pc1s, color = 'blue')
        ax1.plot([5*i for i in list(range(len(pc2s)))], pc2s, color = 'green')
        ax1.set_ylim([-1, 1])

        # show spectrogram
        spect = cfs.spectrogram[:, idxs_plot]
        print(center_frame)
        print(cfs.closest_frames)

        vert = cfs.spectrogram[:, cfs.closest_frames.index(center_frame)][:, None]
        # CHANGED
        for ins in cwt.inserts:
            empty = np.empty(shape = (len(idxs_plot),))
            spect = np.insert(spect, ins, empty.fill(np.nan), 0)
            empty = np.empty(shape = (1,))
            vert = np.insert(vert, ins, empty.fill(np.nan), 0)
        vert = np.vstack([vert.T]*4)


        ax2.imshow(spect, cmap = 'PiYG', vmin = -0.25, vmax = 0.25)
        ax3.imshow(vert, cmap = 'PiYG', vmin = -0.25, vmax = 0.25)

        plt.show()
