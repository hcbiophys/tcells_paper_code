import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import sys
import pickle
from scipy import signal
import pywt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sympy import *
import pyvista as pv
pv.set_plot_theme("document")
import random
import statsmodels.api as sm
import matplotlib.gridspec as gridspec
from statsmodels.tsa.api import VAR
import copy
from mpl_toolkits.mplot3d import Axes3D


from lymphocytes.data.dataloader_good_segs_2 import stack_attributes_2
from lymphocytes.data.dataloader_good_segs_3 import stack_attributes_3
from lymphocytes.cells.cells_class import Cells



def move_sympyplot_to_axes(p, ax):
    backend = p.backend(p)
    backend.ax = ax
    # Fix for > sympy v1.5
    backend._process_series(backend.parent._series, ax, backend.parent)
    backend.ax.spines['right'].set_color('none')
    backend.ax.spines['bottom'].set_position('zero')
    backend.ax.spines['top'].set_color('none')
    plt.close(backend.fig)


morB, morC = 10, 0.3
class CWT():

    def __init__(self, idx_cells = 'all', min_length = 15, chop = 5):
        self.chop = chop


        self.all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series.pickle',"rb"))
        self.all_consecutive_frames_dict = {}
        for i in self.all_consecutive_frames:
            self.all_consecutive_frames_dict[i.name] = i


        def list_all(idx_cell):
            all = []
            for cfs in self.all_consecutive_frames:
                print(cfs.name[:-1])
                if cfs.name[:-1] == idx_cell:
                    all.append(cfs)
            return all




        if not idx_cells == 'all':
            self.all_consecutive_frames = [i for i in self.all_consecutive_frames if i.name[:-1] in idx_cells]


        """
        acorrs = [[], [], [], []]
        for i in self.all_consecutive_frames:
            for idx, l in enumerate([i.pca0_list, i.pca1_list, i.pca2_list, i.run_list]):
                acf = list(sm.tsa.acf(l, nlags = 99, missing = 'conservative'))
                acf += [np.nan for _ in range(100-len(acf))]
                acorrs[idx].append(np.array(acf))

        for acorr, color in zip(acorrs, ['red', 'green', 'blue', 'pink']):
            acorr = np.array([acorr])
            concat = np.concatenate(acorr, axis = 0)
            m = np.nanmean(abs(concat), axis = 0)
            plt.plot(m, c = color)
            plt.plot(list(range(len(m))), [0 for i in m], c = 'black')
        plt.show()
        sys.exit()
        """


        idxs_keep = [i for i,j in enumerate(self.all_consecutive_frames) if len(j.pca0_list) > min_length]
        self.all_consecutive_frames = [j for i,j in enumerate(self.all_consecutive_frames) if i in idxs_keep]



        self.spectograms = None
        self.num_features = None

        self.all_embeddings = None

        """
        N = 5
        for cfs in self.all_consecutive_frames:
            plt.plot(cfs.run_list, c = 'blue')
            b = np.convolve(cfs.run_list, np.ones(N)/N, mode='valid')
            plt.plot(b, c = 'red')
            plt.show()
        """





    def print_freqs(self):
        """
        Print the frequencies corresponding to different scales (this varies depending on the wavelet)
        """
        freqs = [pywt.scale2frequency(wavelet=wavelet, scale = scale) for scale in self.scales]
        print('scales: {}'.format(self.scales))
        print('freqs: {}'.format(freqs))

    def edge_effect_size(self, wavelet, scales):

        fake_series = [1 for i in range(40)]

        coef, freqs = pywt.cwt(fake_series, scales, wavelet)
        #coef = abs(coef)

        plt.imshow(coef)
        plt.show()
        sys.exit()


    def _plot_wavelets(self, wavelet, scales, frames_plot = 'all'):

        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        for idx_scale, scale in enumerate(scales):

            print('idx_scale', idx_scale)
            t_ = Symbol('t_')
            """
            if self.wavelet[:4] == 'cmor':
                func = (1/sqrt(morB*pi))*exp((-((t_-5)/scale)**2)/morB)*exp(2*pi*1.j*morC*((t_-5)/scale))
                func +=  (scale-min(self.scales)) + (scale-min(self.scales))*I
            elif self.wavelet[:4] == 'cgau':
                func = exp(-1.j*((t_-5)/scale))*exp(-((t_-5)/scale)**2)  # t_-scale*2 for plotting each shifted horizontally
                for _ in range(int(self.wavelet[4])):
                    func = func.diff(t_)
                func +=  (scale-min(self.scales)) + (scale-min(self.scales))*I
            """
            if wavelet[:4] == 'mexh':
                func = (1-(((t_-5)/scale)**2))*exp(-0.5*(((t_-5)/scale)**2))
            elif wavelet[:4] == 'gaus':
                func = exp(-(t_**2))
                for _ in range(int(wavelet[4])):
                    func = func.diff(t_)
                func = func.subs(t_, (t_-5)/scale)

            if frames_plot == 'all':
                p_real = plot(re(func), show = False, xlim = [-10, 500], ylim = [-2, 7])
                p_im = plot(im(func), show = False, xlim = [-10, 500], ylim = [-2, 7])
            else:
                p_real = plot(re(func), (t_, -10, len(frames_plot.pca0_list)), show = False, ylim = [-2, 7])
                p_im = plot(im(func), (t_, -10, len(frames_plot.pca0_list)), show = False, ylim = [-2, 7])
            p_real[0].line_color = 'blue'
            p_im[0].line_color = 'red'
            move_sympyplot_to_axes(p_real, ax)
            move_sympyplot_to_axes(p_im, ax)


        ax = fig.add_subplot(2, 1, 2)
        if frames_plot == 'all':
            for i in self.all_consecutive_frames:
                ax.plot([j for j in range(len(i.pca0_list))], i.pca0_list, color = 'red')
                ax.plot([j for j in range(len(i.pca1_list))], i.pca1_list, color = 'blue')
                ax.plot([j for j in range(len(i.pca2_list))], i.pca2_list, color = 'green')
                ax.set_ylim([-1, 1])

        else:

            ax.plot([j for j in range(len(frames_plot.pca0_list))], frames_plot.pca0_list, color = 'red')
            ax.plot([j for j in range(len(frames_plot.pca1_list))], frames_plot.pca1_list, color = 'blue')
            ax.plot([j for j in range(len(frames_plot.pca2_list))], frames_plot.pca2_list, color = 'green')
            ax.set_ylim([-1, 1])
        plt.show()




    def set_spectograms(self):

        features = ['pca0_list', 'pca1_list', 'pca2_list']
        self.num_features = len(features)


        for consecutive_frames in self.all_consecutive_frames:
            spectogram = []
            for idx_attribute, attribute in enumerate(features):
                """
                coef, freqs = pywt.cwt(getattr(consecutive_frames, attribute), self.scales, self.wavelet)
                #coef = abs(coef)
                spectogram.append(coef)
                """

                # CHANGED
                mexh_scales = [0.5*2]
                gaus1_scales = [0.4*2]
                coef, _ = pywt.cwt(getattr(consecutive_frames, attribute), mexh_scales, 'mexh')
                spectogram.append(coef)
                coef, _ = pywt.cwt(getattr(consecutive_frames, attribute), gaus1_scales, 'gaus1')
                spectogram.append(coef)

            spectogram = np.concatenate(spectogram, axis = 0)

            if self.chop is not None:
                spectogram = spectogram[:, self.chop :-self.chop]
                # update other variables after chop
                consecutive_frames.closest_frames = consecutive_frames.closest_frames[self.chop :-self.chop]
                consecutive_frames.pca0_list = consecutive_frames.pca0_list[self.chop :-self.chop]
                consecutive_frames.pca1_list = consecutive_frames.pca1_list[self.chop :-self.chop]
                consecutive_frames.pca2_list = consecutive_frames.pca2_list[self.chop :-self.chop]
                consecutive_frames.delta_centroid_list = consecutive_frames.delta_centroid_list[self.chop :-self.chop]
                consecutive_frames.delta_sensing_direction_list = consecutive_frames.delta_sensing_direction_list[self.chop :-self.chop]
                consecutive_frames.run_list = consecutive_frames.run_list[self.chop :-self.chop]

            consecutive_frames.spectogram = spectogram

    def _plot_spectogram(self, spectogram):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # CHANGED
        ax.imshow(spectogram, vmin = -0.2, vmax = 0.2)

        # CHANGED
        for ins in [1, 3, 5, 7, 9]:
            spectogram = np.insert(spectogram, ins, np.zeros(shape = (spectogram.shape[1],)), 0)
        # CHANGED
        ax.imshow(spectogram, cmap = 'PiYG', vmin = -0.2, vmax = 0.2)
        ax.axis('off')


    def plot_wavelet_series_spectogram(self, name = 'all'):
        for cfs in self.all_consecutive_frames:
            if name == 'all' or cfs.name == name:
                # CHANGED
                mexh_scales = [0.5*2]
                gaus1_scales = [0.4*2]
                self._plot_wavelets(frames_plot = cfs, wavelet = 'mexh', scales = mexh_scales)
                self._plot_wavelets(frames_plot = cfs, wavelet = 'gaus1', scales = gaus1_scales)
                self._plot_spectogram(spectogram = cfs.spectogram)
                fig2 = plt.figure()
                ax = fig2.add_subplot(111)
                ax.plot([j for j in range(len(cfs.pca0_list))], cfs.pca0_list, color = 'red')
                ax.plot([j for j in range(len(cfs.pca1_list))], cfs.pca1_list, color = 'blue')
                ax.plot([j for j in range(len(cfs.pca2_list))], cfs.pca2_list, color = 'green')
                ax.set_ylim([-1, 1])

                plt.show()




    def set_tsne_embeddings(self):

        self.all_embeddings = TSNE(n_components=2).fit_transform(np.concatenate([i.spectogram for i in self.all_consecutive_frames], axis = 1).T)
        idxs_cells = np.cumsum([0] + [i.spectogram.shape[1] for i in self.all_consecutive_frames])
        for idx, consecutive_frame in enumerate(self.all_consecutive_frames):
            consecutive_frame.embeddings = self.all_embeddings[idxs_cells[idx]:idxs_cells[idx+1], :]

    def plot_embeddings(self, load_or_save = 'save', file_name = None, path_of = None):

        if load_or_save == 'load':
            data = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_dots.pickle'.format(file_name), 'rb'))
            colors = data['colors']
            xs = data['xs']
            ys = data['ys']
            names = data['names']
        elif load_or_save == 'save':
            data = {}
            xs, ys, colors, names = [], [], [], []
            for consecutive_frames in self.all_consecutive_frames:
                colors += [np.random.rand(3,)]*consecutive_frames.embeddings.shape[0]
                xs += list(consecutive_frames.embeddings[:, 0])
                ys += list(consecutive_frames.embeddings[:, 1])
                names += [consecutive_frames.name + '-' + str(i) for i in consecutive_frames.closest_frames]
            data['colors'] = colors
            data['xs'] = xs
            data['ys'] = ys
            data['names'] = names
            if path_of is None: #SAVE THE COORDS AS A DICTIONARY WITH NAMES AS KEYS ETC
                pickle.dump(data, open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_dots.pickle'.format(file_name), 'wb'))

        fig = plt.figure()
        ax = fig.add_subplot(111)



        #all = np.concatenate([i.spectogram for i in self.all_consecutive_frames], axis = 1)
        #colors = np.max(abs(all), axis = 0)


        colors = []
        for i in self.all_consecutive_frames:
            """
            plt.plot(i.run_list, c = 'blue')
            plt.plot(np.diff(i.run_list), c = 'red')
            plt.plot([0 for _ in i.run_list], c = 'black')
            plt.show()
            """
            colors += list(i.run_list)







        #idxs_keep = [i for i,j in enumerate(names) if j[:4] == '2_1a']
        #xs = [j for i,j in enumerate(xs) if i in idxs_keep]
        #ys = [j for i,j in enumerate(ys) if i in idxs_keep]
        #colors = [j for i,j in enumerate(colors) if i in idxs_keep]

        xs_copy, ys_copy, colors_copy =  copy.deepcopy(xs), copy.deepcopy(ys), copy.deepcopy(colors)
        zipped = list(zip(xs_copy, ys_copy, colors_copy))
        #random.shuffle(zipped)
        xs_copy, ys_copy, colors_copy = list(zip(*zipped))

        sc = ax.scatter(xs_copy, ys_copy,  c = colors_copy,   vmin = -0.01, vmax = 0.01, cmap = 'PiYG', edgecolors='b')




        if path_of is not None:
            cfs_xs, cfs_ys = [], []
            closest_frames = self.all_consecutive_frames_dict[path_of].closest_frames

            for idx in range(len(xs)):
                if names[idx].split('-')[0] == path_of:
                    cfs_xs.append(xs[idx])
                    cfs_ys.append(ys[idx])
            for idx_section in range(1+len(cfs_xs)//40):
                plt.scatter(xs, ys, c = colors, alpha = 0.2, edgecolors='none')
                section_xs = cfs_xs[40*idx_section:40*idx_section+40]
                section_ys = cfs_ys[40*idx_section:40*idx_section+40]
                section_closest_frames = closest_frames[40*idx_section:40*idx_section+40]
                num = len(section_xs)
                for idx in range(num-1):
                    plt.plot(section_xs[idx:idx+2], section_ys[idx:idx+2], c = [idx/(num-1), 0, 0])
                    if idx %2 == 0:
                        plt.text(section_xs[idx], section_ys[idx], str(section_closest_frames[idx]))
                plt.show()
            sys.exit()




        annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

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

        fig.canvas.mpl_connect("motion_notify_event", hover)


    def k_means_clustering(self, plot = False):

        kmeans = KMeans(n_clusters=3)
        self.clusters = kmeans.fit_predict(self.embeddings)

        if plot:
            fig_kmeans = plt.figure()
            ax = fig_kmeans.add_subplot(111)
            colors = ['red', 'blue', 'green', 'black', 'cyan', 'magenta', 'brown', 'gray', 'orange', 'pink']
            for idx, cluster in enumerate(self.clusters):
                ax.scatter(self.embeddings[idx, 0], self.embeddings[idx, 1], color = colors[cluster])
            #plt.show()

    def kde(self, load_or_save = 'load', file_name = 'mexh_kde.pickle'):


        if load_or_save == 'load':
            pdf_array = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_kde.pickle'.format(file_name), 'rb'))
        elif load_or_save == 'save':
            xs = np.linspace(np.min(self.all_embeddings[:, 0]), np.max(self.all_embeddings[:, 0]), 50)
            ys = np.linspace(np.min(self.all_embeddings[:, 1]), np.max(self.all_embeddings[:, 1]), 50)

            xx, yy = np.meshgrid(xs, ys)
            positions = np.vstack([xx.ravel(), yy.ravel()]).T

            kernel = KernelDensity(bandwidth = 5)
            kernel.fit(self.all_embeddings)


            pdf_array = np.exp(kernel.score_samples(positions))
            pdf_array = np.reshape(pdf_array, xx.shape)
            pickle.dump(pdf_array, open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_kde.pickle'.format(file_name), 'wb'))

        fig_kde = plt.figure()
        ax = fig_kde.add_subplot(111)
        ax.imshow(pdf_array[::-1, :], vmin = 0, vmax = 0.0003)
        ax.set_title('min:{:.2g}, max:{:.2g}'.format(np.min(pdf_array), np.max(pdf_array)))
        #plt.show()
        return pdf_array



    def plot_time_series(self, color_by_split = None):
        if color_by_split is not None:
            vmin, vmax = np.min(np.concatenate(color_by_split)), np.max(np.concatenate(color_by_split))
            cmap = plt.cm.PiYG
            norm = plt.Normalize(vmin, vmax)
        all_features = {'frame': [], 'PC0':[], 'PC1':[], 'PC2':[], 'delta_centroid':[], 'delta_sensing_direction':[]}
        for dict2 in dict1.values():
            for name, list in dict2.items():
                all_features[name] = all_features[name] + list

        for idx_series, (series_code, dict2) in enumerate(self.dict1.items()):
            fig = plt.figure()
            for idx_list, (name, list) in enumerate(dict2.items()):
                if name != 'frame':
                    ax = fig.add_subplot(5, 1, idx_list)
                    ax.plot(list)
                    if color_by_split is not None:
                        ax.scatter([i for i in range(len(list))], list, c = cmap(norm(color_by_split[idx_series])), vmin = vmin, vmax = vmax)

                    ax.set_ylim(min(all_features[name]), max(all_features[name]))
            plt.show()


def show_cell_series_clustered(idx_segments, center_frames):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    cwt = CWT(chop = None)
    cwt.set_spectograms()

    for idx_segment, center_frame in zip(idx_segments, center_frames):

        idx_cell, letter_keep = idx_segment[:-1], idx_segment[-1]
        cells = Cells(stack_attributes_2 + stack_attributes_3, cells_model = [idx_cell], max_l = 15)
        for lymph_series in cells.cells.values():
            dict = {}
            prev_frame = None
            count = 0
            keep = []

            for idx_lymph, lymph in enumerate(lymph_series):
                if prev_frame is not None and lymph.frame-prev_frame > 2:
                    count +=1
                letter = alphabet[count]
                # CHANGED
                if letter == letter_keep and abs(lymph.frame*lymph.t_res - center_frame*lymph.t_res) < 15:
                    keep.append(lymph)
                    lymph_t_res = lymph.t_res
                prev_frame = lymph.frame
        cells.cells[idx_cell] = keep
        cells.plot_orig_series(idx_cell=idx_cell, uropod_align = False, color_by = None, plot_every = 1, flat = True)

        cfs = [i for i in cwt.all_consecutive_frames if i.name == idx_segment][0]


        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2)
        ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        # CHANGED
        idxs_plot = [i for i,j in enumerate(cfs.closest_frames) if abs(j*lymph_t_res-center_frame*lymph_t_res) < 15]

        ax1.plot([j for i,j in enumerate(cfs.closest_frames) if i in idxs_plot], [j for i,j in enumerate(cfs.pca0_list) if i in idxs_plot], color = 'red')
        ax1.plot([j for i,j in enumerate(cfs.closest_frames) if i in idxs_plot], [j for i,j in enumerate(cfs.pca1_list) if i in idxs_plot], color = 'blue')
        ax1.plot([j for i,j in enumerate(cfs.closest_frames) if i in idxs_plot], [j for i,j in enumerate(cfs.pca2_list) if i in idxs_plot], color = 'green')
        ax1.set_ylim([-1, 1])

        # show spectogram
        spect = cfs.spectogram[:, idxs_plot]
        vert = cfs.spectogram[:, cfs.closest_frames.index(center_frame)][:, None]
        # CHANGED
        for ins in [1, 3, 5, 7, 9]:
            empty = np.empty(shape = (len(idxs_plot),))
            spect = np.insert(spect, ins, empty.fill(np.nan), 0)
            empty = np.empty(shape = (1,))
            vert = np.insert(vert, ins, empty.fill(np.nan), 0)
        vert = np.vstack([vert.T]*4)

        # CHANGED

        ax2.imshow(spect, cmap = 'PiYG', vmin = -0.2, vmax = 0.2)
        ax3.imshow(vert, cmap = 'PiYG', vmin = -0.2, vmax = 0.2)
        ax2.axis('off')
        ax3.axis('off')
        plt.show()





# COMPLEX
#wavelets = ['cgau1', 'cgau2', 'cgau3',  'cgau8', 'cmor{}-{}'.format(morB, morC)]
#scales_lists = [[0.6*i for i in range(1, 5)], [0.8*i for i in range(1, 5)], [1*i for i in range(1, 5)], [1.4*i for i in range(1, 5)], [0.65*i for i in range(1, 5)]]

# REAL
wavelets = ['mexh', 'gaus1', 'gaus2', 'gaus3']
scales_lists = [[0.5*i for i in range(2, 5)], [0.4*i for i in range(2, 9, 2)], [0.6*i for i in range(2, 5)], [0.8*i for i in range(2, 4)]]

# f_nyquist = 0.5


"""
show_cell_series_clustered(idx_segments = ['zm_3_3_0a', '3_1_1a'],
                            center_frames = [66, 6])
sys.exit()
"""

















# CHANGED
cwt = CWT(idx_cells = 'all', chop = 5)
#cwt.print_freqs()

#cwt.edge_effect_size()

cwt.set_spectograms()





#cwt.plot_wavelet_series_spectogram(name = 'all')

cwt.set_tsne_embeddings()
cwt.plot_embeddings(load_or_save = 'load', file_name = 'fast', path_of = 'zm_3_3_6a')


cwt.kde(load_or_save = 'load', file_name = 'fast')

plt.show()
