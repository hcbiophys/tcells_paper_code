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

    def __init__(self, wavelet, scales, idx_segment = 'all', min_length = 15):
        self.all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series.pickle',"rb"))
        if not idx_segment == 'all':
            self.all_consecutive_frames = [i for i in self.all_consecutive_frames if i.name == idx_segment]

        for i in self.all_consecutive_frames:
            if i.t_res == 2.5:
                i.frame_list = i.frame_list[::2]
                i.pca0_list = i.pca0_list[::2]
                i.pca1_list = i.pca1_list[::2]
                i.pca2_list = i.pca2_list[::2]




        idxs_keep = [i for i,j in enumerate(self.all_consecutive_frames) if len(j.pca0_list) > min_length]
        self.all_consecutive_frames = [j for i,j in enumerate(self.all_consecutive_frames) if i in idxs_keep]




        self.wavelet = wavelet
        self.scales = scales

        self.spectograms = None
        self.num_features = None

        self.all_embeddings = None


    def print_freqs(self):
        """
        Print the frequencies corresponding to different scales (this varies depending on the wavelet)
        """
        freqs = [pywt.scale2frequency(wavelet=wavelet, scale = scale) for scale in self.scales]
        print('scales: {}'.format(self.scales))
        print('freqs: {}'.format(freqs))

    def edge_effect_size(self):

        fake_series = [1 for i in range(40)]

        coef, freqs = pywt.cwt(fake_series, self.scales, self.wavelet)
        #coef = abs(coef)

        plt.imshow(coef)
        plt.show()
        sys.exit()


    def _plot_wavelets(self, frames_plot = 'all'):

        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        for idx_scale, scale in enumerate(self.scales):

            print('idx_scale', idx_scale)
            t_ = Symbol('t_')
            if self.wavelet[:4] == 'cmor':
                func = (1/sqrt(morB*pi))*exp((-((t_-5)/scale)**2)/morB)*exp(2*pi*1.j*morC*((t_-5)/scale))
                func +=  (scale-min(self.scales)) + (scale-min(self.scales))*I
            elif self.wavelet[:4] == 'cgau':
                func = exp(-1.j*((t_-5)/scale))*exp(-((t_-5)/scale)**2)  # t_-scale*2 for plotting each shifted horizontally
                for _ in range(int(self.wavelet[4])):
                    func = func.diff(t_)
                func +=  (scale-min(self.scales)) + (scale-min(self.scales))*I
            elif self.wavelet[:4] == 'mexh':
                func = (1-(((t_-5)/scale)**2))*exp(-0.5*(((t_-5)/scale)**2))
            elif self.wavelet[:4] == 'gaus':
                func = exp(-(((t_-5)/scale)**2))
                for _ in range(int(self.wavelet[4])):
                    func = func.diff(t_)

            if frames_plot == 'all':
                p_real = plot(re(func), show = False, xlim = [-10, 500], ylim = [-2, 7])
                p_im = plot(im(func), show = False, xlim = [-10, 500], ylim = [-2, 7])
            else:
                print(len(frames_plot.pca0_list))
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

        else:

            ax.plot([j for j in range(len(frames_plot.pca0_list))], frames_plot.pca0_list, color = 'red')
            ax.plot([j for j in range(len(frames_plot.pca1_list))], frames_plot.pca1_list, color = 'blue')
            ax.plot([j for j in range(len(frames_plot.pca2_list))], frames_plot.pca2_list, color = 'green')




    def set_spectograms(self, chop = 3, with_signals = False):

        features = ['pca0_list', 'pca1_list', 'pca2_list']
        self.num_features = len(features)


        for consecutive_frames in self.all_consecutive_frames:
            spectogram = []
            for idx_attribute, attribute in enumerate(features):
                coef, freqs = pywt.cwt(getattr(consecutive_frames, attribute), self.scales, self.wavelet)
                #coef = abs(coef)
                if with_signals:
                    coef = np.vstack([getattr(consecutive_frames, attribute), coef])
                spectogram.append(coef)
            spectogram = np.concatenate(spectogram, axis = 0)
            spectogram = spectogram[:, chop:-chop]
            consecutive_frames.spectogram = spectogram

            # update other variables after chop
            consecutive_frames.frame_list = consecutive_frames.frame_list[chop:-chop]
            consecutive_frames.pca0_list = consecutive_frames.pca0_list[chop:-chop]
            consecutive_frames.pca1_list = consecutive_frames.pca1_list[chop:-chop]
            consecutive_frames.pca2_list = consecutive_frames.pca2_list[chop:-chop]

    def _plot_spectogram(self, spectogram, d3 = False):

        if not d3:
            fig = plt.figure()
            for idx_feature in range(self.num_features):
                ax = fig.add_subplot(self.num_features, 1, idx_feature+1)
                spect_section = spectogram[idx_feature*int(spectogram.shape[0]/self.num_features):(idx_feature+1)*int(spectogram.shape[0]/self.num_features), :]
                ax.imshow(spect_section, vmin = -0.25, vmax = 0.25)
        else:
            fig = plt.figure()
            for idx_feature in range(self.num_features):
                ax = fig.add_subplot(1, self.num_features, idx_feature+1, projection='3d')
                spect_section = spectogram[idx_feature*int(spectogram.shape[0]/self.num_features):(idx_feature+1)*int(spectogram.shape[0]/self.num_features), :]
                x, y = np.array(list(range(spect_section.shape[1]))), np.array(list(range(spect_section.shape[0])))
                xx, yy = np.meshgrid(x, y)
                ax.plot_surface(xx, yy, spect_section, cmap=cm.coolwarm)
                ax.set_xlabel('time')
                ax.set_ylabel('scale')
                ax.set_zlabel('amplitude')

    def plot_wavelet_series_spectogram(self, d3 = False, name = 'all'):
        for consecutive_frames in self.all_consecutive_frames:
            if name == 'all' or consecutive_frames.name == name:
                self._plot_wavelets(frames_plot = consecutive_frames)
                self._plot_spectogram(spectogram = consecutive_frames.spectogram, d3 = d3)
                plt.show()




    def set_tsne_embeddings(self):

        self.all_embeddings = TSNE(n_components=2).fit_transform(np.concatenate([i.spectogram for i in self.all_consecutive_frames], axis = 1).T)
        idxs_cells = np.cumsum([0] + [i.spectogram.shape[1] for i in self.all_consecutive_frames])
        for idx, consecutive_frame in enumerate(self.all_consecutive_frames):
            consecutive_frame.embeddings = self.all_embeddings[idxs_cells[idx]:idxs_cells[idx+1], :]

    def plot_embeddings(self):



        fig = plt.figure()
        ax = fig.add_subplot(111)
        xs, ys, colors, names = [], [], [], []
        for consecutive_frames in self.all_consecutive_frames:
            colors += [np.random.rand(3,)]*len(consecutive_frames.frame_list)
            xs += list(consecutive_frames.embeddings[:, 0])
            ys += list(consecutive_frames.embeddings[:, 1])
            names += [consecutive_frames.name + '-' + str(i) for i in consecutive_frames.frame_list]


        sc = ax.scatter(xs, ys, c = colors)
        for consecutive_frames in self.all_consecutive_frames:
            if consecutive_frames.name == 'zm_3_2_0a':
                ax.plot(consecutive_frames.embeddings[:, 0], consecutive_frames.embeddings[:, 1])

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

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

    def kde(self, plot = False):

        xs = np.linspace(np.min(self.all_embeddings[:, 0]), np.max(self.all_embeddings[:, 0]), 50)
        ys = np.linspace(np.min(self.all_embeddings[:, 1]), np.max(self.all_embeddings[:, 1]), 50)
        xx, yy = np.meshgrid(xs, ys)
        positions = np.vstack([xx.ravel(), yy.ravel()]).T

        kernel = KernelDensity(bandwidth = 5)
        kernel.fit(self.all_embeddings)
        pdf_array = np.exp(kernel.score_samples(positions))
        pdf_array = np.reshape(pdf_array, xx.shape)

        if plot:
            fig_kde = plt.figure()
            ax = fig_kde.add_subplot(111)
            ax.imshow(pdf_array[::-1, :])
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


def show_cell_series_clustered(idx_segments, center_frames, wavelet = None, scales = None, chop = None):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    for idx_segment, center_frame in zip(idx_segments, center_frames):
        idx_cell, letter_keep = idx_segment[:-1], idx_segment[-1]
        cells = Cells(stack_attributes_2 + stack_attributes_3, cells_model = [idx_cell], max_l = 15)
        for lymph_series in cells.cells.values():
            dict = {}
            prev_frame = None
            count = 0
            keep = []
            frame_list_1 = np.linspace(center_frame-6, center_frame+6, 5)
            frame_list_2 = np.linspace(center_frame-6, center_frame+6, 5)
            for idx_lymph, lymph in enumerate(lymph_series):
                if prev_frame is not None and lymph.frame-prev_frame > 2:
                    count +=1
                letter = alphabet[count]
                if letter == letter_keep and lymph.frame in frame_list_1:
                    keep.append(lymph)
                prev_frame = lymph.frame
        cells.cells[idx_cell] = keep
        cells.plot_orig_series(idx_cell=idx_cell, uropod_align = True, color_by = None, plot_every = 1, flat = True)

        all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series.pickle',"rb"))
        cfs = [i for i in all_consecutive_frames if i.name == idx_cell + letter_keep][0]

        fig = plt.figure()
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
        idxs_plot = [i for i,j in enumerate(cfs.frame_list) if j in frame_list_2]
        ax1.plot(range(len(idxs_plot)), [j for i,j in enumerate(cfs.pca0_list) if i in idxs_plot], color = 'red')
        ax1.plot(range(len(idxs_plot)), [j for i,j in enumerate(cfs.pca1_list) if i in idxs_plot], color = 'blue')
        ax1.plot(range(len(idxs_plot)), [j for i,j in enumerate(cfs.pca2_list) if i in idxs_plot], color = 'green')
        ax1.set_ylim([-0.8, 0.8])


        # show spectogram
        if wavelet is not None and scales is not None and chop is not None:
            cwt = CWT(wavelet = wavelet, scales = scales, idx_segment = idx_segment)
            cwt.set_spectograms(chop = chop)
            idxs_keep = [i for i,j in enumerate(cwt.all_consecutive_frames[0].frame_list) if j in frame_list_2]
            truncated_spectogram = cwt.all_consecutive_frames[0].spectogram[:, idxs_keep]
            ax2.imshow(truncated_spectogram, vmin = -0.25, vmax = 0.25)
        plt.show()


show_cell_series_clustered(idx_segments = ['zm_3_2_6a', '2_4a', '3_1_3a', '3_0_0a' ,'3_0_0a', '2_9c', '2_6a', 'zm_3_2_41'],
                            center_frames = [71, 9, 25, 36, 20, 198, 33, 60],
                            wavelet = 'mexh', scales = [0.5*i for i in range(1, 5)], chop = 3)
sys.exit()

CHANGE IT SO SECTOGRAMS AND SERIES SHOW ALL IN WINDOW OF 10




# COMPLEX
#wavelets = ['cgau1', 'cgau2', 'cgau3',  'cgau8', 'cmor{}-{}'.format(morB, morC)]
#scales_lists = [[0.6*i for i in range(1, 5)], [0.8*i for i in range(1, 5)], [1*i for i in range(1, 5)], [1.4*i for i in range(1, 5)], [0.65*i for i in range(1, 5)]]

# REAL
wavelets = ['mexh', 'gaus1', 'gaus2', 'gaus3']
scales_lists = [[0.5*i for i in range(1, 5)], [0.4*i for i in range(1, 5)], [0.6*i for i in range(1, 5)], [0.8*i for i in range(1, 5)]]

# f_nyquist = 0.5
for wavelet, scales in zip(wavelets, scales_lists):


    cwt = CWT(wavelet = wavelet, scales = scales, idx_segment = 'all')
    #cwt.print_freqs()

    #cwt.edge_effect_size()

    cwt.set_spectograms(chop = 3, with_signals = False)

    #spectograms_all = [i.spectogram for i in cwt.all_consecutive_frames]
    #spectograms_all = np.concatenate(spectograms_all, axis = 1)


    #cwt.plot_wavelet_series_spectogram(d3 = False, name = 'all')

    cwt.set_tsne_embeddings()
    cwt.plot_embeddings()
    #cwt.k_means_clustering(plot = True)
    #color_by_split = [i[:, 0].flatten() for i in cwt.embeddings_split]
    #cwt.plot_time_series(color_by_split = color_by_split)

    cwt.kde(plot = True)

    plt.show()
