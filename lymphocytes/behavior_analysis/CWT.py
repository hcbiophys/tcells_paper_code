import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from scipy import signal
import pywt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity


dict1 = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/posture_series/all.pickle',"rb"))

class CWT():

    def __init__(self, dict1):
        self.dict1 = dict1

        self.spectograms = None
        self.num_features = None

        self.embeddings = None
        self.embeddings_split = None
        self.cell_codes = None




    def set_spectograms(self, wavelet = 'gaus1'):

        features = ['PC0', 'PC1', 'PC2', 'delta_centroid', 'delta_sensing_direction']
        #features = ['PC0']
        self.num_features = len(features)

        self.spectograms = []

        for code, dict2 in dict1.items():
            spectogram = []
            for idx_attribute, attribute in enumerate(features):
                coef, freqs = pywt.cwt(dict2[attribute], np.arange(1, 10), wavelet)
                #coef = abs(coef)
                spectogram.append(coef)
            spectogram = np.concatenate(spectogram, axis = 0)
            self.spectograms.append(spectogram)
            #ADD FWD / BACKWARD SIGN FOR delta_centroid AWAY / TOWARDS UROPOD
    def plot_spectograms(self):

        for idx_spect, spectogram in enumerate(self.spectograms):
            fig = plt.figure()
            for idx_feature in range(self.num_features):
                ax = fig.add_subplot(self.num_features, 1, idx_feature+1)
                spect_section = spectogram[idx_feature*int(spectogram.shape[0]/self.num_features):(idx_feature+1)*int(spectogram.shape[0]/self.num_features), :]
                ax.imshow(spect_section, vmin = -0.362, vmax = 0.654)
            plt.show()


    def set_tsne_embeddings(self):

        self.cell_codes = []
        for code, spectogram in zip(self.dict1.keys(), self.spectograms):
            self.cell_codes = self.cell_codes + [code]*spectogram.shape[1]

        self.embeddings = TSNE(n_components=2).fit_transform(np.concatenate(self.spectograms, axis = 1).T)
        idxs_cells = np.cumsum([0] + [i.shape[1] for i in self.spectograms])
        self.embeddings_split = [self.embeddings[idxs_cells[idx]:idxs_cells[idx+1], :] for idx in range(len(idxs_cells)-1)]

    def plot_embeddings(self):

        fig_dots = plt.figure()
        ax = fig_dots.add_subplot(111)
        for idx, embeddings in enumerate(self.embeddings_split):
            ax.plot(embeddings[:, 0], embeddings[:, 1], '-o', label = self.cell_codes[idx])
            ax.scatter(embeddings[0, 0], embeddings[0, 1], marker = 'x', c = 'red', s = 20)
            ax.scatter(embeddings[-1, 0], embeddings[-1, 1], marker = 'x', c = 'black', s = 20)
        plt.legend()
        #plt.show()


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

        xs = np.linspace(np.min(self.embeddings[:, 0]), np.max(self.embeddings[:, 0]), 50)
        ys = np.linspace(np.min(self.embeddings[:, 1]), np.max(self.embeddings[:, 1]), 50)
        xx, yy = np.meshgrid(xs, ys)
        positions = np.vstack([xx.ravel(), yy.ravel()]).T

        kernel = KernelDensity(bandwidth = 5)
        kernel.fit(self.embeddings)
        pdf_array = np.exp(kernel.score_samples(positions))
        pdf_array = np.reshape(pdf_array, xx.shape)

        if plot:
            fig_kde = plt.figure()
            ax = fig_kde.add_subplot(111)
            ax.imshow(pdf_array[::-1, :])
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
            print(series_code)
            fig = plt.figure()
            for idx_list, (name, list) in enumerate(dict2.items()):
                if name != 'frame':
                    ax = fig.add_subplot(5, 1, idx_list)
                    ax.plot(list)
                    if color_by_split is not None:
                        ax.scatter([i for i in range(len(list))], list, c = cmap(norm(color_by_split[idx_series])), vmin = vmin, vmax = vmax)

                    ax.set_ylim(min(all_features[name]), max(all_features[name]))
            plt.show()




for idx, wavelet in enumerate(pywt.wavelist(kind='continuous')):
    if idx >9:
        print(wavelet)
        cwt = CWT(dict1)
        cwt.set_spectograms(wavelet=wavelet)
        #cwt.plot_spectograms()
        cwt.set_tsne_embeddings()
        cwt.plot_embeddings()
        #cwt.k_means_clustering(plot = True)
        #color_by_split = [i[:, 0].flatten() for i in cwt.embeddings_split]
        #cwt.plot_time_series(color_by_split = color_by_split)

        cwt.kde(plot = True)

        plt.show()
