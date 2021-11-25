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
from scipy.spatial import Voronoi, voronoi_plot_2d
import pyvista as pv
pv.set_plot_theme("document")
import random
import statsmodels.api as sm
import matplotlib.gridspec as gridspec
from statsmodels.tsa.api import VAR
import copy
import cv2
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline
from skimage.morphology import skeletonize
import itertools
from scipy.optimize import curve_fit
from scipy import interpolate
from matplotlib import colors as mpl_colors
from scipy.linalg import eig
from scipy import signal


from lymphocytes.data.dataloader_good_segs_2 import stack_attributes_2
from lymphocytes.data.dataloader_good_segs_3 import stack_attributes_3
from lymphocytes.cells.cells_class import Cells


filename = sys.argv[1]
load_or_save_or_run = sys.argv[2]



def get_scales(dyadic = False):
    if dyadic:
        scale_orig_bys = [2**j for j in range(10)]
    else:
        scale_orig_bys = range(1, 10)

    mexh_scales = [0.5*scale_orig_by for scale_orig_by in scale_orig_bys]
    gaus1_scales = [0.4*scale_orig_by for scale_orig_by in scale_orig_bys]



    return mexh_scales, gaus1_scales

"""
def get_scales(freq_step_size = 0.05, num_steps = 10):
    target_freqs = [0.5 - i*freq_step_size for i in range(num_steps)]
    mexh_scales = []
    mexh_scales_all = [0.5*i for i in range(40)]
    for i in target_freqs:
        dists = [abs(i-pywt.scale2frequency(wavelet='mexh', scale = j)) for j in mexh_scales_all]
        mexh_scales.append(mexh_scales_all[dists.index(min(dists))])


    gaus1_scales = []
    gaus1_scales_all = [0.4*i for i in range(40)]
    for i in target_freqs:
        dists = [abs(i-pywt.scale2frequency(wavelet='gaus1', scale = j)) for j in gaus1_scales_all]
        gaus1_scales.append(gaus1_scales_all[dists.index(min(dists))])

    plt.plot([pywt.scale2frequency(wavelet='mexh', scale = j) for j in mexh_scales])
    plt.show()

    return mexh_scales, gaus1_scales
"""




# s,b
thresh_params_dict = {'50': (13, 25), '150': (15, 25), '150_PC2': (13, 35), '150_PC1': (9, 25)}




if filename[:2] == '30':
    mexh_scales = [0.5*2]
    gaus1_scales = [0.4*2]
    chop = 5
    inserts = [1, 3, 5, 7, 9]
    time_either_side = 15
    min_length = 15

elif filename[:2] == '50':
    mexh_scales = [0.5*i for i in range(2, 5)]
    gaus1_scales = [0.4*i for i in range(2, 9, 2)]
    chop = 5
    inserts = [3, 8, 12, 17, 21]
    time_either_side = 25
    min_length = 15


elif filename[:2] == '75':
    filename = '75'
    mexh_scales = [0.5*i for i in range(2, 7, 2)]
    gaus1_scales = [0.4*i for i in range(2, 12, 4)]
    chop = 7
    inserts = [3, 8, 12, 17, 21]
    time_either_side = 37.5
    min_length = 15

elif filename[:3] == '100':
    mexh_scales = [0.5*i for i in range(2, 10, 2)]
    gaus1_scales = [0.4*i for i in range(2, 18, 4)]
    chop = 10
    inserts = [4+5*i for i in range(6)]
    time_either_side = 50
    min_length = 15


elif filename[:3] == '150':
    mexh_scales = [0.5*i for i in range(2, 14, 2)]
    gaus1_scales = [0.4*i for i in range(2, 24, 4)]
    chop = 15
    inserts = [7+6*i for i in range(6)]
    time_either_side = 75
    min_length = 15



elif filename[:3] == '200':
    mexh_scales = [0.5*i for i in range(2, 20, 2)]
    gaus1_scales = [0.4*i for i in range(2, 34, 4)]
    chop = 20
    inserts = [10+9*i for i in range(6)]
    time_either_side = 100
    min_length = 15


elif filename[:3] == '250':
    mexh_scales = [0.5*i for i in range(2, 24, 2)]
    gaus1_scales = [0.4*i for i in range(2, 44, 4)]
    chop = 25
    inserts = [12+11*i for i in range(6)]
    time_either_side = 125
    min_length = 15


elif filename[:3] == '400':
    mexh_scales = [0.5*i for i in range(2, 32, 2)]
    gaus1_scales = [0.4*i for i in range(2, 60, 4)]
    chop = 25
    inserts = [16+15*i for i in range(6)]
    time_either_side = 175
    min_length = 15






def move_sympyplot_to_axes(p, ax):
    backend = p.backend(p)
    backend.ax = ax
    backend._process_series(backend.parent._series, ax, backend.parent)
    backend.ax.spines['right'].set_color('none')
    backend.ax.spines['bottom'].set_position('zero')
    backend.ax.spines['top'].set_color('none')
    plt.close(backend.fig)


morB, morC = 10, 0.3
class CWT():

    def __init__(self, idx_segment = 'all', min_length = 15, chop = 5):
        self.chop = chop



        if filename[-3:] == 'run':
            self.all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series_run.pickle',"rb"))

            print('RUN CELLS')

        elif filename[-4:] == 'stop':
            self.all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series_stop.pickle',"rb"))
            print('STOP CELLS')
        else:
            self.all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series.pickle',"rb"))

        self.all_consecutive_frames = [i for i in self.all_consecutive_frames if i.name !='zm_3_4_2a']


        if not idx_segment == 'all':
            self.all_consecutive_frames = [i for i in self.all_consecutive_frames if i.name == idx_segment]





        """
        for cfs in self.all_consecutive_frames:
            if cfs.name == '2_1a':
                plt.plot([i*5 for i,j in enumerate(cfs.run_list)], [i*75 for i in cfs.run_list], c = 'black', linestyle = '--')

            if cfs.name == 'zm_3_3_5a':
                plt.plot([i*5 for i,j in enumerate(cfs.run_list)], [i*75 for i in cfs.run_list], c = 'black', linestyle = ':')
                plt.plot([i*5 for i,j in enumerate(cfs.run_list)], [0 for _ in cfs.run_list], c = 'black', linewidth = 0.5)
        plt.show()
        sys.exit()
        """



        """
        fig = plt.figure()
        count = 0
        #random.shuffle(self.all_consecutive_frames)

        for cfs in self.all_consecutive_frames:

            if len(cfs.pca0_list) > 100 and count < 4:
                if cfs.name == 'zm_3_3_5a': # zm_3_3_5a, zm_3_3_2a, zm_3_3_4a, zm_3_4_1a
                    ax = fig.add_subplot(4, 1, count+1)
                    ax.plot([i*5 for i,j in enumerate(cfs.pca0_list)], cfs.pca0_list, c = 'red')
                    ax.plot([i*5 for i,j in enumerate(cfs.pca1_list)], cfs.pca1_list, c = 'blue')
                    ax.plot([i*5 for i,j in enumerate(cfs.pca2_list)], cfs.pca2_list, c = 'green')
                    ax.plot([i*5 for i,j in enumerate(cfs.run_list)], [i*75 for i in cfs.run_list], c = 'black', linestyle = '--')
                    #ax.plot([i*5 for i,j in enumerate(cfs.run_mean_list)], [i*75 for i in cfs.run_mean_list], c = 'orange', linestyle = '--')

                    #ax.plot([i*5 for i,j in enumerate(cfs.spin_vec_magnitude_list)], [50*i for i in cfs.spin_vec_magnitude_list], c = 'pink')
                    #ax.plot([i*5 for i,j in enumerate(cfs.spin_vec_magnitude_mean_list)], [50*i for i in cfs.spin_vec_magnitude_mean_list], c = 'pink', linestyle = '--')
                    #ax.plot([i*5 for i,j in enumerate(cfs.spin_vec_std_list)], [50*i for i in cfs.spin_vec_std_list], c = 'pink', linestyle = ':')
                    #ax.plot([i*5 for i,j in enumerate(cfs.direction_std_list)], [i for i in cfs.direction_std_list], c = 'grey')

                    #ax.plot([i*5 for i,j in enumerate(cfs.run_list)], [0 for _ in cfs.run_list], c = 'black', linewidth = 0.5)

                    ax.set_ylim([-1, 1])
                    #ax.set_title(cfs.name)
                    count += 1

        plt.show()
        plt.subplots_adjust(hspace = 0)
        sys.exit()
        """



        #DO KDE WITH SMALLER TIME SCALES TO VALIDATE THAT (low number of) MOTIFS STRUCTURE IS AT ~50s
        idxs_keep = [i for i,j in enumerate(self.all_consecutive_frames) if len(j.pca0_list) > min_length]
        self.all_consecutive_frames = [j for i,j in enumerate(self.all_consecutive_frames) if i in idxs_keep]



        self.spectograms = None
        self.num_features = None

        self.all_embeddings = None


        def butter_highpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
            return b, a

        def butter_highpass_filter(data, cutoff, fs, order=5):
            b, a = butter_highpass(cutoff, fs, order=order)
            y = signal.filtfilt(b, a, data)
            return y

        acorrs1 = [[], [], [], []]
        acorrs2 = [[], [], [], []]
        colors = ['red', 'blue', 'green',  'black']
        def _model_func(x,  k):
            x = np.array(x)
            return np.exp(-k*x)

        fig_series = plt.figure()
        ax1 = fig_series.add_subplot(211)
        ax2 = fig_series.add_subplot(212)
        for i in self.all_consecutive_frames:
            if i.name == '2_1a':
                for idx, l in enumerate([i.pca0_list, i.pca1_list, i.pca2_list, i.run_list]):
                    acf = list(sm.tsa.acf(l, nlags = 99, missing = 'conservative'))
                    acf += [np.nan for _ in range(100-len(acf))]
                    acorrs1[idx].append(np.array(acf))

                    ax1.plot([i*5 for i in range(len(l))], [i*(1/np.nanmax(l)) for i in l], c = colors[idx])

                    if len([i for i in l if np.isnan(i)]) > 0:

                        f = interpolate.interp1d([i*5  for i,j in enumerate(l) if not np.isnan(j)], [j  for i,j in enumerate(l)if not np.isnan(j)])

                        l = f([i*5 for i in range(len(l))][3:-2])
                    l_new  = butter_highpass_filter(l,1/400,fs=0.2)

                    ax2.plot([i*5 for i in range(len(l_new))], [i*(1/np.nanmax(l_new)) for i in l_new], c = colors[idx])
                    acf = list(sm.tsa.acf(l_new, nlags = 99, missing = 'conservative'))
                    acf += [np.nan for _ in range(100-len(acf))]
                    acorrs2[idx].append(np.array(acf))



        fig_acf = plt.figure()
        ax1 = fig_acf.add_subplot(211)
        ax2 = fig_acf.add_subplot(212)



        for idx in range(len(acorrs1)):
            xs = [i*5 for i in range(len(acorrs1[idx][0]))]
            ys = acorrs1[idx][0]

            ax1.plot(xs, ys, c = colors[idx])
            ax1.plot(xs, [0 for _ in xs], linewidth = 0.1, c = 'grey')

            xs = [i*5 for i in range(len(acorrs2[idx][0]))]
            ys = acorrs2[idx][0]

            ax2.plot(xs, ys, c = colors[idx])
            ax2.plot(xs, [0 for _ in xs], linewidth = 0.1, c = 'grey')
        plt.show()




        for acorr, color, linestyle in zip(acorrs, colors, ['-', '-', '-', '--']):
            acorr = np.array([acorr])
            concat = np.concatenate(acorr, axis = 0)
            ys = np.nanmean(concat, axis = 0)
            xs = [i*5 for i,j in enumerate(ys)]
            plt.plot(xs, ys, c = color, linestyle = linestyle)
            plt.plot(xs, [0 for _ in xs], linewidth = 0.1, c = 'grey')

            points=[(xs[0], ys[0])]
            for idx in range(1, len(ys)-2):

                if  ys[idx] > ys[idx-1] and ys[idx] > ys[idx+1] :
                    points.append((xs[idx], ys[idx]))


            xs_scatter = [i[0] for i in points]
            ys_scatter = [i[1] for i in points]
            #plt.scatter(xs_scatter, ys_scatter)

            f = interpolate.interp1d(xs_scatter, ys_scatter)
            xs_scatter = [i for i in range(max(xs_scatter))]
            ys_scatter = f(xs_scatter)

            p0 = (0.01) # starting search koefs
            opt, pcov = curve_fit(_model_func, xs_scatter, ys_scatter)

            k = opt

            ys_model = _model_func(xs_scatter,  k = k)
            #plt.plot(xs_scatter, ys_model, c = color, linestyle = linestyle)
            tau = 1./k
            print(color, tau)

        #plt.ylim([0, 1])
        plt.show()
        sys.exit()




        self.all_consecutive_frames_dict = {cfs.name: cfs for cfs in self.all_consecutive_frames}






    def print_freqs(self):
        """
        Print the frequencies corresponding to different scales (this varies depending on the wavelet)
        """
        mexh_freqs = [pywt.scale2frequency(wavelet='mexh', scale = scale) for scale in mexh_scales]
        print('mexh scales: {}'.format(mexh_scales))
        print('mexh freqs: {}'.format(mexh_freqs))
        gaus1_freqs = [pywt.scale2frequency(wavelet='gaus1', scale = scale) for scale in gaus1_scales]
        print('gaus1 scales: {}'.format(gaus1_scales))
        print('gaus1 freqs: {}'.format(gaus1_freqs))

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.plot(mexh_scales, mexh_freqs)
        ax = fig.add_subplot(122)
        ax.plot(gaus1_scales, gaus1_freqs)
        plt.show()
        sys.exit()

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

                coef, _ = pywt.cwt(getattr(consecutive_frames, attribute), mexh_scales, 'mexh')
                spectogram.append(coef)
                coef, _ = pywt.cwt(getattr(consecutive_frames, attribute), gaus1_scales, 'gaus1')
                spectogram.append(coef)

            spectogram = np.concatenate(spectogram, axis = 0)

            if self.chop is not None:
                spectogram = spectogram[:, self.chop :-self.chop]
                consecutive_frames.closest_frames = consecutive_frames.closest_frames[self.chop :-self.chop]
                consecutive_frames.pca0_list = consecutive_frames.pca0_list[self.chop :-self.chop]
                consecutive_frames.pca1_list = consecutive_frames.pca1_list[self.chop :-self.chop]
                consecutive_frames.pca2_list = consecutive_frames.pca2_list[self.chop :-self.chop]
                consecutive_frames.delta_centroid_list = consecutive_frames.delta_centroid_list[self.chop :-self.chop]
                consecutive_frames.delta_sensing_direction_list = consecutive_frames.delta_sensing_direction_list[self.chop :-self.chop]
                consecutive_frames.run_list = consecutive_frames.run_list[self.chop :-self.chop]
                consecutive_frames.run_mean_list = consecutive_frames.run_mean_list[self.chop :-self.chop]
                consecutive_frames.spin_vec_magnitude_list = consecutive_frames.spin_vec_magnitude_list[self.chop :-self.chop]
                consecutive_frames.spin_vec_magnitude_mean_list = consecutive_frames.spin_vec_magnitude_mean_list[self.chop :-self.chop]
                consecutive_frames.spin_vec_std_list = consecutive_frames.spin_vec_std_list[self.chop :-self.chop]
                consecutive_frames.direction_std_list = consecutive_frames.direction_std_list[self.chop :-self.chop]

            consecutive_frames.spectogram = spectogram

    def _plot_spectogram(self, spectogram):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # CHANGED
        #ax.imshow(spectogram, vmin = -0.6, vmax = 0.6)
        ax.imshow(spectogram, vmin = -0.25, vmax = 0.25)

        # CHANGED
        for ins in inserts:
            spectogram = np.insert(spectogram, ins, np.zeros(shape = (spectogram.shape[1],)), 0)
        # CHANGED
        #ax.imshow(spectogram, cmap = 'PiYG', vmin = -0.6, vmax = 0.6)
        ax.imshow(spectogram, cmap = 'PiYG', vmin = -0.25, vmax = 0.25)
        ax.axis('off')


    def plot_wavelet_series_spectogram(self, name = 'all'):
        for cfs in self.all_consecutive_frames:
            if name == 'all' or cfs.name == name:
                # CHANGED

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




    def set_tsne_embeddings(self, load_or_save_or_run, filename):

        if load_or_save_or_run == 'load':
            data = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_dots.pickle'.format(filename), 'rb'))
            colors = data['colors']
            xs = data['xs']
            ys = data['ys']
            names = data['names']

            self.all_embeddings = np.array(list(zip(xs, ys)))

        elif load_or_save_or_run == 'save' or load_or_save_or_run == 'run':
            concat = np.concatenate([i.spectogram for i in self.all_consecutive_frames], axis = 1).T

            per_PC = int(concat.shape[1]/3)
            if '_' in filename:
                if filename.split('_')[1] == 'PC1':
                    concat = concat[:, :per_PC]
                elif filename.split('_')[1] == 'PC2':
                    concat = concat[:, per_PC:2*per_PC]
                elif '_' in filename and filename.split('_')[1] == 'PC3':
                    concat = concat[:, 2*per_PC:]


            self.all_embeddings = TSNE(n_components=2).fit_transform(concat)



        idxs_cells = np.cumsum([0] + [i.spectogram.shape[1] for i in self.all_consecutive_frames])
        for idx, consecutive_frame in enumerate(self.all_consecutive_frames):
            consecutive_frame.embeddings = self.all_embeddings[idxs_cells[idx]:idxs_cells[idx+1], :]


    def _set_embedding_colors(self, xs, ys):


        run_all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series_run.pickle',"rb"))
        run_all_consecutive_frames_dict = {cfs.name: cfs for cfs in run_all_consecutive_frames}
        run_idx_cells = list([i[:-1] for i in run_all_consecutive_frames_dict.keys()])
        stop_all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series_stop.pickle',"rb"))
        stop_all_consecutive_frames_dict = {cfs.name: cfs for cfs in stop_all_consecutive_frames}
        stop_idx_cells = list([i[:-1] for i in stop_all_consecutive_frames_dict.keys()])


        colors_pc = []
        colors_run = []
        colors_mode = []
        for cfs in self.all_consecutive_frames:
            """
            x_list = np.array([k*5 for k,j in enumerate(i.run_list)])
            y_list = np.array(i.run_list)
            x_list2 = np.arange(min(x_list), max(x_list), 0.1)
            #plt.plot(x_list, y_list,  c = 'black')
            spl = UnivariateSpline(x_list, y_list, s = 1e-4)
            plt.plot(x_list2, spl(x_list2), c = 'blue')
            plt.plot(x_list2, [0 for _ in x_list2], c = 'blue')
            plt.plot(x_list2, 30*spl(x_list2, 1), c = 'red')
            plt.plot(x_list2, 100*spl(x_list2, 2), c = 'red', linestyle = '--')
            plt.show()
            sys.exit()
            """



            colors_run += list(cfs.run_list)
            colors_pc += list(cfs.pca0_list)

            if cfs.name[:-1] in run_idx_cells:

                for j in cfs.closest_frames:
                    if j in run_all_consecutive_frames_dict[cfs.name].closest_frames:
                        colors_mode.append('red')
                    else:
                        colors_mode.append('grey')

            elif cfs.name[:-1] in stop_idx_cells:
                for j in cfs.closest_frames:
                    if j in stop_all_consecutive_frames_dict[cfs.name].closest_frames:
                        colors_mode.append('blue')
                    else:
                        colors_mode.append('grey')
            else:
                for j in cfs.closest_frames:
                    colors_mode.append('grey')


        colors_2d = colors_mode
        colors_3d = None



        return xs, ys, colors_2d, colors_3d


    def _plot_path_of(self, xs, ys, names, path_of, num_per_section = 40):

        cfs_xs, cfs_ys = [], []
        closest_frames = self.all_consecutive_frames_dict[path_of].closest_frames

        for idx in range(len(xs)):
            if names[idx].split('-')[0] == path_of:
                cfs_xs.append(xs[idx])
                cfs_ys.append(ys[idx])
        plt.plot(cfs_xs, cfs_ys, '-o')

        contours = []
        for i in np.linspace(np.min(self.all_embeddings[:, 0])-10, np.max(self.all_embeddings[:, 0])+10, 6)[1:-1]:
            for j in np.linspace(np.min(self.all_embeddings[:, 1])-10, np.max(self.all_embeddings[:, 1])+10, 6)[1:-1]:
                contours.append(np.array([i, j]))
        plt.scatter([c[0] for c in contours], [c[1] for c in contours], s =50, c = 'black' )

        #for idx in range(len(cfs_xs)):
            #plt.text(cfs_xs[idx], cfs_ys[idx], str(chop*5 + 5*idx))

        """
        for idx_section in range(1+len(cfs_xs)//num_per_section):
            section_xs = cfs_xs[num_per_section*idx_section:num_per_section*idx_section+num_per_section]
            section_ys = cfs_ys[num_per_section*idx_section:num_per_section*idx_section+num_per_section]
            section_closest_frames = closest_frames[num_per_section*idx_section:num_per_section*idx_section+num_per_section]
            num = len(section_xs)
            print('num', num)


            for idx in range(num-1):
                plt.plot(section_xs[idx:idx+2], section_ys[idx:idx+2], c = [idx/(num-1), 0, 0])
                if idx %2 == 0:
                    plt.text(section_xs[idx], section_ys[idx], str(section_closest_frames[idx]))
        """





    def plot_embeddings(self, load_or_save_or_run = 'save', filename = None, path_of = None):

        if load_or_save_or_run == 'load':
            data = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_dots.pickle'.format(filename), 'rb'))
            colors = data['colors']
            xs = data['xs']
            ys = data['ys']
            names = data['names']


        elif load_or_save_or_run == 'save' or load_or_save_or_run == 'run':
            data = {}
            xs, ys, zs, colors, names = [], [], [], [], []
            for consecutive_frames in self.all_consecutive_frames:
                colors += [np.random.rand(3,)]*consecutive_frames.embeddings.shape[0]
                xs += list(consecutive_frames.embeddings[:, 0])
                ys += list(consecutive_frames.embeddings[:, 1])
                names += [consecutive_frames.name + '-' + str(i) for i in consecutive_frames.closest_frames]
            data['colors'] = colors
            data['xs'] = xs
            data['ys'] = ys
            data['names'] = names
            if load_or_save_or_run == 'save':
                pickle.dump(data, open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_dots.pickle'.format(filename), 'wb'))



        #all = np.concatenate([i.spectogram for i in self.all_consecutive_frames], axis = 1)
        #colors = np.max(abs(all), axis = 0)


        xs, ys, colors_2d, colors_3d = self._set_embedding_colors(xs, ys)




        """
        plotter = pv.Plotter()
        for x, y, z, color in zip(xs, ys, zs, colors_3d):
            if not np.isnan(x):
                plotter.add_mesh(pv.Sphere(radius=0.4, center=np.array([x, y, z])), color = color)
        plotter.show()
        """





        if path_of is not None:
            self._plot_path_of(xs, ys, names, path_of, num_per_section = 200)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            sc = ax.scatter(xs, ys, c = colors_2d, vmax = 0.015, cmap = 'Blues')






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




    def k_means_clustering(self, n_clusters, plot = False):

        kmeans = KMeans(n_clusters=n_clusters)
        self.clusters = kmeans.fit_predict(self.all_embeddings)

        if plot:
            fig_kmeans = plt.figure()
            ax = fig_kmeans.add_subplot(111)
            colors = ['red', 'blue', 'green', 'black', 'cyan', 'magenta', 'brown', 'gray', 'orange', 'pink']
            for idx, cluster in enumerate(self.clusters):
                ax.scatter(self.all_embeddings[idx, 0], self.all_embeddings[idx, 1], color = colors[cluster])
            plt.show()

    def kde(self, load_or_save_or_run = 'load', filename = 'mexh_kde.pickle'):


        if load_or_save_or_run == 'load':
            pdf_array = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_kde.pickle'.format(filename), 'rb'))
        elif load_or_save_or_run == 'save' or load_or_save_or_run == 'run':
            xs = np.linspace(np.min(self.all_embeddings[:, 0])-10, np.max(self.all_embeddings[:, 0])+10, 50)
            ys = np.linspace(np.min(self.all_embeddings[:, 1])-10, np.max(self.all_embeddings[:, 1])+10, 50)

            xx, yy = np.meshgrid(xs, ys)
            positions = np.vstack([xx.ravel(), yy.ravel()]).T

            kernel = KernelDensity(bandwidth = 5)
            kernel.fit(self.all_embeddings)


            pdf_array = np.exp(kernel.score_samples(positions))
            pdf_array = np.reshape(pdf_array, xx.shape)
            if load_or_save_or_run == 'save':
                pickle.dump(pdf_array, open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_kde.pickle'.format(filename), 'wb'))

        self.pdf_array = pdf_array
        plt.imshow(self.pdf_array[::-1, :])
        plt.show()
        return pdf_array


    def _get_contours(self, pdf, s, b):

        if s is None and b is None:
            plt.imshow(pdf)
            plt.show()
            s_list = [ 9, 11, 13, 15, 17, 19] # ROWS
            b_list = [15, 20, 25, 30, 35] # COLUMNS
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
                                pdf_new[row, col] = 1
                            else:
                                pdf_new[row, col] = 0
                    ax.imshow(pdf_new)
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
        im = np.zeros_like(pdf)
        #cv2.drawContours(im, [cnt], 0, (255), 3)
        contours = [np.squeeze(i) for i in contours]

        return contours, pdf_borders


    def _coords_to_kdes(self, all_xs, all_ys, xs, ys, inverse = False):
        xs_new, ys_new = [], []

        min_xs = min(all_xs) - 10
        max_xs = max(all_xs) + 10
        min_ys = min(all_ys) - 10
        max_ys = max(all_ys) + 10

        for x,y in zip(xs, ys):
            if not inverse:
                x = 50*(x - min_xs)/(max_xs - min_xs)
                y = 50*(y - min_ys)/(max_ys - min_ys)
            else:
                per_pixel_x = (max_xs-min_xs)/50
                per_pixel_y = (max_ys-min_ys)/50
                x = min_xs + per_pixel_x*x
                y = min_ys + per_pixel_y*y
            xs_new.append(x)
            ys_new.append(y)


        return xs_new, ys_new

    def _get_idx_contours(self, contours, all_xs, all_ys, xs, ys):

        xs, ys = self._coords_to_kdes(all_xs, all_ys, xs, ys)
        idx_closests = []
        for x,y in zip(xs, ys):
            point = np.array([x, y])
            dists_all = []
            for i in contours:
                dists_contour = np.linalg.norm(i-point, axis = 1)
                min_dist = np.min(dists_contour)
                dists_all.append(min_dist)
            idx_closest = dists_all.index(min(dists_all))
            idx_closests.append(idx_closest)

        return idx_closests



    def transition_matrix(self, s, b, grid):

        def _entropy(T):
            vals, vecs, _ = eig(T,left=True)
            for i,j in enumerate(vals):
                if abs(j.real-1) <  1e-6 and j.imag == 0:
                    idx_1 = i
                    break
            vec = vecs[:, idx_1]
            normalized = vec/sum(vec)


            total = 0
            for row in range(T.shape[0]):
                entropy = 0
                for col in range(T.shape[1]):
                    el = T[row, col]
                    if el > 0:
                        entropy += el*np.log2(el)
                entropy = -entropy

                total += normalized[row]*entropy

            print('total', total)





        pdf = copy.deepcopy(self.pdf_array)

        pdf[self.pdf_array<8e-5] = np.nan





        def get_tm(contours, sequences):
            T = [[0]*len(contours) for _ in range(len(contours))]
            for sequence in sequences:
                for (i,j) in zip(sequence,sequence[1:]):
                    T[i][j] += 1
            for row in T:
                n = sum(row)
                if n > 0:
                    row[:] = [f/sum(row) for f in row]
            T = np.array(T)
            return T


        fig_both = plt.figure()
        ax = fig_both.add_subplot(1, 3, 1)

        sequences = []
        for cfs in self.all_consecutive_frames:
            if grid:
                contours = []
                for i in np.linspace(0, 50, 6)[1:-1]:
                    for j in np.linspace(0, 50, 6)[1:-1]:
                        contours.append(np.array([[i, j]]))
                sequence = self._get_idx_contours(contours, list(self.all_embeddings[:, 0]), list(self.all_embeddings[:, 1]), list(cfs.embeddings[:, 0]), list(cfs.embeddings[:, 1]))

            else:
                contours, pdf_borders = self._get_contours(pdf, s = s, b = b)
                ax.imshow(pdf_borders[::-1, :])
                sequence = self._get_idx_contours(contours, list(self.all_embeddings[:, 0]), list(self.all_embeddings[:, 1]), list(cfs.embeddings[:, 0]), list(cfs.embeddings[:, 1]))
            sequences.append(sequence)

        ax = fig_both.add_subplot(1, 3, 2)
        T = get_tm(contours, sequences)
        _entropy(T)
        ax.imshow(T, cmap = 'Blues')
        ax = fig_both.add_subplot(1, 3, 3)
        sequences_no_duplicates = []
        for sequence in sequences:
            sequences_no_duplicates.append([key for key, grp in itertools.groupby(sequence)])
        T = get_tm(contours, sequences_no_duplicates)
        _entropy(T)
        ax.imshow(T, cmap = 'Blues')
        plt.show()




    def motif_hierarchies(self, filename1, filename2):

        pdf1_orig = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_kde.pickle'.format(filename1), 'rb'))
        pdf2_orig = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_kde.pickle'.format(filename2), 'rb'))
        pdf1, pdf2 = copy.deepcopy(pdf1_orig), copy.deepcopy(pdf2_orig)
        pdf1[pdf1<8e-5] = np.nan
        pdf2[pdf2<8e-5] = np.nan

        contours1, pdf_borders1 = self._get_contours(pdf1, s = thresh_params_dict[filename1][0], b = thresh_params_dict[filename1][1])

        contours2, pdf_borders2 = self._get_contours(pdf2, s = thresh_params_dict[filename2][0], b = thresh_params_dict[filename2][1])

        data1 = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_dots.pickle'.format(filename1), 'rb'))
        names1 = data1['names']
        xs1 = data1['xs']
        ys1 = data1['ys']
        data2 = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_dots.pickle'.format(filename2), 'rb'))
        names2  = data2['names']
        xs2  = data2['xs']
        ys2 = data2['ys']

        def remove_duplicate_names(names_temp, xs_temp, ys_temp):

            names, xs, ys = [], [], []
            for idx in range(len(names_temp)):
                if names_temp[idx] not in names:
                    names.append(names_temp[idx])
                    xs.append(xs_temp[idx])
                    ys.append(ys_temp[idx])
            return names, xs, ys

        names2, xs2, ys2 = remove_duplicate_names(names2, xs2, ys2)
        names1, xs1, ys1 = remove_duplicate_names(names1, xs1, ys1)





        idxs_keep = [i for i,j in enumerate(names1) if j in names2]
        names1 = [j for i,j in enumerate(names1) if i in idxs_keep]
        xs1 = [j for i,j in enumerate(xs1) if i in idxs_keep]
        ys1 = [j for i,j in enumerate(ys1) if i in idxs_keep]


        dict1 = {names1[idx]: (xs1[idx], ys1[idx]) for idx in range(len(names1))}
        dict2 = {names2[idx]: (xs2[idx], ys2[idx]) for idx in range(len(names2))}



        idx_contours1 = []
        idx_contours2 = []
        xs2_name_ordered = []
        ys2_name_ordered = []
        for name in dict1.keys():
            idx_contours1.append(self._get_idx_contours(contours1, xs1, ys1, [dict1[name][0]], [dict1[name][1]])[0])
            idx_contours2.append(self._get_idx_contours(contours2, xs1, ys1, [dict1[name][0]], [dict1[name][1]])[0])
            xs2_name_ordered.append(dict2[name][0])
            ys2_name_ordered.append(dict2[name][1])

        colors = ['black', 'green', 'red', 'blue', 'indigo', 'violet', 'grey', 'aqua', 'maroon']
        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.imshow(pdf1_orig[::-1, :])
        ax.axis('off')
        for idx, i in enumerate(contours1):
            i = np.vstack([i, i[0, :]])
            plt.plot(i[:, 0].flatten(), pdf1.shape[0]-1-i[:, 1].flatten(), c = colors[idx])

        ax = fig.add_subplot(223)

        for idx, i in enumerate(contours1):
            i = np.vstack([i, i[0, :]])
            to_plot_x, to_plot_y = self._coords_to_kdes(xs1, ys1, i[:, 0].flatten(), i[:,1].flatten(), inverse = True)
            plt.plot(to_plot_x, to_plot_y, c = colors[idx], linewidth = 3)
        ax.scatter(xs1, ys1, c = [colors[i] for i in idx_contours1], s = 5)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(222)
        ax.imshow(pdf2_orig[::-1, :])
        ax.axis('off')
        for idx, i in enumerate(contours2):
            i = np.vstack([i, i[0, :]])
            plt.plot(i[:, 0].flatten(), pdf2.shape[0]-1-i[:, 1].flatten(), c = colors[idx])
        ax = fig.add_subplot(224)

        for idx, i in enumerate(contours2):
            i = np.vstack([i, i[0, :]])
            to_plot_x, to_plot_y = self._coords_to_kdes(xs2_name_ordered, ys2_name_ordered , i[:, 0].flatten(), i[:,1].flatten(), inverse = True)
            plt.plot(to_plot_x, to_plot_y, c = 'black', linewidth = 3)
        ax.scatter(xs2_name_ordered, ys2_name_ordered, c = [colors[i] for i in idx_contours1], s = 5)
        ax.set_xticks([])
        ax.set_yticks([])

        for idx in range(len(idx_contours2)):
            if idx_contours2[idx] == 0:
                print(idx_contours1[idx])


        fig_matrix1 = plt.figure()
        ax = fig_matrix1.add_subplot(111)
        T = [[0]*len(contours1) for _ in range(len(contours2))]
        for (i,j) in zip(idx_contours2,idx_contours1):
            T[i][j] += 1
        for row in T:
            n = sum(row)
            if n > 0:
                row[:] = [f/sum(row) for f in row]
        T = np.array(T)
        ax.imshow(T)



        arr = np.zeros(shape = (4, *T.shape))
        arr[3, :, :] = T
        #arr[3, :, :] = np.full(arr.shape[1:], 1)
        for idx_col in range(T.shape[1]):
            for idx_row in range(T.shape[0]):
                arr[:3, idx_row, idx_col] = mpl_colors.to_rgba(colors[idx_col])[:-1]
        print(arr.shape)
        arr = np.swapaxes(arr, 1, 2)
        arr = np.swapaxes(arr, 0, 2)
        print(arr.shape)

        fig_matrix2 = plt.figure()
        ax = fig_matrix2.add_subplot(111)
        ax.imshow(arr)

        plt.show()


        data2 = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_dots.pickle'.format(filename2), 'rb'))
        xs2 = data2['xs']
        ys2 = data2['ys']
        names2 = data2['names']



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




    def run_power_spectrum(self, attribute_list, name, high_or_low_run):

        f_max = 0.04
        if attribute_list[:3] == 'pca':
            f_max = 0.02


        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        all_fs = []
        all_Ps = []
        for cfs in self.all_consecutive_frames:
            if name == 'all' or cfs.name == name:

                time_series = getattr(cfs, attribute_list)


                time_series_split = []
                new = []
                for i in time_series:
                    if not np.isnan(i):
                        new.append(i)
                    else:
                        time_series_split.append(new)
                        new = []
                    time_series_split.append(new)
                lengths = [len(i) for i in time_series_split]
                time_series = time_series_split[lengths.index(max(lengths))]



                #if len(time_series) > 50:
                if len(time_series) > 50 and cfs.high_or_low == high_or_low_run or high_or_low_run == 'all':
                    print(cfs.name)
                    p = ax1.plot([i*5 for i in range(len(time_series))], time_series)
                    ax1.set_xlim([0, 5*230])
                    ax1.set_ylim([-0.01, 0.04])
                    if attribute_list[:3] == 'pca':
                        ax1.set_ylim([-1, 1])

                    idxs_del = []
                    for idx in range(len(time_series)):
                        if np.isnan(time_series[idx]):
                            idxs_del.append(idx)
                        else:
                            break
                    for idx in reversed(range(-len(time_series), 0)):
                        if np.isnan(time_series[idx]):
                            idxs_del.append(len(time_series)+idx)

                        else:
                            break

                    time_series2 = [j for i,j in enumerate(time_series) if i not in idxs_del]


                    f, Pxx_den = signal.periodogram(time_series2, fs = 1/5, scaling = 'spectrum')
                    all_fs += list(f)
                    all_Ps += list(Pxx_den)
                    ax2.scatter(f, Pxx_den, c = p[0].get_color())
                    ax2.set_xlim([0, f_max])
                    ax2.set_ylim([0, 1.2e-5])
                    if attribute_list[:3] == 'pca':
                        ax2.set_ylim([0, 0.06])



        bins = np.linspace(0, f_max, 10)
        digitized = list(np.digitize(all_fs, bins).squeeze())

        means = []
        stds = []
        for bin in range(10):
            digitized_bin = [j for idx,j in enumerate(all_Ps) if digitized[idx] == bin]
            means.append(np.nanmean(digitized_bin))
            stds.append(np.nanstd(digitized_bin))
        ax3.bar(np.linspace(0, f_max, 10), means, width = bins[1]-bins[0])
        ax3.errorbar(np.linspace(0, f_max, 10), means, yerr = stds, ls = 'none', ecolor = 'red')
        ax3.set_xlim([0, f_max])
        ax3.set_ylim([0, 0.4e-5])
        if attribute_list[:3] == 'pca':
            ax3.set_ylim([0, 0.0125])

        plt.show()


    def longer_motifs(self):


        from matplotlib.patches import Rectangle

        pca0_all = []
        pca1_all = []
        pca2_all = []

        time_series = self.all_embeddings.T

        m = 100

        all_i = []
        all_j = []
        diffs = []
        print(time_series.shape[1])
        for i in np.arange(0, time_series.shape[1], 20):
            for j in np.arange(0, time_series.shape[1], 20):
                slice1 = time_series[:, i:i+m]
                slice2 = time_series[:, j:j+m]
                if slice1.shape[1] == m and slice2.shape[1] == m:
                    diff = np.linalg.norm(slice1-slice2)
                    if diff != 0:
                        diffs.append(diff)
                        all_i.append(i)
                        all_j.append(j)

        idx_min = diffs.index(min(diffs))
        print(all_i[idx_min], all_j[idx_min])

        plt.plot(diffs)
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.plot(time_series[0, :], c = 'red')
        plt.axvspan(all_i[idx_min], all_i[idx_min]+m, color='pink', alpha=0.5)
        ax = fig.add_subplot(223)
        ax.plot(time_series[1, :], c = 'blue')
        plt.axvspan(all_i[idx_min], all_i[idx_min]+m, color='pink', alpha=0.5)

        ax = fig.add_subplot(222)
        ax.plot(time_series[0, :], c = 'red')
        plt.axvspan(all_j[idx_min], all_j[idx_min]+m, color='purple', alpha=0.5)
        ax = fig.add_subplot(224)
        ax.plot(time_series[1, :], c = 'blue')
        plt.axvspan(all_j[idx_min], all_j[idx_min]+m, color='purple', alpha=0.5)

        plt.show()








def show_cell_series_clustered(idx_segments, center_frames):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    cwt = CWT(chop = None)
    cwt.set_spectograms()

    for idx_segment, center_frame in zip(idx_segments, center_frames):

        idx_cell, letter_keep = idx_segment[:-1], idx_segment[-1]
        cells = Cells(stack_attributes_2 + stack_attributes_3, cells_model = [idx_cell], max_l = 15)
        cells._set_centroid_attributes('searching', time_either_side = 7)
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
                if letter == letter_keep and abs(lymph.frame*lymph.t_res - center_frame*lymph.t_res) < time_either_side:
                    keep.append(lymph)
                    lymph_t_res = lymph.t_res
                prev_frame = lymph.frame
        cells.cells[idx_cell] = keep
        cells.plot_orig_series(idx_cell=idx_cell, uropod_align = False, color_by = None, plot_every = 1)



        cfs = [i for i in cwt.all_consecutive_frames if i.name == idx_segment][0]



        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2)
        ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        # CHANGED
        idxs_plot = [i for i,j in enumerate(cfs.closest_frames) if abs(j*lymph_t_res-center_frame*lymph_t_res) < time_either_side]

        ax1.plot([j for i,j in enumerate(cfs.closest_frames) if i in idxs_plot], [j for i,j in enumerate(cfs.pca0_list) if i in idxs_plot], color = 'red')
        ax1.plot([j for i,j in enumerate(cfs.closest_frames) if i in idxs_plot], [j for i,j in enumerate(cfs.pca1_list) if i in idxs_plot], color = 'blue')
        ax1.plot([j for i,j in enumerate(cfs.closest_frames) if i in idxs_plot], [j for i,j in enumerate(cfs.pca2_list) if i in idxs_plot], color = 'green')
        ax1.set_ylim([-1, 1])

        # show spectogram
        spect = cfs.spectogram[:, idxs_plot]

        vert = cfs.spectogram[:, cfs.closest_frames.index(center_frame)][:, None]
        # CHANGED
        for ins in inserts:
            empty = np.empty(shape = (len(idxs_plot),))
            spect = np.insert(spect, ins, empty.fill(np.nan), 0)
            empty = np.empty(shape = (1,))
            vert = np.insert(vert, ins, empty.fill(np.nan), 0)
        vert = np.vstack([vert.T]*4)

        # CHANGED
        #ax2.imshow(spect, cmap = 'PiYG', vmin = -0.6, vmax = 0.6)
        #ax3.imshow(vert, cmap = 'PiYG', vmin = -0.6, vmax = 0.6)
        ax2.imshow(spect, cmap = 'PiYG', vmin = -0.25, vmax = 0.25)
        ax3.imshow(vert, cmap = 'PiYG', vmin = -0.25, vmax = 0.25)
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








# CHANGED
cwt = CWT(idx_segment = 'all', chop = chop)




#show_cell_series_clustered(idx_segments = ['zm_3_4_1a', 'zm_3_3_5a'],
#                                center_frames = [7, 163])



"""
fig = plt.figure()
for attribute_list in ['run_list', 'pca0_list', 'pca1_list', 'pca2_list']:
    print(attribute_list)
    for high_or_low_run in ['all', 'low', 'high']:
        print(high_or_low_run)
        cwt.run_power_spectrum(attribute_list = attribute_list, name = 'all', high_or_low_run = high_or_low_run)
sys.exit()
"""


#cwt.print_freqs()

#cwt.edge_effect_size()


#cwt.motif_hierarchies('50', '150')
#sys.exit()

cwt.set_spectograms()



#cwt.plot_wavelet_series_spectogram(name = 'all')


cwt.set_tsne_embeddings(load_or_save_or_run = load_or_save_or_run, filename = filename)
#cwt.longer_motifs()
cwt.kde(load_or_save_or_run = load_or_save_or_run, filename = filename)

cwt.transition_matrix(s = None, b = None, grid = True)
#cwt.transition_matrix(s = thresh_params_dict[filename][0], b = thresh_params_dict[filename][1], grid = True)
sys.exit()
#['2_1a', 'zm_3_4_0a', 'zm_3_3_3a']
#['zm_3_3_5a', 'zm_3_3_2a', 'zm_3_3_4a', 'zm_3_4_1a']

for name in ['zm_3_3_5a', 'zm_3_3_2a', 'zm_3_3_4a', 'zm_3_4_1a']:
    #for cfs in cwt.all_consecutive_frames:
    cwt.plot_embeddings(load_or_save_or_run = load_or_save_or_run, filename = filename, path_of = name)

plt.show()
