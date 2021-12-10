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
from scipy import signal
import lymphocytes.utils.utils_cwt as utils_cwt
import lymphocytes.utils.general as utils_general


from lymphocytes.data.dataloader_good_segs_2 import stack_attributes_2
from lymphocytes.data.dataloader_good_segs_3 import stack_attributes_3
from lymphocytes.cells.cells_class import Cells


filename = sys.argv[1]
load_or_save_or_run = sys.argv[2]


# s,b
thresh_params_dict = { '150': (7, 20)}



mexh_scales, gaus1_scales, chop, inserts, time_either_side, min_length = utils_cwt.get_params_from_filename(filename)



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
        elif filename[-4:] == 'stop':
            self.all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series_stop.pickle',"rb"))
        else:
            self.all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series.pickle',"rb"))




        if not idx_segment == 'all':
            self.all_consecutive_frames = [i for i in self.all_consecutive_frames if i.name == idx_segment]

        idxs_keep = [i for i,j in enumerate(self.all_consecutive_frames) if len(j.pca0_list) > min_length]
        self.all_consecutive_frames = [j for i,j in enumerate(self.all_consecutive_frames) if i in idxs_keep]

        PC_uncertainties = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/PC_uncertainties.pickle', 'rb'))
        for cfs in self.all_consecutive_frames:
            cfs.PC_uncertainties = PC_uncertainties[cfs.name[:-1]]
            cfs.names_list = [cfs.name + '-' + str(i) for i in cfs.closest_frames]
            print(cfs.name, cfs.PC_uncertainties)



        self.spectograms = None
        self.all_embeddings = None
        self.all_consecutive_frames_dict = {cfs.name: cfs for cfs in self.all_consecutive_frames}

    def names_to_colors(self, names):
        names = [i.split('-')[0][:-1] for i in names]
        idx_cells =  ['2_{}'.format(i) for i in range(10)] + ['3_1_{}'.format(i) for i in range(6) if i != 0] + ['zm_3_3_{}'.format(i) for i in range(8)] + ['zm_3_4_{}'.format(i) for i in range(4)] + ['zm_3_5_2', 'zm_3_6_0']
        colors_dict = {i:np.random.rand(3,) for i in idx_cells}
        colors = []
        for name in names:
            colors.append(colors_dict[name])
        return colors





    def plot_series(self):
        fig = plt.figure()
        count = 0
        for cfs in self.all_consecutive_frames:
            if len(cfs.pca0_list) > 100 and count < 4:
                print(cfs.name)
                ax = fig.add_subplot(4, 1, count+1)
                for var_list, color in zip([cfs.pca0_list, cfs.pca1_list, cfs.pca2_list, cfs.run_uropod_list], ['red', 'blue', 'green', 'black']):
                    var_list = self.interpolate_list(var_list)
                    var_list  = self.butter_highpass_filter(var_list,1/400,fs=0.2)
                    if color == 'black':
                        var_list = [i*100 for i in var_list]
                    ax.plot([i*5 for i,j in enumerate(var_list)], var_list, c = color)
                    ax.set_title(cfs.name)
                    #ax.set_ylim([-1, 1])
                count += 1

        plt.show()
        plt.subplots_adjust(hspace = 0)
        sys.exit()


    def interpolate_list(self, l):
        if len([i for i in l if np.isnan(i)]) > 0: # if it contains nans
            f = interpolate.interp1d([i*5  for i,j in enumerate(l) if not np.isnan(j)], [j  for i,j in enumerate(l) if not np.isnan(j)])
            to_model = [i*5 for i in range(len(l))]
            idxs_del, _ = utils_cwt.remove_border_nans(l)
            to_model = [j for i,j in enumerate(to_model) if i not in idxs_del]

            l = f(to_model)
        return l



    def butter_highpass_filter(self, data, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        y = signal.filtfilt(b, a, data)
        return y



    def fit_exponentials(self, acfs2):
        def _linear_model(x,  k):
            x = np.array(x)
            return -k*x
        def _exp_model(x,  k):
            x = np.array(x)
            return np.exp(-k*x)

        taus = []
        for idx_attribute, acfs in enumerate(acfs2):

            #fig = plt.figure()
            #ax1 = fig.add_subplot(121)
            #ax2 = fig.add_subplot(122)

            points_fit = []
            for acf in acfs:
                xs = [i*5 for i,j in enumerate(acf)]
                #ax2.plot(xs, acf)

                points_fit.append((xs[0], np.log(acf[0])))
                for idx in range(1, len(acf)-2):
                    if  acf[idx] > 0 and acf[idx] > acf[idx-1] and acf[idx] > acf[idx-2] and acf[idx] > acf[idx+1] and acf[idx] > acf[idx+2]:
                        points_fit.append((xs[idx], np.log(acf[idx])))

            xs_fit = [x for x,y in sorted(points_fit)]
            ys_fit = [y for x,y in sorted(points_fit)]
            #ax1.scatter([i[0] for i in points_fit], [i[1] for i in points_fit])


            p0 = (0.01)
            #opt, pcov = curve_fit(_linear_model, xs_fit, ys_fit,  sigma = [0.27**y for y in ys_fit], absolute_sigma=True)
            opt, pcov = curve_fit(_linear_model, xs_fit, ys_fit)



            k = opt
            xs_show = np.linspace(min([i[0] for i in points_fit]), max([i[0] for i in points_fit]), 5000)
            #ax1.plot(xs_show, _linear_model(xs_show,  k = k))

            #ax2.scatter([i[0] for i in points_fit], [np.exp(i[1]) for i in points_fit])
            #ax2.plot(xs_show, _exp_model(xs_show,  k = k))
            tau = 1./k
            taus.append(tau[0])
            print('idx_attribute', idx_attribute, 'tau', tau)
            #plt.show()

        return taus



    def ACF(self):


        acfs1 = [[], [], [], []]
        acfs2 = [[], [], [], []]
        colors = ['red', 'blue', 'green',  'black']


        def get_acf(time_series):
            acf = list(sm.tsa.acf(time_series, nlags = 99, missing = 'conservative')) # LOOK INTO THIS FOR PURE SINE WAVE
            acf += [np.nan for _ in range(100-len(acf))]
            return acf


        fig_series = plt.figure()
        ax1 = fig_series.add_subplot(211)
        ax2 = fig_series.add_subplot(212)
        SNRs = [[], [], []] # each list is for a PC
        for cfs in self.all_consecutive_frames:

                for idx_attribute, l in enumerate([cfs.pca0_list, cfs.pca1_list, cfs.pca2_list, cfs.run_uropod_list]):

                    if len(l) > 50:
                        acf = get_acf(l)
                        acfs1[idx_attribute].append(np.array(acf))
                        #ax1.plot([i*5 for i in range(len(l))], [i*(1/np.nanmax(l)) for i in l], c = colors[idx_attribute])
                        ax1.plot([i*5 for i in range(len(l))], l, c = colors[idx_attribute])

                        l = self.interpolate_list(l)
                        l_new  = self.butter_highpass_filter(l,1/400,fs=0.2)

                        #ax2.plot([i*5 for i in range(len(l_new))], [i*(1/np.nanmax(l_new)) for i in l_new], c = colors[idx])
                        ax2.plot([i*5 for i in range(len(l_new))], l_new, c = colors[idx_attribute])
                        acf = get_acf(l_new)
                        acfs2[idx_attribute].append(np.array(acf))

                        if idx_attribute < 3: # if it's a PC
                            PC_uncertainty = cfs.PC_uncertainties[idx_attribute]
                            signal_var = np.std(l_new)
                            #print('signal_var', signal_var, 'PC_uncertainty', PC_uncertainty)
                            SNR = signal_var/PC_uncertainty
                            SNRs[idx_attribute].append(SNR)



        print('SNRs', SNRs)
        fig_acf = plt.figure()
        ax1 = fig_acf.add_subplot(211)
        ax2 = fig_acf.add_subplot(212)

        for idx1 in range(len(acfs1)):
            for idx2 in range(len(acfs1[idx1])):
                xs = [i*5 for i in range(len(acfs1[idx1][idx2]))]
                ys = acfs1[idx1][idx2]
                ax1.plot(xs, ys, c = colors[idx1])


                xs = [i*5 for i in range(len(acfs2[idx1][idx2]))]
                ys = acfs2[idx1][idx2]
                ax2.plot(xs, ys, c = colors[idx1])

        taus = self.fit_exponentials(acfs2)
        print('taus', taus)

        fig_taus = plt.figure()
        ax_taus = fig_taus.add_subplot(111)
        xs_bars = [1, 2, 3, 4]
        taus_stop = [125.09645584939082, 151.2182899779405, 223.22233724328925, 127.652516621403]
        ax_taus.bar([i+0.2 for i in xs_bars], taus_stop, width=0.4, color = 'blue')
        ax_taus.bar([i-0.2 for i in xs_bars], taus, width=0.4, color = 'red')



        ax1.plot(xs, [0 for _ in xs], linewidth = 0.1, c = 'grey')
        ax2.plot(xs, [0 for _ in xs], linewidth = 0.1, c = 'grey')
        plt.show()




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

    def edge_effect_size(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        fake_series = [100 for i in range(40)]
        coef, _ = pywt.cwt(fake_series, mexh_scales, 'mexh')

        ax1.imshow(coef)
        ax1.set_title('mexh')
        coef, _ = pywt.cwt(fake_series, gaus1_scales, 'gaus1')
        ax2.imshow(coef)
        ax2.set_title('gaus1')
        plt.show()
        sys.exit()


    def _plot_wavelets(self, wavelet, scales, frames_plot = 'all'):

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


        for consecutive_frames in self.all_consecutive_frames:
            spectogram = []
            for idx_attribute, attribute in enumerate(features):

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
                consecutive_frames.run_uropod_list = consecutive_frames.run_uropod_list[self.chop :-self.chop]
                consecutive_frames.run_uropod_running_mean_list = consecutive_frames.run_uropod_running_mean_list[self.chop :-self.chop]
                consecutive_frames.turning_list = consecutive_frames.turning_list[self.chop :-self.chop]
                consecutive_frames.names_list = consecutive_frames.names_list[self.chop :-self.chop]




            consecutive_frames.spectogram = spectogram

    def _plot_spectogram(self, spectogram):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.imshow(spectogram, vmin = -0.25, vmax = 0.25)

        for ins in inserts:
            spectogram = np.insert(spectogram, ins, np.zeros(shape = (spectogram.shape[1],)), 0)

        ax.imshow(spectogram, cmap = 'PiYG', vmin = -0.25, vmax = 0.25)
        ax.axis('off')


    def plot_wavelet_series_spectogram(self, name = 'all'):
        for cfs in self.all_consecutive_frames:
            if name == 'all' or cfs.name == name:

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
            colors, xs, ys, names = data['colors'], data['xs'], data['ys'], data['names']
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


        colors_pc = []
        colors_run_uropod = []
        colors_mode = []
        colors_turning = []
        for cfs in self.all_consecutive_frames:

            colors_run_uropod += list(cfs.run_uropod_list)
            colors_pc += list(cfs.pca0_list)
            colors_turning += list(cfs.turning_list)

            new_colors_mode = []
            for i in cfs.run_uropod_running_mean_list:
                if i > 0.005:
                    new_colors_mode.append('red')
                elif i < 0.002:
                    new_colors_mode.append('blue')
                else:
                    new_colors_mode.append('grey')
            colors_mode += new_colors_mode



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
            xs = data['xs']
            ys = data['ys']
            names = data['names']


        elif load_or_save_or_run == 'save' or load_or_save_or_run == 'run':
            data = {}
            xs, ys, zs, names = [], [],  [], []
            for consecutive_frames in self.all_consecutive_frames:
                xs += list(consecutive_frames.embeddings[:, 0])
                ys += list(consecutive_frames.embeddings[:, 1])
                names +=  list(consecutive_frames.names_list)

            data['xs'] = xs
            data['ys'] = ys
            data['names'] = names
            if load_or_save_or_run == 'save':
                pickle.dump(data, open('/Users/harry/OneDrive - Imperial College London/lymphocytes/{}_dots.pickle'.format(filename), 'wb'))


        #all = np.concatenate([i.spectogram for i in self.all_consecutive_frames], axis = 1)
        #colors = np.max(abs(all), axis = 0)


        #xs, ys, colors_2d, colors_3d = self._set_embedding_colors(xs, ys)
        colors_2d = self.names_to_colors(names)


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



    def kde(self, load_or_save_or_run = 'load', filename = None):


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
        return pdf_array


    def _get_contours(self, pdf, s, b):

        if s is None and b is None:
            plt.imshow(pdf[::-1, :])
            plt.show()
            s_list = [ 5, 7, 9, 11, 13, 15, ] # ROWS
            b_list = [20, 25, 30, 35, 40] # COLUMNS
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
                    ax.imshow(pdf[::-1, :])
                    contours = self.clean_contours(contours)


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
        contours = self.clean_contours(contours)



        return contours


    def clean_contours(self, contours):
        contours = [np.squeeze(i) for i in contours]
        contours_new = []
        for idx, i in enumerate(contours):
            if len(i.shape) == 1:
                i = np.expand_dims(i, axis=0)
            i = np.vstack([i, i[0, :]])
            contours_new.append(i)

        contours_new = [c for c in contours_new if cv2.contourArea(c) > 5]

        return contours_new



    def transition_matrix(self, s, b, grid, stop_run_over_all = None):
        if stop_run_over_all is not None:
            run_all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series_run.pickle',"rb"))
            run_names = []
            for cfs in run_all_consecutive_frames:
                run_names += [cfs.name + '-' + str(i) for i in cfs.closest_frames]


            stop_all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series_stop.pickle',"rb"))
            stop_names = []
            for cfs in stop_all_consecutive_frames:
                stop_names += [cfs.name + '-' + str(i) for i in cfs.closest_frames]



        pdf = copy.deepcopy(self.pdf_array)

        pdf[self.pdf_array<3e-5] = np.nan


        fig_both = plt.figure()
        ax = fig_both.add_subplot(1, 3, 1)
        ax.imshow(pdf[::-1, :], cmap = 'Reds')


        if grid:
            contours = []
            for i in np.linspace(0, 50, 6)[1:-1]:
                for j in np.linspace(0, 50, 6)[1:-1]:
                    contours.append(np.array([[i, j]]))
        else:
            contours = self._get_contours(pdf, s = s, b = b)
            for contour in contours:
                ax.plot(contour[:, 0].flatten(), 49-contour[:,1].flatten(), c = 'black', linewidth = 3)

        sequences = []
        for cfs in self.all_consecutive_frames:
            xs = list(cfs.embeddings[:, 0])
            ys = list(cfs.embeddings[:, 1])
            if stop_run_over_all == 'stop':
                idxs_keep = [i for i in range(len(xs)) if cfs.names_list[i] in stop_names]
                xs = [j for i,j in enumerate(xs) if i in idxs_keep]
                ys = [j for i,j in enumerate(ys) if i in idxs_keep]

            elif stop_run_over_all == 'run':
                idxs_keep = [i for i in range(len(xs)) if cfs.names_list[i] in run_names]
                xs = [j for i,j in enumerate(xs) if i in idxs_keep]
                ys = [j for i,j in enumerate(ys) if i in idxs_keep]

            sequence = utils_cwt.get_idx_contours(contours, list(self.all_embeddings[:, 0]), list(self.all_embeddings[:, 1]), xs, ys)

            sequences.append(sequence)

        ax = fig_both.add_subplot(1, 3, 2)
        T = utils_cwt.get_tm(contours, sequences)
        utils_cwt.entropy(T)
        ax.imshow(T, cmap = 'Blues')
        ax = fig_both.add_subplot(1, 3, 3)
        sequences_no_duplicates = []
        for sequence in sequences:
            sequences_no_duplicates.append([key for key, grp in itertools.groupby(sequence)])
        T = utils_cwt.get_tm(contours, sequences_no_duplicates)
        utils_cwt.entropy(T)
        ax.imshow(T, cmap = 'Blues')
        plt.show()





    def run_power_spectrum(self, attribute_list):
        print('attribute_list', attribute_list)

        cfs_run = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series_run.pickle',"rb"))
        cfs_stop = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series_stop.pickle',"rb"))


        f_max = 0.02


        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        for cfs_all, color in zip([cfs_run, cfs_stop], ['red', 'blue']):
            all_fs = []
            all_Ps = []
            for cfs in cfs_all:

                time_series = getattr(cfs, attribute_list)

                time_series = self.interpolate_list(time_series)

                if len(time_series) > 50:

                    idxs_del, time_series = utils_cwt.remove_border_nans(time_series)

                    p = ax1.plot([i*5 for i in range(len(time_series))], time_series, c = color)
                    ax1.set_xlim([0, 5*230])
                    ax1.set_ylim([-0.01, 0.04])
                    if attribute_list[:3] == 'pca':
                        ax1.set_ylim([-1, 1])

                    f, Pxx_den = signal.periodogram(time_series, fs = 1/5, scaling = 'spectrum')
                    f, Pxx_den = f[1:] , Pxx_den[1:]
                    all_fs += list(f)
                    all_Ps += list(Pxx_den)
                    ax2.scatter(f, np.log10(Pxx_den), c = p[0].get_color(), zorder = 1, s = 2, label = cfs.name)
                    ax2.set_xlim([0, f_max])
                    #ax2.set_ylim([0, 1.2e-5])
                    #if attribute_list[:3] == 'pca':
                    #    ax2.set_ylim([0, 0.06])

            bins = np.linspace(0, f_max, 10)
            digitized = list(np.digitize(all_fs, bins).squeeze())

            means = []
            stds = []
            for bin in range(10):
                digitized_bin = [j for idx,j in enumerate(all_Ps) if digitized[idx] == bin]
                means.append(np.nanmean(digitized_bin))
                stds.append(np.nanstd(digitized_bin))
            ax2.plot(np.linspace(0, f_max, 10), np.log10(means), zorder = 0, c = color)
            #ax3.errorbar(np.linspace(0, f_max, 10), means, yerr = stds, ls = 'none', ecolor = 'red')
            #ax3.set_ylim([0, 0.4e-5])

            #plt.legend()
        plt.show()














def show_cell_series_clustered(codes):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    cwt = CWT(chop = None)
    cwt.set_spectograms()


    for code in codes:
        idx_segment = code.split('-')[0]
        center_frame = int(code.split('-')[1])

        idx_cell, letter_keep = idx_segment[:-1], idx_segment[-1]
        cells = Cells(stack_attributes_2 + stack_attributes_3, cells_model = [idx_cell], max_l = 15, uropods_bool = True)
        lymph_series = cells.cells[idx_cell]
        lymph_t_res = lymph_series[0].t_res

        keep = []

        for lymph in lymph_series:
            if  abs((lymph.frame-center_frame)*lymph_t_res) < time_either_side:
                keep.append(lymph)

        cells.cells[idx_cell] = keep
        plot_every = int(len(keep)/4)
        cells.plot_orig_series(idx_cell=idx_cell, uropod_align = False, color_by = None, plot_every = plot_every, plot_flat = True)



        cfs = [i for i in cwt.all_consecutive_frames if i.name == idx_segment][0]



        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2)
        ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        idxs_plot = [i for i,j in enumerate(cfs.closest_frames) if abs((j-center_frame)*lymph_t_res) < time_either_side]

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


        ax2.imshow(spect, cmap = 'PiYG', vmin = -0.25, vmax = 0.25)
        ax3.imshow(vert, cmap = 'PiYG', vmin = -0.25, vmax = 0.25)

        plt.show()



cwt = CWT(idx_segment = 'all', chop = chop)
#cwt.ACF()



#show_cell_series_clustered(codes = ['3_1_4a-25', '3_1_3a-48', 'zm_3_3_4a-24'])




#for attribute_list in ['pca0_list', 'pca1_list', 'pca2_list', 'run_uropod_list']:
#    cwt.run_power_spectrum(attribute_list = attribute_list)
#sys.exit()


#cwt.print_freqs()

#cwt.edge_effect_size()


#cwt.motif_hierarchies('50', '150')
#sys.exit()

cwt.set_spectograms()



#cwt.plot_wavelet_series_spectogram(name = 'all')


cwt.set_tsne_embeddings(load_or_save_or_run = load_or_save_or_run, filename = filename)
cwt.kde(load_or_save_or_run = load_or_save_or_run, filename = filename)
cwt.plot_embeddings(load_or_save_or_run = load_or_save_or_run, filename = filename, path_of = None)
#cwt.longer_motifs()





#cwt.transition_matrix(s = None, b = None, grid = False)
cwt.transition_matrix(s = thresh_params_dict[filename][0], b = thresh_params_dict[filename][1], grid = False, stop_run_over_all = 'run')


#for name in ['zm_3_3_5a', 'zm_3_3_2a', 'zm_3_3_4a', 'zm_3_4_1a']:
#for cfs in cwt.all_consecutive_frames:
#    cwt.plot_embeddings(load_or_save_or_run = load_or_save_or_run, filename = filename, path_of = cfs.name)

plt.show()















































"""
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
        idx_contours1.append(utils_cwt.get_idx_contours(contours1, xs1, ys1, [dict1[name][0]], [dict1[name][1]])[0])
        idx_contours2.append(utils_cwt.get_idx_contours(contours2, xs1, ys1, [dict1[name][0]], [dict1[name][1]])[0])
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
        to_plot_x, to_plot_y = utils_cwt.coords_to_kdes(xs1, ys1, i[:, 0].flatten(), i[:,1].flatten(), inverse = True)
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
        to_plot_x, to_plot_y = utils_cwt.coords_to_kdes(xs2_name_ordered, ys2_name_ordered , i[:, 0].flatten(), i[:,1].flatten(), inverse = True)
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
"""






"""
def longer_motifs(self, plot_full = False):


    from matplotlib.patches import Rectangle

    pca0_all = []
    pca1_all = []
    pca2_all = []

    time_series = []

    for cfs in self.all_consecutive_frames:
        time_series.append(cfs.embeddings)
        time_series.append(np.array([[np.nan, np.nan]]))
    time_series = np.concatenate(time_series, axis = 0).T


    m = 50

    i_j_pairs = []
    diffs = []
    for i in np.arange(0, time_series.shape[1], 1):
        for j in np.arange(0, time_series.shape[1], 1):
            slice1 = time_series[:, i:i+m]
            slice2 = time_series[:, j:j+m]
            if slice1.shape[1] == m and slice2.shape[1] == m:
                #diff = np.linalg.norm(slice1-slice2)
                diff = np.sqrt(np.mean(np.square(slice1-slice2))) # RMSD like in Andre's paper
                if diff != 0 and not np.isnan(diff) and abs(i-j) > 20:
                    diffs.append(diff)
                    i_j_pairs.append((i,j))
                    #LOOK INTO THE SIMILARITY METRIC - IT IS COMPARING THE WRONG THINGS




    ymin, ymax = np.nanmin(time_series), np.nanmax(time_series)

    while len(diffs) > 0:
        idx_min = diffs.index(min(diffs))
        i, j = i_j_pairs[idx_min][0], i_j_pairs[idx_min][1]
        print(i, j, diffs[idx_min])

        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.set_ylim([ymin, ymax])
        ax.set_title('i')
        if plot_full:
            ax.plot(time_series[0, :], c = 'red')
            plt.axvspan(i, i+m, color='pink', alpha=0.5)
        else:
            ax.plot(time_series[0, i:i+m], c = 'red')

        ax = fig.add_subplot(223)
        ax.set_ylim([ymin, ymax])
        if plot_full:
            ax.plot(time_series[1, :], c = 'red')
            plt.axvspan(i, i+m, color='pink', alpha=0.5)
        else:
            ax.plot(time_series[1, i:i+m], c = 'blue')


        ax = fig.add_subplot(222)
        ax.set_ylim([ymin, ymax])
        if plot_full:
            ax.plot(time_series[0, :], c = 'red')
            plt.axvspan(j, j+m, color='purple', alpha=0.5)
        else:
            ax.plot(time_series[0, j:j+m], c = 'red')

        ax.set_title('j')
        ax = fig.add_subplot(224)
        ax.set_ylim([ymin, ymax])
        if plot_full:
            ax.plot(time_series[1, :], c = 'blue')
            plt.axvspan(j, j+m, color='purple', alpha=0.5)
        else:
            ax.plot(time_series[1, j:j+m], c = 'blue')


        plt.show()


        for p in range(i-10,i+10):
            for q in range(j-10, j+10):
                if (p,q) in i_j_pairs:
                    idx1 = i_j_pairs.index((p,q))
                    del diffs[idx1]
                    del i_j_pairs[idx1]
                if (q,p) in i_j_pairs:
                    idx2 = i_j_pairs.index((q,p))
                    del diffs[idx2]
                    del i_j_pairs[idx2]

"""
