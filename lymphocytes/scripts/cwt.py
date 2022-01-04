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


idx_cells =  ['2_{}'.format(i) for i in range(10)] + ['3_1_{}'.format(i) for i in range(6) if i != 0] + ['zm_3_3_{}'.format(i) for i in range(8)] + ['zm_3_4_{}'.format(i) for i in range(4)] + ['zm_3_5_2', 'zm_3_6_0'] + ['zm_3_5_1']
colors = [np.array([0.57151656, 0.32208642, 0.79325759]), np.array([0.67739055, 0.82755935, 0.77116142]), np.array([0.0952304 , 0.09013385, 0.98938936]), np.array([0.05764147, 0.98641696, 0.75908016]), np.array([0.97425911, 0.48333032, 0.17135435]), np.array([0.43114909, 0.2235878 , 0.8842425 ]), np.array([0.32933019, 0.5921141 , 0.61633489]), np.array([0.07315546, 0.44819796, 0.16833376]), np.array([0.01532791, 0.73857975, 0.69280004]), np.array([0.67843096, 0.6826372 , 0.08518478]), np.array([0.08110285, 0.79746762, 0.908427  ]), np.array([0.30928829, 0.32599009, 0.42407218]), np.array([0.60985161, 0.36160205, 0.35521415]), np.array([0.47062361, 0.25963724, 0.91398498]), np.array([0.00744883, 0.07700202, 0.16986398]), np.array([0.87592732, 0.75720082, 0.17710782]), np.array([0.59714551, 0.40399573, 0.12145515]), np.array([0.26211748, 0.57891925, 0.28847181]), np.array([0.47409021, 0.04009612, 0.37440976]), np.array([0.01394242, 0.40145539, 0.70053317]), np.array([0.28150027, 0.31116461, 0.84870038]), np.array([0.10455617, 0.91580071, 0.53926957]), np.array([0.79352826, 0.12960295, 0.81574088]), np.array([0.46107105, 0.02359315, 0.45115123]), np.array([0.87501311, 0.29718405, 0.75983003]), np.array([0.54075337, 0.33526137, 0.71694272]), np.array([0.75402239, 0.83224114, 0.72639337]), np.array([0.30155334, 0.83126122, 0.14805019]), np.array([0.99656294, 0.70101507, 0.83437361]), np.array([0.99656294, 0.70101507, 0])]
colors_dict = {i:j for i,j in zip(idx_cells, colors)}

# s,b
thresh_params_dict = { '150': (7, 20), '150_go':(17, 35)}



mexh_scales, gaus1_scales, chop, inserts, time_either_side, min_length = utils_cwt.get_params_from_filename(filename)






morB, morC = 10, 0.3
class CWT():

    def __init__(self, idx_segment = 'all', min_length = 15, chop = 5):
        self.chop = chop


        if filename[-2:] == 'go':
            self.all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series_go.pickle',"rb"))
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
            cfs.color = colors_dict[cfs.name[:-1]]



        self.spectograms = None
        self.all_embeddings = None
        self.all_consecutive_frames_dict = {cfs.name: cfs for cfs in self.all_consecutive_frames}

    def names_to_colors(self, names):
        names = [i.split('-')[0][:-1] for i in names]
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



    def fit_exponentials(self, acfs):
        def _linear_model(x,  k):
            x = np.array(x)
            return -k*x
        def _exp_model(x,  k):
            x = np.array(x)
            return np.exp(-k*x)

        taus = [[] for _ in acfs]
        for idx_attribute, acfs in enumerate(acfs):

            #fig = plt.figure()

            for acf in acfs:
                points_fit = []
                if not np.isnan(acf[0]):
                    print(idx_attribute)
                    xs = [i*5 for i,j in enumerate(acf)]
                    #plt.plot(xs, acf)

                    points_fit.append((xs[0], np.log(acf[0])))
                    for idx in range(1, len(acf)-2):
                        if  acf[idx] > 0 and acf[idx] > acf[idx-1] and acf[idx] > acf[idx-2] and acf[idx] > acf[idx+1] and acf[idx] > acf[idx+2]:
                            points_fit.append((xs[idx], np.log(acf[idx])))


                    xs_fit = [x for x,y in sorted(points_fit)]
                    ys_fit = [y for x,y in sorted(points_fit)]

                    p0 = (0.01)
                    #opt, pcov = curve_fit(_linear_model, xs_fit, ys_fit,  sigma = [0.27**y for y in ys_fit], absolute_sigma=True)
                    opt, pcov = curve_fit(_linear_model, xs_fit, ys_fit)

                    k = opt
                    xs_show = np.linspace(min([i[0] for i in points_fit]), max([i[0] for i in points_fit]), 5000)
                    #plt.plot(xs_show, _linear_model(xs_show,  k = k))
                    #plt.scatter([i[0] for i in points_fit], [i[1] for i in np.exp(points_fit)])
                    #plt.scatter([i[0] for i in points_fit], [i[1] for i in points_fit])
                    #plt.plot(xs_show, _exp_model(xs_show,  k = k))
                    tau = 1./k
                    taus[idx_attribute].append(tau[0])
                    #plt.show()
                else:
                    taus[idx_attribute].append(np.nan)

        return taus



    def ACF(self):


        acfs = [[], [], [], []]
        colors = ['red', 'blue', 'green',  'black']


        def get_acf(time_series):
            acf = list(sm.tsa.acf(time_series, nlags = 99, missing = 'conservative'))
            acf += [np.nan for _ in range(100-len(acf))]
            return acf


        fig_series = plt.figure()
        ax = fig_series.add_subplot(111)
        for cfs in self.all_consecutive_frames:

            for idx_attribute, l in enumerate([cfs.pca0_list, cfs.pca1_list, cfs.pca2_list, cfs.run_uropod_list]):
                if len(l) > 50:
                    l = self.interpolate_list(l)
                    l_new  = self.butter_highpass_filter(l,1/400,fs=0.2)
                    ax.plot([i*5 for i in range(len(l_new))], l_new, c = colors[idx_attribute])
                    acf = get_acf(l_new)

                    if idx_attribute < 3: # if it's a PC
                        PC_uncertainty = cfs.PC_uncertainties[idx_attribute]
                        signal_std = np.std(l_new)
                        SNR = signal_std/PC_uncertainty
                        print(cfs.name, 'attr:{}'.format(idx_attribute), 'signal_std', signal_std, 'PC_uncertainty', PC_uncertainty, 'SNR', SNR)

                        if SNR > 2.5:
                            acfs[idx_attribute].append(np.array(acf))
                        else:
                            print('Not adding')
                            acfs[idx_attribute].append([np.nan])
                    elif idx_attribute == 3: # if it's run_uropod


                        if cfs.name[:-1] not in ['3_1_1', 'zm_3_1_1', 'zm_3_3_7', 'zm_3_4_0']:
                            acfs[idx_attribute].append(np.array(acf))
                        else:
                            print('Not adding run_uropod')
                            acfs[idx_attribute].append([np.nan])


        fig_acf = plt.figure()
        ax = fig_acf.add_subplot(111)

        for idx1 in range(len(acfs)):
            for idx2 in range(len(acfs[idx1])):
                xs = [i*5 for i in range(len(acfs[idx1][idx2]))]
                ys = acfs[idx1][idx2]
                ax.plot(xs, ys, c = colors[idx1])
                ax.plot(xs, [0 for _ in xs], linewidth = 0.1, c = 'grey')


        taus = self.fit_exponentials(acfs)
        print('taus', taus)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        xs_bars = [1, 2, 3, 4]
        taus_stop = np.array([[94.95063674255715, np.nan, 219.89129766150137, 202.11024966716346],
        [119.67880972287459, 135.35817543105247, 255.30734761466599, 190.12325536488083],
        [209.6203003574912, np.nan, 180.47003089089924, 366.9536914504547],
        [108.37914216911506, np.nan, 202.96691138520023, 138.34128013699024]])

        for mode, shift, color in zip([taus, taus_stop], [-0.2, 0.2], ['red', 'blue']):
            ax.bar([i+shift for i in xs_bars], [np.nanmean(i) for i in mode], width=0.4, color = color, zorder = 0)
            for idx_attribute in range(len(taus)):
                ys = mode[idx_attribute]
                ax.scatter([idx_attribute + 1 + shift for _ in ys], ys, zorder = 1, c = 'black')

        plt.show()
        sys.exit()



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

    def COI(self):
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
            utils_cwt.move_sympyplot_to_axes(p_real, ax)
            utils_cwt.move_sympyplot_to_axes(p_im, ax)


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
            xs, ys, names = data['xs'], data['ys'], data['names']
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



        colors_2d = colors_run_uropod
        colors_3d = None



        return xs, ys, colors_2d, colors_3d


    def _plot_path_of(self, xs, ys, names, path_of, num_per_section = 40):

        cfs_xs, cfs_ys = [], []
        closest_frames = self.all_consecutive_frames_dict[path_of].closest_frames

        for idx in range(len(xs)):
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
                plt.text(cfs_xs[idx], cfs_ys[idx], str(chop*5 + 5*idx), fontsize = 12)


        plt.xlim(np.min(self.all_embeddings[:, 0])-10, np.max(self.all_embeddings[:, 0])+10)
        plt.ylim(np.min(self.all_embeddings[:, 1])-10, np.max(self.all_embeddings[:, 1])+10)





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


        xs, ys, colors_2d, colors_3d = self._set_embedding_colors(xs, ys)
        #colors_2d = self.names_to_colors(names)


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

        plt.xlim(np.min(self.all_embeddings[:, 0])-10, np.max(self.all_embeddings[:, 0])+10)
        plt.ylim(np.min(self.all_embeddings[:, 1])-10, np.max(self.all_embeddings[:, 1])+10)

        #from mpl_toolkits.axes_grid1 import make_axes_locatable
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes('right', size='5%', pad=0.05)
        #fig.colorbar(sc, cax=cax, orientation='vertical')
        #plt.show()




    def kde(self, load_or_save_or_run = 'load', filename = None, plot = True):


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
        if plot:
            plt.imshow(self.pdf_array[::-1, :], cmap = 'Reds')
            plt.show()
        return pdf_array


    def _get_contours(self, pdf, s, b):

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



    def transition_matrix(self, s, b, grid, stop_run_over_all = None, plot = True):

        if stop_run_over_all is not None:
            run_all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series_go.pickle',"rb"))
            run_names = []
            for cfs in run_all_consecutive_frames:
                run_names += [cfs.name + '-' + str(i) for i in cfs.closest_frames]


            stop_all_consecutive_frames = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series_stop.pickle',"rb"))
            stop_names = []
            for cfs in stop_all_consecutive_frames:
                stop_names += [cfs.name + '-' + str(i) for i in cfs.closest_frames]



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
            contours = self._get_contours(pdf, s = s, b = b)
            colors = ['black', 'magenta', 'darkgreen', 'red', 'blue', 'orange', 'blueviolet', 'grey', 'aqua', 'maroon', 'lime']
            for idx, contour in enumerate(contours):
                #ax.plot(contour[:, 0].flatten(), 49-contour[:,1].flatten(), c = 'black', linewidth = 3)
                if plot:
                    ax.plot(contour[:, 0].flatten(), 49-contour[:,1].flatten(), c = colors[idx], linewidth = 3)

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





    def run_power_spectrum(self, attribute_list, idx_attribute):
        print('attribute_list', attribute_list)

        cfs_run = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series_go.pickle',"rb"))
        cfs_stop = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/shape_series_stop.pickle',"rb"))

        PC_uncertainties = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/PC_uncertainties.pickle', 'rb'))
        for cfs in cfs_run:
            cfs.PC_uncertainties = PC_uncertainties[cfs.name[:-1]]
        for cfs in cfs_stop:
            cfs.PC_uncertainties = PC_uncertainties[cfs.name[:-1]]

        if attribute_list == 'run_uropod_list':
            cfs_stop = [cfs for cfs in cfs_stop if cfs.name[:-1] not in ['3_1_1', 'zm_3_1_1', 'zm_3_3_7', 'zm_3_4_0']]



        f_max = 0.02


        fig = plt.figure()

        for cfs_all, color in zip([cfs_run, cfs_stop], ['red', 'blue']):
            all_fs = []
            all_Ps = []
            for cfs in cfs_all:

                time_series = getattr(cfs, attribute_list)

                time_series = self.interpolate_list(time_series)

                if len(time_series) > 50:

                    idxs_del, time_series = utils_cwt.remove_border_nans(time_series)


                    if idx_attribute < 3: # if it's a PC
                        PC_uncertainty = cfs.PC_uncertainties[idx_attribute]
                        signal_std = np.std(time_series)
                        SNR = signal_std/PC_uncertainty
                        print(cfs.name, 'attr:{}'.format(idx_attribute), 'signal_std', signal_std, 'PC_uncertainty', PC_uncertainty, 'SNR', SNR)

                        if SNR < 2.5:
                            print('Removed')
                            break


                    f, Pxx_den = signal.periodogram(time_series, fs = 1/5, scaling = 'spectrum')
                    f, Pxx_den = f[1:] , Pxx_den[1:]
                    all_fs += list(f)
                    all_Ps += list(Pxx_den)
                    plt.scatter(f, np.log10(Pxx_den), c = color, zorder = 1, s = 2, label = cfs.name)
                    plt.xlim([0, f_max])



            bins = np.linspace(0, f_max, 10)
            digitized = list(np.digitize(all_fs, bins).squeeze())

            means = []
            stds = []
            for bin in range(10):
                digitized_bin = [j for idx,j in enumerate(all_Ps) if digitized[idx] == bin]
                means.append(np.nanmean(digitized_bin))
                stds.append(np.nanstd(digitized_bin))
            plt.plot(np.linspace(0, f_max, 10), np.log10(means), zorder = 0, c = color)
            #ax3.errorbar(np.linspace(0, f_max, 10), means, yerr = stds, ls = 'none', ecolor = 'red')
            #ax3.set_ylim([0, 0.4e-5])

            #plt.legend()
        plt.show()














def show_cell_series_clustered(codes):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    cwt = CWT(chop = None)
    cwt.set_spectograms()

    all_pc0_series = []
    all_pc1_series = []
    all_pc2_series = []

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

        pc0s = [j for i,j in enumerate(cfs.pca0_list) if i in idxs_plot]
        pc1s = [j for i,j in enumerate(cfs.pca1_list) if i in idxs_plot]
        pc2s = [j for i,j in enumerate(cfs.pca2_list) if i in idxs_plot]


        ax1.plot([5*i for i in list(range(len(pc0s)))], pc0s, color = 'red')
        ax1.plot([5*i for i in list(range(len(pc1s)))], pc1s, color = 'blue')
        ax1.plot([5*i for i in list(range(len(pc2s)))], pc2s, color = 'green')
        ax1.set_ylim([-1, 1])



        # show spectogram
        spect = cfs.spectogram[:, idxs_plot]
        print(center_frame)
        print(cfs.closest_frames)

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


def plot_local_morphodynamics(codes_all):
    fig = plt.figure(figsize = (20, 20))
    linestyles = ['-', '--', ':', '-.', ':']
    num_cols = (len(codes_all) // 4) + 1



    cwt = CWT(chop = None)
    cwt.set_spectograms()

    for idx_motif, codes in enumerate(codes_all):
        ax = fig.add_subplot(4, num_cols, idx_motif+1)

        all_pc0_series, all_pc1_series, all_pc2_series = [], [], []
        for code in codes:
            idx_segment = code.split('-')[0]
            center_frame = int(code.split('-')[1])
            idx_cell, letter_keep = idx_segment[:-1], idx_segment[-1]
            cfs = [i for i in cwt.all_consecutive_frames if i.name == idx_segment][0]
            idxs_plot = [i for i,j in enumerate(cfs.closest_frames) if abs((j-center_frame)*cfs.t_res_initial) < time_either_side]

            pc0s = [j for i,j in enumerate(cfs.pca0_list) if i in idxs_plot]
            pc1s = [j for i,j in enumerate(cfs.pca1_list) if i in idxs_plot]
            pc2s = [j for i,j in enumerate(cfs.pca2_list) if i in idxs_plot]
            all_pc0_series.append(pc0s)
            all_pc1_series.append(pc1s)
            all_pc2_series.append(pc2s)

        for idx in range(len(all_pc0_series)):
            ax.plot([5*i for i in range(len(all_pc0_series[idx]))], all_pc0_series[idx] - all_pc0_series[idx][len(all_pc0_series[idx])//2], c = 'red', linestyle = linestyles[idx], linewidth = 0.5)
            ax.plot([5*i for i in range(len(all_pc1_series[idx]))], all_pc1_series[idx] - all_pc1_series[idx][len(all_pc1_series[idx])//2], c = 'blue',  linestyle = linestyles[idx], linewidth = 0.5)
            ax.plot([5*i for i in range(len(all_pc2_series[idx]))], all_pc2_series[idx] - all_pc2_series[idx][len(all_pc2_series[idx])//2], c = 'green',  linestyle = linestyles[idx], linewidth = 0.5)

        if idx_motif%num_cols != 0:
            ax.set_yticks([])
    for ax in fig.axes:
        ax.set_ylim([-0.6, 0.6])
    plt.subplots_adjust(wspace = 0, hspace = 0)
    plt.savefig('/Users/harry/Desktop/local_morphodynamics.png')


cwt = CWT(idx_segment = 'all', chop = chop)
#cwt.COI()
#cwt.ACF()


# 'zm_3_3_5a-', 'zm_3_5_1a-', 'zm_3_3_4a-', 'zm_3_4_1a-'

#show_cell_series_clustered(['zm_3_4_1a-114'])




"""
plot_local_morphodynamics([['zm_3_3_3a-40', 'zm_3_4_1a-52', 'zm_3_3_5a-121'],
                            ['zm_3_3_5a-72', 'zm_3_4_1a-191', 'zm_3_3_4a-132'],
                            ['zm_3_5_2a-148', '3_1_3a-38', 'zm_3_3_6a-158'],
                            ['3_1_3a-61', 'zm_3_4_0a-96', '2_1a-95'],
                            ['zm_3_3_3a-152', 'zm_3_3_5a-28', 'zm_3_6_0a-176'],
                            ['zm_3_3_6a-185', '3_1_2b-31', 'zm_3_5_2a-82'],
                            ['zm_3_3_4a-26', '2_1a-41', '3_1_4a-29'],
                            ['zm_3_3_5a-42', 'zm_3_4_1a-26', 'zm_3_3_4a-110'],
                            ['zm_3_4_2a-26', 'zm_3_4_0a-35', 'zm_3_6_0a-118'],
                            ['zm_3_3_5a-136', 'zm_3_3_4a-55', 'zm_3_3_1b-65'],
                            ['zm_3_4_0a-76', 'zm_3_3_2a-126', '2_1a-71']])
sys.exit()

"""


plot_local_morphodynamics([['zm_3_4_1a-48', 'zm_3_3_5a-121', 'zm_3_3_5a-176'],
                            ['zm_3_3_4a-86', 'zm_3_3_5a-26', 'zm_3_3_4a-172'],
                            ['zm_3_3_4a-136', 'zm_3_3_5a-66', 'zm_3_4_1a-190'],
                            ['zm_3_5_1a-70','zm_3_5_1a-68',  'zm_3_3_5a-92'],
                            ['zm_3_3_4a-110', 'zm_3_3_5a-46',  'zm_3_4_1a-25'],
                            ['zm_3_3_4a-190', 'zm_3_3_4a-155', 'zm_3_4_1a-114']])
sys.exit()






"""
for idx_attribute, attribute_list in enumerate(['pca0_list', 'pca1_list', 'pca2_list', 'run_uropod_list']):
    cwt.run_power_spectrum(attribute_list = attribute_list, idx_attribute = idx_attribute)
sys.exit()
"""

#cwt.print_freqs()

#cwt.COI()


#cwt.motif_hierarchies('50', '150')
#sys.exit()
cwt.set_spectograms()



#cwt.plot_wavelet_series_spectogram(name = 'all')


cwt.set_tsne_embeddings(load_or_save_or_run = load_or_save_or_run, filename = filename)
cwt.kde(load_or_save_or_run = load_or_save_or_run, filename = filename, plot = True)
cwt.plot_embeddings(load_or_save_or_run = load_or_save_or_run, filename = filename, path_of = None)
plt.show()







#cwt.transition_matrix(s = None, b = None, grid = True, plot = True)
#cwt.transition_matrix(s = thresh_params_dict[filename][0], b = thresh_params_dict[filename][1], grid = False, stop_run_over_all = None)
#cwt.bar_entropy()



for cfs in cwt.all_consecutive_frames:
    print(cfs.name)
    cwt.plot_embeddings(load_or_save_or_run = load_or_save_or_run, filename = filename, path_of = cfs.name)
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
