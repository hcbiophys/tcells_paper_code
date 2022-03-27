

def _butter_highpass_filter(data, cutoff, fs, order=5):
    """
    Highpass time series filter
    Args:
    - data: the time series to be filtered
    - cutoff: the cutoff frequency
    - fs: the sampling frequency
    - order: order of the filter
    """

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y




def ACF(all_run_stop):
    """
    Compute the autocorrelation functions (ACFs)
    """


    if all_run_stop == 'run':
        all_consecutive_frames = pickle.load(open('../../data/time_series/shape_series_go.pickle',"rb"))
    elif all_run_stop == 'stop':
        all_consecutive_frames = pickle.load(open('../../data/time_series/shape_series_stop.pickle',"rb"))
    else:
        all_consecutive_frames = pickle.load(open('../../data/time_series/shape_series.pickle',"rb"))

    acfs = [[], [], [], []]
    colors = ['red', 'blue', 'green',  'black']


    def get_acf(time_series):
        acf = list(sm.tsa.acf(time_series, nlags = 99, missing = 'conservative'))
        acf += [np.nan for _ in range(100-len(acf))]
        return acf


    fig_series = plt.figure()
    ax = fig_series.add_subplot(111)
    for cfs in self.all_consecutive_frames:

        for idx_attribute, l in enumerate([cfs.pca0_list, cfs.pca1_list, cfs.pca2_list, cfs.speed_uropod_list]):
            if len(l) > 50:
                l = self.interpolate_list(l)
                l_new  = _butter_highpass_filter(l,1/400,fs=0.2)
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
                elif idx_attribute == 3: # if it's speed_uropod

                    if cfs.name[:-1] not in ['3_1_1', 'zm_3_1_1', 'zm_3_3_7', 'zm_3_4_0']:
                        acfs[idx_attribute].append(np.array(acf))
                    else:
                        print('Not adding speed_uropod')
                        acfs[idx_attribute].append([np.nan])


    fig_acf = plt.figure()
    ax = fig_acf.add_subplot(111)

    for idx1 in range(len(acfs)):
        for idx2 in range(len(acfs[idx1])):
            xs = [i*5 for i in range(len(acfs[idx1][idx2]))]
            ys = acfs[idx1][idx2]
            ax.plot(xs, ys, c = colors[idx1])
            ax.plot(xs, [0 for _ in xs], linewidth = 0.1, c = 'grey')


    taus = fit_exponentials(acfs)
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






def fit_exponentials(acfs):
    """
    Fit exponential decay models to the autocorrelation functions (ACFs) to get decay timescales
    """
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








def run_power_spectrum(attribute_list, idx_attribute):
    """
    Compute the power spectra
    """
    print('attribute_list', attribute_list)

    cfs_run = pickle.load(open('../data/cwt_saved/shape_series_go.pickle',"rb"))
    cfs_stop = pickle.load(open('../data/cwt_saved/shape_series_stop.pickle',"rb"))

    PC_uncertainties = pickle.load(open('../data/PC_uncertainties.pickle', 'rb'))
    for cfs in cfs_run:
        cfs.PC_uncertainties = PC_uncertainties[cfs.name[:-1]]
    for cfs in cfs_stop:
        cfs.PC_uncertainties = PC_uncertainties[cfs.name[:-1]]

    if attribute_list == 'speed_uropod_list':
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



#cwt.ACF()



"""
for idx_attribute, attribute_list in enumerate(['pca0_list', 'pca1_list', 'pca2_list', 'speed_uropod_list']):
    cwt.run_power_spectrum(attribute_list = attribute_list, idx_attribute = idx_attribute)
sys.exit()
"""
