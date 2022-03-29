import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

from scipy import interpolate


def _interpolate_list(l):
    """
    Linearly interpolate list l
    """

    if len([i for i in l if np.isnan(i)]) > 0: # if it contains nans
        f = interpolate.interp1d([i*5  for i,j in enumerate(l) if not np.isnan(j)], [j  for i,j in enumerate(l) if not np.isnan(j)])
        to_model = [i*5 for i in range(len(l))]
        idxs_del, _ = remove_border_nans(l)
        to_model = [j for i,j in enumerate(to_model) if i not in idxs_del]
        l = f(to_model)
    return l

def coords_to_kdes(all_xs, all_ys, xs, ys, inverse = False):
    """
    Convert morphodynamic space (found from t-SNE) coordinates to indices (row, column) of the kernel density estimate (KDE) grid
    """
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

def get_idx_contours(contours, all_xs, all_ys, xs, ys):
    """
    Get the indices of the contours on a trajectory of {xs, ys} (where index is the closest contour)
    """

    xs, ys = coords_to_kdes(all_xs, all_ys, xs, ys)
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



def get_tm(contours, sequences):
    """
    Comput the transition matrix (tm) given the contours and trajectories as contour indices (i.e. sequences)
    """

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


def remove_border_nans(time_series):
    """
    Remove nan values at the beginning and end of a time series
    """

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

    return idxs_del, time_series2


def move_sympyplot_to_axes(p, ax):
    backend = p.backend(p)
    backend.ax = ax
    backend._process_series(backend.parent._series, ax, backend.parent)
    backend.ax.spines['right'].set_color('none')
    backend.ax.spines['bottom'].set_position('zero')
    backend.ax.spines['top'].set_color('none')
    plt.close(backend.fig)



def entropy(T):
    """
    Calculate the Markov chain entropy of a transition matrix, T
    """
    vals, vecs, _ = eig(T,left=True)
    idx_1 = np.nan
    for i,j in enumerate(vals):
        if abs(j.real-1) <  1e-6 and j.imag == 0:
            idx_1 = i
            break
    if np.isnan(idx_1):
        print('-'*300)
        print('NAN HERE')
        return np.nan
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
    return total
    print('total', total)
