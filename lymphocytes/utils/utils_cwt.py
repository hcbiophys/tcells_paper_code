import numpy as np





def coords_to_kdes(self, all_xs, all_ys, xs, ys, inverse = False):
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

def get_idx_contours(self, contours, all_xs, all_ys, xs, ys):

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


def remove_border_nans(time_series):

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




def get_params_from_filename(filename):

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

    return mexh_scales, gaus1_scales, chop, inserts, time_either_side, min_length
