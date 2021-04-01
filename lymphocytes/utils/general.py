import numpy as np
import scipy
import matplotlib.pyplot as plt
import pyvista as pv
import random

"""
def get_idxs_l_in_vector(vector, max_l):
    dict = {}
    for l in np.arange(0, max_l + 1):
        dict[l] = []
        for m in np.arange(0, l+1):

            if m == 0:
                dict[l].append(l*l)
            else:
                dict[l].append(l*l + 2*m - 1)
                dict[l].append(l*l + 2*m)
    return dict
"""




def get_nestedList_connectedFrames(lymph_series, attribute):
    nestedLists = []
    frames = [lymph_series[0].frame]
    to_add = [getattr(lymph_series[0], attribute)]
    for lymph in lymph_series[1:]:

        if lymph.frame - frames[-1] != 1:
            nestedLists.append(to_add)
            to_add = [getattr(lymph, attribute)]
        else:
            to_add.append(getattr(lymph, attribute))
        frames.append(lymph.frame)
    nestedLists.append(to_add)

    return nestedLists

#def shuffle_two_lists(list1, list2):
#    c = list(zip(list1, list2))
#    random.shuffle(c)
#    list1, list2 = zip(*c)
#    return list1, list2



def list_all_lymphs(lymph_serieses):
    lymphs = []
    for lymph_series in lymph_serieses.lymph_serieses.values():
        lymphs += [lymph for lymph in lymph_series if lymph is not None]
    return lymphs


def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix



def get_color_lims(lymph_serieses, color_by):
    lymphs = list_all_lymphs(lymph_serieses)
    if color_by == 'speed':
        scalars = [lymph.speed for lymph in lymphs if lymph is not None and lymph.speed is not None]
    return min(scalars), max(scalars)

def get_color(lymph, color_by, vmin, vmax):
    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin, vmax)

    if color_by == 'speed':
        if lymph.speed is None:
            color = 'red'
        else:
            color = cmap(norm(lymph.speed))
    elif color_by == 'angle':
        if lymph.angle is None:
            color = 'red'
        else:
            color = cmap(norm(lymph.angle))
    return color

def faces_from_phisThetas(phis, thetas):
    tri = scipy.spatial.Delaunay([(phis[idx], thetas[idx]) for idx in range(len(phis))])
    faces = tri.simplices
    faces = np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis = 1)
    faces = np.hstack(faces)
    return faces

def subsample_lists(freq, *args):
    return [i[::freq] for i in args]


def decimate_mat_voxels(mat_filename, idx_snap, decimation_factor, show, save_as):

    f = h5py.File(mat_filename, 'r')
    dataset = f['DataOut/Surf']
    voxels = f[dataset[2, idx_snap]]

    voxels = zoom(voxels, (decimation_factor, decimation_factor, decimation_factor), order = 0).astype('double')


    scipy.io.savemat(save_as, mdict={'bim': voxels,
                                    'origin': np.array([0, 0, 0]).astype('double'),
                                    'vxsize': np.array([1, 1, 1]).astype('double')})




def del_whereNone(lymph_serieses, attribute):

    print('series PERMANENTLY EDITED')

    new_dict = {}
    for key, values in lymph_serieses.items():
        new_values = []
        for lymph in values:
            if attribute == 'lymph':
                if lymph is not None:
                    new_values.append(lymph)
            if attribute == 'speed':
                if lymph.speed is not None:
                    new_values.append(lymph)
            elif attribute == 'angle':
                if lymph.angle is not None:
                    new_values.append(lymph)
        new_dict[key] = new_values


    return new_dict
