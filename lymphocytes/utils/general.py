import numpy as np
import scipy
import matplotlib.pyplot as plt
import pyvista as pv
import random


def get_nestedList_connectedLymphs(lymph_series):
    """
    Return a nested list in which each sub list has no gaps in the frames
    """
    nestedLists = []
    frames = [lymph_series[0].frame]
    to_add = [lymph_series[0]]
    for lymph in lymph_series[1:]:
        if lymph.frame - frames[-1] != 1:
            nestedLists.append(to_add)
            to_add = [lymph]
        else:
            to_add.append(lymph)
        frames.append(lymph.frame)
    nestedLists.append(to_add)

    return nestedLists




def list_all_lymphs(cells):
    """
    Unpack and list all lymphs in one list
    """
    lymphs = []
    for lymph_series in cells.cells.values():
        lymphs += [lymph for lymph in lymph_series]
    return lymphs


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Get the rotation matrix to map one vector to another
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix



def get_color_lims(cells, color_by):
    """
    Get the min and max limits of a lymph attribute (color_by)
    """
    lymphs = list_all_lymphs(cells)
    scalars = [getattr(lymph, color_by) for lymph in lymphs if getattr(lymph, color_by) is not None]
    return min(scalars), max(scalars)

def get_color(lymph, color_by, vmin, vmax):
    """
    Get the color based on a lymph attribute (color_by)
    - vmin: minimum attribute value
    - vmax: maximum attribute value
    """
    cmap = plt.cm.PiYG
    norm = plt.Normalize(vmin, vmax)
    if getattr(lymph, color_by) is None:
        color = 'lightgrey'
    else:
        color = cmap(norm(getattr(lymph, color_by)))
    return color

def faces_from_phisThetas(phis, thetas):
    """
    Delauney triangulation to get the faces in the order of the phis, thetas
    """
    tri = scipy.spatial.Delaunay([(phis[idx], thetas[idx]) for idx in range(len(phis))])
    faces = tri.simplices
    faces = np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis = 1)
    faces = np.hstack(faces)
    return faces

def subsample_lists(freq, *args):
    """
    Subsample any list with subsample rate 'freq'
    """
    return [i[::freq] for i in args]





def del_whereNone(cells, attribute):
    """
    Remove lymph frames where a certain attribute is None
    """
    print('series PERMANENTLY EDITED')
    new_dict = {}
    for key, values in cells.items():
        new_values = []
        for lymph in values:
            if attribute == 'lymph':
                if lymph is not None:
                    new_values.append(lymph)
            if attribute == 'delta_centroid':
                if lymph.delta_centroid is not None:
                    new_values.append(lymph)
            elif attribute == 'angle':
                if lymph.angle is not None:
                    new_values.append(lymph)
        new_dict[key] = new_values
    return new_dict
