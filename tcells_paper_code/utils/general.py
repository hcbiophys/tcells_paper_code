import numpy as np
import scipy
import matplotlib.pyplot as plt


def split_by_consecutive_frames(video, attribute, and_nan = False):
    """
    Split up a cell video into lists of connected (for continuous sections of an attribute) frames. and_nan argument determines whether nan values are considered
    """

    all_lists = []
    new_list = [video[0]]
    prev_frame = video[0].idx_frame
    for frame in video[1:]:
        if frame.idx_frame - prev_frame == 1:
            new_list.append(frame)
        else:
            all_lists.append(new_list)
            new_list = [frame]
        prev_frame = frame.idx_frame
    all_lists.append(new_list)

    if and_nan:
        all_lists_2 = []

        for video in all_lists:
            idx_start = None
            for idx in range(len(video)):
                if getattr(video[idx], attribute) is not None and not np.isnan(getattr(video[idx], attribute)):
                    new_list = [video[idx]]
                    idx_start = idx
                    break
            if idx_start is not None:
                for frame in video[idx_start+1:]:
                    if getattr(frame, attribute) is not None and not np.isnan(getattr(frame, attribute)):
                        new_list.append(frame)
                    else:
                        all_lists_2.append(new_list)
                        new_list = []
                if len(new_list) != 0:
                    all_lists_2.append(new_list)

        all_lists = all_lists_2

    all_lists = [i for i in all_lists if len(i) != 0]
    return all_lists




def get_frame_dict(video):
    """
    self.cells[idx_cell] is a list of frames, ordered by frame
    this function returns a dict so frames can easily be acccessed by frame, like dict[idx_frame] = frames
    """
    idxs_frames = [frame.idx_frame for frame in video]
    dict = {}
    for idx_frame, frame in zip(idxs_frames, video):
        dict[idx_frame] = frame
    return dict

def get_nestedList_connectedframes(video):
    """
    Return a nested list in which each sub list has no gaps in the frames
    """
    nestedLists = []
    idxs_frames = [video[0].idx_frame]
    to_add = [video[0]]
    for frame in video[1:]:
        if frame.idx_frame - idxs_frames[-1] != 1:
            nestedLists.append(to_add)
            to_add = [frame]
        else:
            to_add.append(frame)
        idxs_frames.append(frame.idx_frame)
    nestedLists.append(to_add)

    return nestedLists




def list_all_frames(cells):
    """
    Unpack and list all frames in one list
    """
    frames = []
    for video in cells.cells.values():
        frames += [frame for frame in video]
    return frames


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
    Get the min and max limits of a frame attribute (color_by)
    """
    frames = list_all_frames(cells)
    scalars = [getattr(frame, color_by) for frame in frames if getattr(frame, color_by) is not None]


    return np.nanmin(scalars), np.nanmax(scalars)

def get_color(frame, color_by, vmin, vmax):
    """
    Get the color based on a frame attribute (color_by)
    - vmin: minimum attribute value
    - vmax: maximum attribute value
    """
    cmap = plt.cm.PiYG
    norm = plt.Normalize(vmin, vmax)
    if getattr(frame, color_by) is None:
        color = 'lightgrey'
    else:
        color = cmap(norm(getattr(frame, color_by)))
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
