import numpy as np
import os
import h5py
import sys
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
import nrrd
pv.set_plot_theme("document")



def _modify_faces(faces):
    faces = faces.astype(np.intc).T
    faces = np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis = 1).flatten()
    return faces

def get_attribute_from_mat(mat_filename, zeiss_type, idx_cell = None):
    """
    Returns attributes: frames,  vertices, faces
    Args:
    - mat_filename: filename for the .mat file with surface segmentation information
    - zeiss_type: just a label for the microscope batch; some have indexing beginning at 0 rather than 1
    """

    f = h5py.File(mat_filename, 'r')

    frames_all, vertices_all, faces_all = [], [], []

    #if zeiss_type == 'not_zeiss':
    OUT_group = f.get('SURF')

    #frames = f['SURF/FRAME']

    frames = OUT_group.get('FRAME')
    frames = np.array(frames).flatten()


    for frame in frames:
        frames_all.append(frame)
        idx = np.where(frames == frame)

        vertices = OUT_group.get('VERTICES')
        vertices_ref = vertices[idx]
        vertices = np.array(f[vertices_ref[0][0]]).T
        vertices_all.append(vertices)

        faces = OUT_group.get('FACES')
        faces_ref = faces[idx]
        if zeiss_type == 'not_zeiss':
            faces = np.array(f[faces_ref[0][0]]) - 1 # note: indexing for these starts at 1, so subtraction of 1 needed
        elif zeiss_type == 'zeiss':
            faces = np.array(f[faces_ref[0][0]])
        faces_all.append(_modify_faces(faces))

    return frames_all,  vertices_all, faces_all
