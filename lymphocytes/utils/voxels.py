import numpy as np
import nibabel as nib
import skimage.morphology
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import label
import sys

def find_voxel_ranges(voxels):

    start = 0
    end = 0
    for idx_slice in range(voxels.shape[0]):
        slice = voxels[idx_slice, :, :]
        if slice.max() == 1:
            start = idx_slice
            break
    for idx_slice in np.arange(start, voxels.shape[0]):
        slice = voxels[idx_slice, :, :]
        if slice.max() == 0:
            end = idx_slice
            break
    x_range = end-start

    start = 0
    end = 0
    for idx_slice in range(voxels.shape[1]):
        slice = voxels[:, idx_slice, :]
        if slice.max() == 1:
            start = idx_slice
            break
    for idx_slice in np.arange(start, voxels.shape[1]):
        slice = voxels[:, idx_slice, :]
        if slice.max() == 0:
            end = idx_slice
            break
    y_range = end-start

    start = 0
    end = 0
    for idx_slice in range(voxels.shape[2]):
        slice = voxels[:, :, idx_slice]
        if slice.max() == 1:
            start = idx_slice
            break
    for idx_slice in np.arange(start, voxels.shape[2]):
        slice = voxels[:, :, idx_slice]
        if slice.max() == 0:
            end = idx_slice
            break
    z_range = end-start

    return x_range, y_range, z_range



def find_optimal_3dview(voxels):
    x_range, y_range, z_range = find_voxel_ranges(voxels)
    ranges = [x_range, y_range, z_range]

    if ranges.index(min(ranges)) == 0:
        azim = 0
    if ranges.index(min(ranges)) == 1:
        azim = 90
    else:
        azim = 90

    elev = 0
    return elev, azim


def process_voxels(voxels):

    voxels = keep_only_largest_object(voxels)
    voxels = binary_fill_holes(voxels).astype(int)

    return voxels





def keep_only_largest_object(voxels):
    #labeled, ncomponents = label(voxels)

    labels = skimage.morphology.label(voxels, connectivity = 1)
    labels_num = [len(labels[labels==each]) for each in np.unique(labels)]
    rank = np.argsort(np.argsort(labels_num))
    max_index = list(rank).index(len(rank)-2)
    new_voxels = np.zeros_like(voxels)
    new_voxels[labels == max_index] = 1

    #labeled, ncomponents = label(new_voxels)

    return new_voxels
