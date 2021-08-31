import numpy as np
import nibabel as nib
import skimage.morphology
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import label
import sys

def process_voxels(voxels):
    voxels = keep_only_largest_object(voxels)
    voxels = binary_fill_holes(voxels).astype(int)
    return voxels



def keep_only_largest_object(voxels):
    labels = skimage.morphology.label(voxels, connectivity = 1)
    labels_num = [len(labels[labels==each]) for each in np.unique(labels)]
    rank = np.argsort(np.argsort(labels_num))
    max_index = list(rank).index(len(rank)-2)
    new_voxels = np.zeros_like(voxels)
    new_voxels[labels == max_index] = 1
    return new_voxels
