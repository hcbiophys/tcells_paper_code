import nibabel as nib



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

    x_range, y_range, z_range = self.find_voxel_ranges(voxels)

    ranges = [x_range, y_range, z_range]

    print('shortest: ', ranges.index(min(ranges)))

    """
    if ranges.index(max(ranges)) == 0:
        elev = 0
        azim = 0
    else:
        elev = 0
        azim = 90

    return elev, azim
    """

    if ranges.index(min(ranges)) == 0:
        azim = 0
    if ranges.index(min(ranges)) == 1:
        azim = 90
    else:
        azim = 90

    elev = 0
    return elev, azim


def voxel_volume(voxels):

    voxels = keep_only_largest_object(voxels)
    voxels = binary_fill_holes(voxels).astype(int)

    return np.sum(voxels)

def read_niigz(niigz):

    voxels = keep_only_largest_object(voxels)
    voxels = binary_fill_holes(voxels).astype(int)

    return voxels



def keep_only_largest_object(voxels):

        labels = morphology.label(voxels, connectivity = 1)
        labels_num = [len(labels[labels==each]) for each in np.unique(labels)]
        rank = np.argsort(np.argsort(labels_num))
        max_index = list(rank).index(len(rank)-2)
        new_array = np.zeros_like(array)
        new_array[labels == max_index] = 1
        return new_array
