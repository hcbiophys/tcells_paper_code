import numpy as np





def decimate_mat_voxels(mat_filename, idx_snap, decimation_factor, show, save_as):

    f = h5py.File(mat_filename, 'r')
    dataset = f['DataOut/Surf']
    voxels = f[dataset[2, idx_snap]]

    voxels = zoom(voxels, (decimation_factor, decimation_factor, decimation_factor), order = 0).astype('double')


    scipy.io.savemat(save_as, mdict={'bim': voxels,
                                    'origin': np.array([0, 0, 0]).astype('double'),
                                    'vxsize': np.array([1, 1, 1]).astype('double')})




def del_whereNone(nestedLists, attribute):

    print('series PERMANENTLY EDITED')

    nestedLists_new = []
    for list in nestedLists:
        list_new = []
        for item in list:
            if attribute == 'lymph':
                if item is not None:
                    list_new.append(item)
            if attribute == 'speed':
                if item.speed is not None:
                    list_new.append(item)
            elif attribute == 'angle':
                if item.angle is not None:
                    list_new.append(item)

        nestedLists_new.append(list_new)

    return nestedLists_new
