import numpy as np
import nibabel as nib
import glob
import os
import h5py
from scipy.ndimage import zoom


def copy_voxels_notDone(doneDir, toCopyDir):
    """
    Compare doneDir & toCopyDir, and copy discrepancies into ~/Desktop/RUNNING/in/
    """

    done_paths = glob.glob(doneDir+'*')

    done_idxs = [int(os.path.basename(i).split('_')[1]) for i in done_paths]

    for path in glob.glob(toCopyDir+'*'):
        idx = int(os.path.basename(path).split('_')[1].split('.')[0])

        if not idx in done_idxs:
            #copyfile(path, '/Users/harry/Desktop/RUNNING/in/' + os.path.basename(path))\

            voxels = read_niigz(path)


            lw, num = measurements.label(voxels)
            area = measurements.sum(voxels, lw, index=np.arange(lw.max() + 1))
            print('num', num, 'area', area)

            new_image = nib.Nifti1Image(voxels, affine=np.eye(4))
            nib.save(new_image, '/Users/harry/Desktop/RUNNING/in/' + os.path.basename(path))


def load_and_check_nib():

    '/Users/harry/Desktop/RUNNING/in/test.nii.gz'
    voxels = nib.load('/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom1_someStack10Missing/stack2/Stack2_0.nii.gz').get_fdata()

    voxels = 1-voxels

    lw, num = measurements.label(voxels)
    print(voxels.min(), voxels.max())
    print(voxels.shape)
    area = measurements.sum(voxels, lw, index=np.arange(lw.max() + 1))
    print('area', area)
    print('num', num)



def write_all_zoomed_niigz(mat_filename, saveFormat, zoom_factor):

    f = h5py.File(mat_filename, 'r')
    OUT_group = f.get('OUT')

    frames = np.asarray(OUT_group.get('FRAME')).flatten()

    for frame in np.array(frames).flatten():
        idx = np.where(frames == frame)
        voxels = OUT_group.get('BINARY_MASK')
        voxels_ref = voxels[idx]

        voxels = f[voxels_ref[0][0]] # takes a long time

        voxels = zoom(voxels, (zoom_factor, zoom_factor, zoom_factor), order = 0)

        new_image = nib.Nifti1Image(voxels, affine=np.eye(4))
        nib.save(new_image, saveFormat.format(int(frame)))





def rm_done_from_inDir(inDir, Step1_SegPostProcessDir):

    for file_path1 in glob.glob(Step1_SegPostProcessDir + '*'):
        base1 = os.path.basename(file_path1)[:-8]
        for file_path2 in glob.glob(inDir + '*'):
            base2 = os.path.basename(file_path2)[:-9]
            if base1 == base2:
                print('Removing: {}'.format(file_path2))
                os.remove(file_path2)



if __name__ == "__main__":
    save_form = '/Users/harry/Desktop/lymphocytes/good_seg_data/stack3/zoomedVoxels_0.2/{}.nii.gz'
    write_all_zoomed_niigz('/Users/harry/Desktop/lymphocytes/good_seg_data/stack3/Stack3-BC-Surf-Lim-New-T-corr_GAUSS_Export_Surf_corr.mat', save_form, 0.2)
    #pass
