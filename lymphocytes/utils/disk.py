import numpy as np
import nibabel as nib
import glob
import os
import h5py
from scipy.ndimage import zoom
import sys
#from lymphocytes.data.dataloader_good_segs_2 import stack_triplets


def write_all_zoomed_niigz(mat_filename, saveFormat, zoom_factor):

    f = h5py.File(mat_filename, 'r')
    OUT_group = f.get('OUT')

    frames = np.asarray(OUT_group.get('FRAME')).flatten()

    for frame in np.array(frames).flatten():
        print(frame)
        idx = np.where(frames == frame)
        voxels = OUT_group.get('BINARY_MASK')
        voxels_ref = voxels[idx]

        voxels = f[voxels_ref[0][0]] # takes a long time
        #print(voxels.shape)
        voxels = zoom(voxels, (zoom_factor, zoom_factor, zoom_factor), order = 0) # order 0 means not interpolation
        #print(voxels.shape)
        new_image = nib.Nifti1Image(voxels, affine=np.eye(4))
        nib.save(new_image, saveFormat.format(int(frame)))

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


def rm_done_from_inDir(inDir, Step1_SegPostProcessDir):

    for file_path1 in glob.glob(Step1_SegPostProcessDir + '*'):
        base1 = os.path.basename(file_path1)[:-8]
        for file_path2 in glob.glob(inDir + '*'):
            base2 = os.path.basename(file_path2)[:-9]
            if base1 == base2:
                print('Removing: {}'.format(file_path2))
                os.remove(file_path2)



if __name__ == "__main__":
    """
    for idx, i in enumerate(stack_triplets):
        #mat_filename = '/Users/harry/Desktop/lymphocytes/good_seg_data_2/20190405_M101_1_5_mg_27_5_deg/stack4/Stack4_BC-T-corr-0_35um_Export_Surf_corr.mat'
        mat_filename = i[0]
        save_form = os.path.dirname(mat_filename) + '/zoomedVoxels_0.2/{}_{{}}.nii.gz'.format(idx)
        write_all_zoomed_niigz(mat_filename, save_form, 0.2)
    """

    rm_done_from_inDir('/Users/harry/Desktop/RUNNING/STACK2/', '/Users/harry/Desktop/RUNNING/out/Step1_SegPostProcess/')
