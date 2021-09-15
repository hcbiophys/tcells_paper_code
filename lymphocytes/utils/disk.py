import numpy as np
import nibabel as nib
import glob
import os
import h5py
from scipy.ndimage import zoom
import sys
#from lymphocytes.data.dataloader_good_segs_2 import stack_triplets

def get_attribute_from_mat(mat_filename, zeiss_bool):
    """
    returns attributes: frames, voxels, vertices, faces
    """
    f = h5py.File(mat_filename, 'r')

    frames_all, voxels_all, vertices_all, faces_all = [], [], [], []

    if not zeiss_bool:
        OUT_group = f.get('OUT')

        frames = f['OUT/FRAME']

        frames = OUT_group.get('FRAME')
        frames = np.array(frames).flatten()

        for frame in frames:
            frames_all.append(frame)
            idx = np.where(frames == frame)

            voxels = OUT_group.get('BINARY_MASK')
            voxels_ref = voxels[idx]
            voxels = f[voxels_ref[0][0]] # takes a long time
            voxels_all.append(voxels)

            vertices = OUT_group.get('VERTICES')
            vertices_ref = vertices[idx]
            vertices = np.array(f[vertices_ref[0][0]]).T
            vertices_all.append(vertices)

            faces = OUT_group.get('FACES')
            faces_ref = faces[idx]
            faces = np.array(f[faces_ref[0][0]]) - 1 # note: indexing for these starts at 1, so subtraction of 1 needed
            faces = faces.astype(np.intc).T
            faces = np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis = 1).flatten()
            faces_all.append(faces)


    else:

        OUT_group = f.get('DataOut')
        surf_refs = OUT_group.get('Surf')
        frame_refs = np.array(surf_refs[0, :]).flatten()
        frames = np.array([np.array(f[frame_ref])[0][0] for frame_ref in frame_refs])
        max_frame = int(np.max(frames))

        for frame in range(1, max_frame+1):
            frames_all.append(frame)
            idx = np.where(frames == frame)[0][0]


            voxels = f[surf_refs[2, idx]] # takes a long time
            voxels_all.append(voxels)


            vertices = f[surf_refs[3, idx]]
            vertices = np.array(vertices).T
            vertices_all.append(vertices)

            faces = f[surf_refs[4, idx]]
            faces = np.array(faces) # note: indexing for these starts at 0, so no subtraction of 1 needed
            faces = faces.astype(np.intc).T
            faces = np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis = 1).flatten()
            faces_all.append(faces)



    return frames_all, voxels_all, vertices_all, faces_all








def write_all_zoomed_niigz(mat_filename, save_format, zoom_factor = 0.2):



    frames_all, voxels_all, vertices_all, faces_all = get_attribute_from_mat(mat_filename=mat_filename, zeiss_bool=False)


    for frame in np.array(frames).flatten():
        print(frame)
        idx = np.where(frames == frame)
        voxels = OUT_group.get('BINARY_MASK')
        voxels_ref = voxels[idx]

        voxels = f[voxels_ref[0][0]] # takes a long time
        voxels = zoom(voxels, (zoom_factor, zoom_factor, zoom_factor), order = 0) # order 0 means not interpolation
        #print(voxels.shape)
        new_image = nib.Nifti1Image(voxels, affine=np.eye(4))
        nib.save(new_image, save_format.format(int(frame)))

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


def copy_coefs_into_dir(inDir, outDir, idx_cell):
    for file in glob.glob(inDir + '{}_*SPHARM.coef'.format(idx_cell)):
        os.rename(file, outDir + os.path.basename(file))





if __name__ == "__main__":
    inDir = '/Users/harry/Desktop/RUNNING/out/Step3_ParaToSPHARMMesh/'
    outDir =  '/Users/harry/OneDrive - Imperial College London/lymphocytes/good_seg_data_3/210426_M415_OT1GFP_CNA35mCherry_CTDR_1_5000_1_5_25deg/cell_7/coeffs/'

    #mat_filename = '/Users/harry/OneDrive - Imperial College London/lymphocytes/good_seg_data_3/210426_M415_OT1GFP_CNA35mCherry_CTDR_1_5000_1_5_25deg/cell_7/Cell7_BC_0_35_Export_Surf_corr.mat'
    #save_format = '/Users/harry/OneDrive - Imperial College London/lymphocytes/good_seg_data_3/210426_M415_OT1GFP_CNA35mCherry_CTDR_1_5000_1_5_25deg/cell_7/zoomedVoxels_0.2/7_{}.nii.gz'

    #write_all_zoomed_niigz(mat_filename, save_format)

    copy_coefs_into_dir(inDir, outDir, idx_cell=7)




    #rm_done_from_inDir('/Users/harry/Desktop/RUNNING/STACK2/', '/Users/harry/Desktop/RUNNING/out/Step1_SegPostProcess/')
