import nibabel as nib
import glob
import os


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


"""
if __name__ == "__main__":
utils_disk = Utils_Disk()
utils_disk.rm_done_from_inDir('/Users/harry/Desktop/RUNNING/in/', '/Users/harry/Desktop/RUNNING/out/Step1_SegPostProcess/')
"""
