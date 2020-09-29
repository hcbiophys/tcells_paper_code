import nibabel as nib

class Utils_Disk:


    def copy_voxels_notDone(self, doneDir, toCopyDir):
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


    def load_and_check_nib(self):

        '/Users/harry/Desktop/RUNNING/in/test.nii.gz'
        voxels = nib.load('/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom1_someStack10Missing/stack2/Stack2_0.nii.gz').get_fdata()

        voxels = 1-voxels

        lw, num = measurements.label(voxels)
        print(voxels.min(), voxels.max())
        print(voxels.shape)
        area = measurements.sum(voxels, lw, index=np.arange(lw.max() + 1))
        print('area', area)
        print('num', num)
