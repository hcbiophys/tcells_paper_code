"""
GOOD SEG DATA
Original data with segmentation errors corrected.
"""



stack2_mat_filename = '../../good_seg_data_1/stack2/Stack2-BC-Surf-Lim-New-T-corr_GAUSS_Export_Surf_corr.mat'
stack2_coeffPathFormat = '../../good_seg_data_1/stack2/coeffs/{}_pp_surf_SPHARM.txt'
stack2_zoomedVoxelsPathFormat = '../../good_seg_data_1/stack2/zoomedVoxels_0.2/{}.nii.gz'

stack3_mat_filename = '../../good_seg_data_1/stack3/Stack3-BC-Surf-Lim-New-T-corr_GAUSS_Export_Surf_corr.mat'
stack3_coeffPathFormat = '../../good_seg_data_1/stack3/coeffs/{}_pp_surf_SPHARM.txt'
stack3_zoomedVoxelsPathFormat = '../../good_seg_data_1/stack3/zoomedVoxels_0.2/{}.nii.gz'


stack_quads_1 = [(stack2_mat_filename, stack2_coeffPathFormat, stack2_zoomedVoxelsPathFormat, (0.103, 0.103, 0.211)),
                (stack3_mat_filename, stack3_coeffPathFormat, stack3_zoomedVoxelsPathFormat, (0.103, 0.103, 0.211))]
