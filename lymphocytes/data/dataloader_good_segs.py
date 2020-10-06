"""
GOOD SEG DATA
Original data with segmentation errors corrected.
"""



stack2_mat_filename = '/Users/harry/Desktop/lymphocytes/good_seg_data/stack2/Stack2-BC-Surf-Lim-New-T-corr_GAUSS_Export_Surf_corr.mat'
stack2_coeffPathFormat = '/Users/harry/Desktop/lymphocytes/good_seg_data/stack2/coeffs/stack2_newSeg_{}_pp_surf_SPHARM_ellalign.txt'
stack2_zoomedVoxelsPathFormat = '/Users/harry/Desktop/lymphocytes/good_seg_data/stack2/zoomedVoxels_0.2/{}.nii.gz'

stack3_mat_filename = '/Users/harry/Desktop/lymphocytes/good_seg_data/stack3/Stack3-BC-Surf-Lim-New-T-corr_GAUSS_Export_Surf_corr.mat'
stack3_coeffPathFormat = '/Users/harry/Desktop/lymphocytes/good_seg_data/stack3/coeffs/{}_pp_surf_SPHARM_ellalign.txt'
stack3_zoomedVoxelsPathFormat = '/Users/harry/Desktop/lymphocytes/good_seg_data/stack3/zoomedVoxels_0.2/{}.nii.gz'

"""
stack_triplets = [(stack2_mat_filename, stack2_coeffPathFormat, stack2_zoomedVoxelsPathFormat),
                (stack3_mat_filename, stack3_coeffPathFormat, stack3_zoomedVoxelsPathFormat)]
"""

stack_triplets = [(stack3_mat_filename, stack3_coeffPathFormat, stack3_zoomedVoxelsPathFormat)]
