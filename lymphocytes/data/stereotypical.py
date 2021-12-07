


### RUN ###

# zm_3_3_5
mat_filename_run1 = '../../good_seg_data_3/ZeissLLS_2/20210828_ot1gfp in collagen-01-lattice lightsheet-11-3_clean_Export_Multi_Surf.mat'
coeffPathFormat_run1 = '../../run_series/run1/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(5)
zoomedVoxelsPathFormat_run1 = '../../run_series/run1/zoomedVoxels_dist0.5/{}_{{}}.nii.gz'.format(5)

# zm_3_3_2
mat_filename_run2 = '../../good_seg_data_3/ZeissLLS_2/20210828_ot1gfp in collagen-01-lattice lightsheet-11-3_clean_Export_Multi_Surf.mat'
coeffPathFormat_run2 = '../../run_series/run2/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(2)
zoomedVoxelsPathFormat_run2 = '../../run_series/run2/zoomedVoxels_dist0.5/{}_{{}}.nii.gz'.format(2)


# zm_3_3_4
mat_filename_run3 = '../../good_seg_data_3/ZeissLLS_2/20210828_ot1gfp in collagen-01-lattice lightsheet-11-3_clean_Export_Multi_Surf.mat'
coeffPathFormat_run3 = '../../run_series/run3/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(4)
zoomedVoxelsPathFormat_run3 = '../../run_series/run3/zoomedVoxels_dist0.5/{}_{{}}.nii.gz'.format(4)


# zm_3_4_1
mat_filename_run4 = '../../good_seg_data_3/ZeissLLS_3/20210828_ot1gfp in collagen-03-deskew-01_Export_Multi_Surf.mat'
coeffPathFormat_run4 = '../../run_series/run4/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(1)
zoomedVoxelsPathFormat_run4 = '../../run_series/run4/zoomedVoxels_dist0.5/{}_{{}}.nii.gz'.format(1)







### STOP ###

# 2_1
mat_filename_stop1 = '../../good_seg_data_2/20190405_M101_1_5_mg_37_deg/Stack2/Stack2-BC-T-corr-0_35um_Export_Surf_corr.mat'
coeffPathFormat_stop1 = '../../stop_series/stop1/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(1)
zoomedVoxelsPathFormat_stop1 = '../../run_series/stop1/c/zoomedVoxels_dist0.5/{}_{{}}.nii.gz'.format(1)

# zm_3_4_0
mat_filename_stop2 = '../../good_seg_data_3/ZeissLLS_3/20210828_ot1gfp in collagen-03-deskew-01_Export_Multi_Surf.mat'
coeffPathFormat_stop2 = '../../stop_series/stop2/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(0)
zoomedVoxelsPathFormat_stop2 = '../../run_series/stop2/zoomedVoxels_dist0.5/{}_{{}}.nii.gz'.format(0)

# zm_3_3_3
mat_filename_stop3 = '../../good_seg_data_3/ZeissLLS_2/20210828_ot1gfp in collagen-01-lattice lightsheet-11-3_clean_Export_Multi_Surf.mat'
coeffPathFormat_stop3 = '../../stop_series/stop3/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(3)
zoomedVoxelsPathFormat_stop3 = '../../run_series/stop3/zoomedVoxels_dist0.5/{}_{{}}.nii.gz'.format(3)

# zm_3_6_0
mat_filename_stop4 = '../../good_seg_data_3/ZeissLLS_5/20211112_OT1GFP in collagen-03-deskewed_Export_Multi_Surf.mat'
coeffPathFormat_stop4 = '../../stop_series/stop4/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(0, 0)
zoomedVoxelsPathFormat_stop4 = '../../stop_series/stop4/cell_{}/zoomedVoxels_dist0.5/{}_{{}}.nii.gz'.format(0, 0)


stack_attributes_stereotypical = [('zm_3_3_5', mat_filename_run1, coeffPathFormat_run1,  (0.5, 0.5, 0.5), (0.8, 0.1, 0.7), 4.166),
                                    ('zm_3_3_2', mat_filename_run2, coeffPathFormat_run2,  (0.5, 0.5, 0.5), (0, 0, 0.3), 4.166),
                                    ('zm_3_3_4', mat_filename_run3, coeffPathFormat_run3,  (0.5, 0.5, 0.5), (0.5, 0.9, 0.2), 4.166),
                                    ('zm_3_4_1', mat_filename_run4, coeffPathFormat_run4,  (0.5, 0.5, 0.5), (0.3, 0.7, 0.8), 4.166),

                                    ('2_1', mat_filename_stop1, coeffPathFormat_stop1,  [5*i for i in [0.103, 0.103, 0.211]], (0.7, 0.7, 0), 2.5),
                                    ('zm_3_4_0', mat_filename_stop2, coeffPathFormat_stop2,  (0.5, 0.5, 0.5), (0.1, 0.1, 0.3), 4.166),
                                    ('zm_3_3_3', mat_filename_stop3, coeffPathFormat_stop3,  (0.5, 0.5, 0.5), (0.3, 0.7, 0.8), 4.166),
                                    ('zm_3_6_0', mat_filename_stop4, coeffPathFormat_stop4,  (0.5, 0.5, 0.5), (0, 0.1, 0), 4.166)]
