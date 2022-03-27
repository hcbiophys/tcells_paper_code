
### RUN ###

mat_filename_run1 = '../data/surface_segmentations/CELL21.mat'
coeffPathFormat_run1 = '../data/run_series/run1/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(5)

mat_filename_run2  = '../data/surface_segmentations/CELL29.mat'
coeffPathFormat_run2  = '../data/run_series/run2/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(1, 1)


mat_filename_run3 = '../data/surface_segmentations/CELL20.mat'
coeffPathFormat_run3 = '../data/run_series/run3/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(4)


mat_filename_run4 = '../data/surface_segmentations/CELL25.mat'
coeffPathFormat_run4 = '../data/run_series/run4/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(1)



### STOP ###

mat_filename_stop1 = '../data/surface_segmentations/CELL1.mat'
coeffPathFormat_stop1 = '../data/stop_series/stop1/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(1)

mat_filename_stop2 = '../data/surface_segmentations/CELL24.mat'
coeffPathFormat_stop2 = '../data/stop_series/stop2/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(0)

mat_filename_stop3 = '../data/surface_segmentations/CELL19.mat'
coeffPathFormat_stop3 = '../data/stop_series/stop3/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(3)

mat_filename_stop4 = '../data/surface_segmentations/CELL31.mat'
coeffPathFormat_stop4 = '../data/stop_series/stop4/coeffs/{}_{{}}_pp_surf_SPHARM.txt'.format(0, 0)


stack_attributes_stereotypical = [('CELL21', mat_filename_run1, coeffPathFormat_run1,  (0.5, 0.5, 0.5), (0.8, 0.1, 0.7), 4.166),
                                    ('CELL29', mat_filename_run2, coeffPathFormat_run2,  (0.5, 0.5, 0.5), (0, 0, 0.3), 4.166),
                                    ('CELL20', mat_filename_run3, coeffPathFormat_run3,  (0.5, 0.5, 0.5), (0.5, 0.9, 0.2), 4.166),
                                    ('CELL25', mat_filename_run4, coeffPathFormat_run4,  (0.5, 0.5, 0.5), (0.3, 0.7, 0.8), 4.166),

                                    ('CELL1', mat_filename_stop1, coeffPathFormat_stop1,  [5*i for i in [0.103, 0.103, 0.211]], (0.7, 0.7, 0), 2.5),
                                    ('CELL24', mat_filename_stop2, coeffPathFormat_stop2,  (0.5, 0.5, 0.5), (0.1, 0.1, 0.3), 4.166),
                                    ('CELL19', mat_filename_stop3, coeffPathFormat_stop3,  (0.5, 0.5, 0.5), (0.3, 0.7, 0.8), 4.166),
                                    ('CELL31', mat_filename_stop4, coeffPathFormat_stop4,  (0.5, 0.5, 0.5), (0, 0.1, 0), 4.166)]
