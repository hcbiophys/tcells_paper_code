"""
BAD SEG DATA
Original data with segmentation errors (tunneling to the inside).
"""

mat_filename_405s2 = '../batch1/405s2/mat/405s2.mat'
coeffPathStart_405s2 = '../batch1/zoom0.2_coeffs/405s2/405s2_'
niigzDir_405s2 = '../batch1/niigz_zoom0.2/405s2/'
exit_idxs_405s2 = []

mat_filename_405s3 = '../batch1/405s3/mat/405s3.mat'
coeffPathStart_405s3 = '../batch1/zoom0.2_coeffs/405s3/405s3_'
niigzDir_405s3 = '../batch1/niigz_zoom0.2/405s3/'
exit_idxs_405s3 = []

mat_filename_406s2 = '../batch1/406s2/mat/406s2.mat'
coeffPathStart_406s2 = '../batch1/zoom0.2_coeffs/406s2/406s2_'
niigzDir_406s2 = '../batch1/niigz_zoom0.2/406s2/'
exit_idxs_406s2 = []

mat_filename_406s2_SMALL = '../zoom0.08_406s2/406s2.mat'
coeffPathStart_406s2_SMALL = '../zoom0.08_406s2/coeffs/406s2_'
niigzDir_406s2_SMALL = '../zoom0.08_406s2/niigz/'


# -------------------

mat_filename_stack2 = '../batch2/Stack2.mat'
coeffPathStart_stack2 = '../batch2/zoom0.2_coeffs/stack2/stack2_'
niigzDir_stack2 = '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack2/'
exit_idxs_stack2 = list(range(285, 288)) + list(range(294, 300))

mat_filename_stack3 = '../batch2/Stack3.mat'
coeffPathStart_stack3 = '../batch2/zoom0.2_coeffs/stack3/stack3_'
niigzDir_stack3 = '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack3/'
exit_idxs_stack3 = list(range(90, 100)) + list(range(117, 131)) + list(range(139, 300))

mat_filename_stack4 = '../batch2/Stack4.mat'
coeffPathStart_stack4 = '../batch2/zoom0.2_coeffs/stack4/stack4_'
niigzDir_stack4 = '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack4/'
exit_idxs_stack4 = list(range(2, 11)) + list(range(20, 34)) + list(range(37, 175))

mat_filename_stack5 = '../batch2/Stack5.mat'
coeffPathStart_stack5 = '../batch2/zoom0.2_coeffs/stack5/stack5_'
niigzDir_stack5 = '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack5/'
exit_idxs_stack5  = list(range(50, 115))

mat_filename_stack7 = '../batch2/Stack7.mat'
coeffPathStart_stack7 = '../batch2/zoom0.2_coeffs/stack7/stack7_'
niigzDir_stack7 = '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack7/'
exit_idxs_stack7 = list(range(9, 71))

mat_filename_stack9 = '../batch2/Stack9.mat'
coeffPathStart_stack9 = '../batch2/zoom0.2_coeffs/stack9/stack9_'
niigzDir_stack9 = '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack9/'
exit_idxs_stack9 = list(range(39, 45)) + list(range(59, 132))

mat_filename_stack10 = '../batch2/Stack10.mat'
coeffPathStart_stack10 = '../batch2/zoom0.2_coeffs/stack10/stack10_'
niigzDir_stack10 = '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack10/'
exit_idxs_stack10 = list(range(23, 57)) + list(range(60, 63)) + list(range(68, 202))



lymph_series_405s2 = [mat_filename_405s2, coeffPathStart_405s2, niigzDir_405s2, exit_idxs_405s2]
lymph_series_405s3 = [mat_filename_405s3, coeffPathStart_405s3, niigzDir_405s3, exit_idxs_405s3]
lymph_series_406s2 = [mat_filename_406s2, coeffPathStart_406s2, niigzDir_406s2, exit_idxs_406s2]

lymph_series_406s2_SMALL = [mat_filename_406s2_SMALL, coeffPathStart_406s2_SMALL, niigzDir_406s2_SMALL]

lymph_series_stack2 = [mat_filename_stack2, coeffPathStart_stack2, niigzDir_stack2, exit_idxs_stack2]
lymph_series_stack3 = [mat_filename_stack3, coeffPathStart_stack3, niigzDir_stack3, exit_idxs_stack3]
lymph_series_stack4 = [mat_filename_stack4, coeffPathStart_stack4, niigzDir_stack4, exit_idxs_stack4]
lymph_series_stack5 = [mat_filename_stack5, coeffPathStart_stack5, niigzDir_stack5, exit_idxs_stack5]
lymph_series_stack7 = [mat_filename_stack7, coeffPathStart_stack7, niigzDir_stack7, exit_idxs_stack7]
lymph_series_stack9 = [mat_filename_stack9, coeffPathStart_stack9, niigzDir_stack9, exit_idxs_stack9]
lymph_series_stack10 = [mat_filename_stack10, coeffPathStart_stack10, niigzDir_stack10, exit_idxs_stack10]



stack_quads_list = [lymph_series_stack2, lymph_series_stack3, lymph_series_stack4,
            lymph_series_stack5, lymph_series_stack7,lymph_series_stack9,
            lymph_series_stack10, lymph_series_405s2, lymph_series_405s3,
            lymph_series_406s2]
