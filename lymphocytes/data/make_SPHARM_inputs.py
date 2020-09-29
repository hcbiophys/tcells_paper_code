import h5py # Hierarchical Data Format 5
from lymphocytes.lymph_snap.lymph_snap_class import Lymph_Snap


mat_filename = '/Users/harry/Desktop/lymphocytes/good_seg_data/Stack2-BC-Surf-Lim-New-T-corr_GAUSS_Export_Surf_corr.mat'

f = h5py.File(mat_filename, 'r')
idxs = f['OUT/FRAME']

for idx_ in [0]:

    Lymph_Snap(mat_filename, coeffPathStart = None, idx = idx_, niigz = None, speed = None, angle = None, exited = False)

    Lymph_Snap.write_voxels_to_niigz(self, save_folder, zoom_factor = 1)
