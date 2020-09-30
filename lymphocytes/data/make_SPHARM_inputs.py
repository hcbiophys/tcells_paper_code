import numpy as np
import h5py # Hierarchical Data Format 5
from lymphocytes.lymph_snap.lymph_snap_class import Lymph_Snap


mat_filename = '/Users/harry/Desktop/lymphocytes/good_seg_data/Stack2-BC-Surf-Lim-New-T-corr_GAUSS_Export_Surf_corr.mat'


f = h5py.File(mat_filename, 'r')
OUT_group = f.get('OUT')
frames = OUT_group.get('FRAME')

frames = np.array(frames).flatten()


for frame in frames:

    lymph_snap = Lymph_Snap(mat_filename = mat_filename, frame = frame, coeffPathStart = None, zoomed_voxels_path = None, speed = None, angle = None)

    lymph_snap.write_voxels_to_niigz(save_base = '/Users/harry/Desktop/new_voxels_Z0.3/stack2_newSeg_', zoom_factor = 0.2)
