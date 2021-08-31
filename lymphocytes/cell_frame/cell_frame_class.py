import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import h5py # Hierarchical Data Format 5
import nibabel as nib
from scipy.ndimage import zoom
from scipy.special import sph_harm
from matplotlib import cm, colors
import matplotlib.tri as mtri
from mayavi import mlab
import pyvista as pv
import time

from lymphocytes.cell_frame.raw_methods import Raw_Methods
from lymphocytes.cell_frame.SH_methods import SH_Methods

from lymphocytes.utils.voxels import process_voxels



class Cell_Frame(Raw_Methods, SH_Methods):
    """
    Class for a single snap/frame of a lymphocyte series
    Mixins are:
    - Raw_Methods: methods without spherical harmonics
    - SH_Methods: methods with spherical harmonics
    """

    def __init__(self, frame, mat_filename, coeffPathFormat, zoomedVoxelsPathFormat, xyz_res, idx_cell, max_l, uropod):
        """
        Args:
        - frame: frame number (beware of gaps in these as cells can exit the arenas)
        - mat_filename: .mat file holding the series (read using h5py)
        - coeffPathStart: start of paths for SPHARM coefficients
        - zoomedVoxelsPathStart: start of paths for the zoomed voxels (saves on processing time)
        - xyz_res: resolution of the voxels (pre-zooming)
        - idx_cell: index of the cell, e.g. '3_1_0'
        - max_l: l tunrcagtion for shape descriptor
        - uropod: uropod coordinates
        """

        self.mat_filename = mat_filename
        self.frame = frame
        self.idx_cell = idx_cell
        self.xyz_res = xyz_res
        self.color = None
        self.t_res = None
        self.max_l = max_l
        self.uropod = uropod




        ### read the .mat file ###

        f = h5py.File(mat_filename, 'r')
        OUT_group = f.get('OUT')

        frames = OUT_group.get('FRAME')
        frames = np.array(frames).flatten()
        idx = np.where(frames == frame)

        voxels = OUT_group.get('BINARY_MASK')
        voxels_ref = voxels[idx]
        self.voxels = f[voxels_ref[0][0]] # takes a long time
        #self.voxels = process_voxels(voxels)
        #voxelsize = OUT_group.get('VOXELSIZE')

        vertices = OUT_group.get('VERTICES')
        vertices_ref = vertices[idx]
        self.vertices = np.array(f[vertices_ref[0][0]]).T

        faces = OUT_group.get('FACES')
        faces_ref = faces[idx]
        faces = np.array(f[faces_ref[0][0]]) - 1
        faces = faces.astype(np.intc).T
        self.faces = np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis = 1).flatten()

        self.zoomed_voxels = None
        if zoomedVoxelsPathFormat is not None:
            zoomed_voxels = np.asarray(nib.load(zoomedVoxelsPathFormat.format(frame)).dataobj)
            self.zoomed_voxels = process_voxels(zoomed_voxels)
            self.zoomed_voxels = np.moveaxis(np.moveaxis(self.zoomed_voxels, 0, -1), 0, 1) # reorder
            self.volume = np.sum(self.zoomed_voxels)*(5**3)*xyz_res[0]*xyz_res[1]*xyz_res[2]
        self.centroid = None
        self._set_centroid()







        self.coeff_array = None
        self._set_spharm_coeffs(coeffPathFormat.format(frame))
        self.vector = None
        self.RI_vector = None
        self.RI_vector0 = None
        self.RI_vector1 = None
        self.RI_vector2 = None
        self.RI_vector3 = None
        self._set_vector()
        self._set_RIvector()

        self.morph_deriv = None

        # running means
        self.mean_uropod = None
        self.mean_centroid = None

        self.delta_centroid = None
        self.delta_sensing_direction = None
        self.pca = None
        self.pca0 = None
        self.pca1 = None
        self.pca2 = None


        self.uropod_aligned = False # not yet aligned by uropod-centroid vector
