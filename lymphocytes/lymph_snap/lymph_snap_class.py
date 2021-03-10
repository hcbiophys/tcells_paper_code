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


from lymphocytes.lymph_snap.raw_methods import Raw_Methods
from lymphocytes.lymph_snap.SH_methods import SH_Methods

from lymphocytes.utils.voxels import process_voxels



class Lymph_Snap(Raw_Methods, SH_Methods):
    """
    Class for a single snap/frame of a lymphocyte series.
    Mixins are:
    - Raw_Methods: methods without spherical harmonics.
    - SH_Methods: methods with spherical harmonics.
    """

    def __init__(self, frame, mat_filename, coeffPathFormat, zoomedVoxelsPathFormat, max_l):
        """
        Args:
        - frame: frame number (beware of gaps in these as cells can exit the arenas).
        - mat_filename: .mat file holding the series (read using h5py).
        - coeffPathStart: start of paths for SPHARM coefficients.
        - zoomedVoxelsPathStart: start of paths for the zoomed voxels (saves on processing time).
        - speed: calculated speed at this snap.
        - angle: calculated angle at this snap.
        """

        self.mat_filename = mat_filename
        self.frame = frame
        self.max_l = max_l

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
        self.vertices = f[vertices_ref[0][0]]

        faces = OUT_group.get('FACES')
        faces_ref = faces[idx]
        self.faces = np.array(f[faces_ref[0][0]]) -1


        self.zoomed_voxels = None
        self.volume = None
        if zoomedVoxelsPathFormat is not None:
            self.zoomed_voxels_path = zoomedVoxelsPathFormat.format(frame)
            zoomed_voxels = np.asarray(nib.load(self.zoomed_voxels_path).dataobj)
            self.zoomed_voxels = process_voxels(zoomed_voxels)
            self.volume = np.sum(self.zoomed_voxels)*(5**3)*(0.103**2)*0.211

        self.coeff_array = None
        self._set_spharm_coeffs(coeffPathFormat.format(frame))
        self.vector = None
        self.RI_vector = None
        self._set_vector()
        self._set_rotInv_vector()

        self.speed = None
        self.angle = None
        self.pca = None
