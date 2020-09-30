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


class Lymph_Snap(Raw_Methods, SH_Methods):
    """
    Class for a single snap/frame of a lymphocyte series.
    Mixins are:
    - Raw_Methods: methods without spherical harmonics.
    - SH_Methods:methods with spherical harmonics.
    """

    def __init__(self, frame, mat_filename, coeffPathStart, zoomedVoxelsPathStart, speed = None, angle = None):
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
        print(frame)

        f = h5py.File(mat_filename, 'r')
        OUT_group = f.get('OUT')

        frames = OUT_group.get('FRAME')
        frames = np.array(frames).flatten()
        idx = np.where(frames == frame)

        voxels = OUT_group.get('BINARY_MASK')
        voxels_ref = voxels[idx]
        self.voxels = f[voxels_ref[0][0]] # takes a long time

        vertices = OUT_group.get('VERTICES')
        vertices_ref = vertices[idx]
        self.vertices = f[vertices_ref[0][0]]

        faces = OUT_group.get('FACES')
        faces_ref = faces[idx]
        self.faces = f[faces_ref[0][0]]

        self.speed = speed
        self.angle = angle

        self.zoomed_voxels = None
        self.coeff_array = None

        if zoomedVoxelsPathStart is not None:
            self.zoomed_voxels = nib.load(zoomedVoxelsPathStart + '{}'.format(frame))
        if not coeffPathStart is None:
            self.SH_set_spharm_coeffs(coeffPathStart + '{}_pp_surf_SPHARM_ellalign.txt'.format(frame))
