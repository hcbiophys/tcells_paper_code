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

    def __init__(self, mat_filename, coeffPathStart, idx, niigz, speed = None, angle = None, exited = False):

        f = h5py.File(mat_filename, 'r')
        """
        dataset = f['DataOut/Surf']
        self.mat_filename = mat_filename
        self.idx = idx
        self.voxels = f[dataset[2, idx]]
        self.vertices = f[dataset[3, idx]]
        self.faces = f[dataset[4, idx]]
        """
        dataset = f['OUT/BINARY_MASK']
        a = f[dataset]
        print(a)

        sys.exit()


        self.coeff_array = None

        if not coeffPathStart is None:
            self.SH_set_spharm_coeffs(coeffPathStart + '{}_pp_surf_SPHARM_ellalign.txt'.format(idx))

        self.niigz = niigz
        self.speed = speed
        self.angle = angle
        self.exited = exited
