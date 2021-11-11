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

    def __init__(self, frame, mat_filename, coeffPathFormat, voxels, xyz_res, idx_cell, max_l, uropod,  vertices, faces):
        """
        Args:
        - frame: frame number (beware of gaps in these as cells can exit the arenas)
        - mat_filename: .mat file holding the series (read using h5py)
        - coeffPathStart: start of paths for SPHARM coefficients
        - xyz_res: resolution of the voxels (pre-zooming)
        - idx_cell: index of the cell, e.g. '3_1_0'
        - max_l: l tunrcagtion for shape descriptor
        - uropod: uropod coordinates
        """


        self.mat_filename = mat_filename
        self.frame = frame
        self.idx_cell = idx_cell
        self.voxels = voxels
        self.xyz_res = xyz_res
        self.color = None
        self.t_res = None
        self.max_l = max_l
        self.uropod = uropod




        surf = pv.PolyData(vertices, faces)
        surf = surf.connectivity(largest=True)

        self.vertices = surf.points
        self.faces = surf.faces


        self.centroid = surf.center_of_mass()
        self.volume = surf.volume

        self.coeff_array = None
        self.vector = None
        self.RI_vector = None
        self.RI_vector0 = None
        self.RI_vector1 = None
        self.RI_vector2 = None
        self.RI_vector3 = None


        self._set_spharm_coeffs(coeffPathFormat.format(int(frame)))
        self._set_vector()
        self._set_RIvector()




        self.morph_deriv = None
        self.run = None
        self.run_mean = None
        self.spin_vec = None
        self.spin_vec_magnitude = None
        self.spin_vec_magnitude_mean = None
        self.direction = None
        self.spin_vec_std = None
        self.direction_std = None


        self.mean_uropod = None
        self.mean_centroid = None

        self.delta_centroid = None
        self.delta_sensing_direction = None
        self.pca = None
        self.pca0 = None
        self.pca1 = None
        self.pca2 = None



        self.uropod_aligned = False # not yet aligned by uropod-centroid vector
