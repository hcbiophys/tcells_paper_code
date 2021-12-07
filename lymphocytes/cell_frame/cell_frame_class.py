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
        self.coeffPathFormat = coeffPathFormat
        self.frame = frame
        self.idx_cell = idx_cell
        self.voxels = voxels
        self.xyz_res = xyz_res
        self.color = None
        self.t_res = None
        self.max_l = max_l
        self.uropod = uropod


        self.vertices = None
        self.faces = None
        if vertices is not None:
            surf = pv.PolyData(vertices, faces)
            surf = surf.connectivity(largest=True) # in case there are a) holes b) floating artefacts

            self.vertices = surf.points
            self.faces = surf.faces

            self.centroid = surf.center_of_mass()
            self.volume = surf.volume


        self.coeff_array = None
        self.vector = None
        self.RI_vector = None




        if coeffPathFormat is not None and uropod is not None:
            self._set_spharm_coeffs(coeffPathFormat.format(int(frame)))
            self._set_vector()
            self._set_RIvector()



        self.morph_deriv = None
        self.morph_deriv_low = None
        self.morph_deriv_high = None

        self.run_uropod = None
        self.run_centroid = None
        self.run_theta = None

        self.delta_uropod = None
        self.delta_centroid= None

        self.spin_vec = None

        self.angle = None
        self.spin_vec_std = None
        self.angle_mean = None


        self.spin_vec_2 = None
        self.angle_2 = None
        self.spin_vec_std_2 = None
        self.angle_mean_2 = None

        self.mean_uropod = None
        self.run_uropod_running_mean = None
        self.mean_centroid = None

        self.pca = None
        self.pca0 = None
        self.pca1 = None
        self.pca2 = None

        self.ellipsoid_lengths = None
        self.ellipsoid_vecs = None

        self.uropod_aligned = False # not yet aligned by uropod-centroid vector

        self.is_interpolation = False
