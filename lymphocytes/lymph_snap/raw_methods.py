import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
import os
from lymphocytes.utils.disk import *
from lymphocytes.utils.plotting import no_pane, equal_axes_3D

class Raw_Methods:
    """
    Inherited by Lymph_Snap class.
    Contains methods without spherical harmonics.
    """



    def surface_plot(self, subsample_rate):
        """
        Plot the (subsampled) triangulation surface.
        """
        fig_surfacePlot = plt.figure()
        ax = fig_surfacePlot.add_subplot(111, projection='3d')

        ax.plot_trisurf(self.vertices[0, :], self.vertices[1, :], self.vertices[2, :], triangles = np.asarray(self.faces[:, ::subsample_rate]).T)

        ax.grid(False)

        no_pane(ax)
        equal_axes_3D(ax)


    def show_voxels(self, origOrZoomed):
        """
        Plot voxels, either orig(inal) or zoomed (in which case read from
        saved subsampled for speed reasons).
        """

        if origOrZoomed == 'orig':
            voxels = self.voxels
        elif origOrZoomed == 'zoomed':
            voxels = self.zoomed_voxels

        fig_showVoxels = plt.figure()

        x, y, z = np.indices(voxels.shape)
        cols = np.empty(voxels.shape + (3,))
        cols[..., 0] = np.true_divide(z, 150)
        #c = colors.hsv_to_rgb(cols)
        ax = fig_showVoxels.add_subplot(111, projection = '3d')
        ax.voxels(voxels,  edgecolors = 'white')

    """
    def get_face_vertices(self, cell_idx, face_idx):


        face = self.faces[:, 5]
        face_vertices = [vertices[:, face[0]], vertices[:, face[1]], vertices[:, face[2]]]
        xs = [i[0] for i in face_vertices]
        ys = [i[1] for i in face_vertices]
        zs = [i[2] for i in face_vertices]
        return xs, ys, zs
    """
