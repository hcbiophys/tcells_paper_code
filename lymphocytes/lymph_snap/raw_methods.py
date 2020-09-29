import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from lymphocytes.utils.disk import Utils_Disk


class Raw_Methods:

    def write_voxels_to_niigz(self, save_folder, zoom_factor = 1):

        voxels = zoom(self.voxels, (zoom_factor, zoom_factor, zoom_factor))

        new_image = nib.Nifti1Image(voxels, affine=np.eye(4))
        nib.save(new_image, save_folder + '/' + os.path.basename(self.mat_filename[:-4]) + '_' + str(self.idx) + '.nii.gz')


    def surface_plot(self):
        fig_surfacePlot = plt.figure()
        ax = fig_surfacePlot.add_subplot(111, projection='3d')

        ax.plot_trisurf(self.vertices[0, :], self.vertices[1, :], self.vertices[2, :], triangles = np.asarray(self.faces[:, ::4]).T)

        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        equal_axes(ax)



    def scatter_vertices(self, keep_every):



        xs = self.vertices[0, :][0::keep_every]
        ys = self.vertices[1, :][0::keep_every]
        zs = self.vertices[2, :][0::keep_every]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(xs, ys, zs)

        voxels = self.voxels
        voxels = Utils_Disk.keep_only_largest_object(voxels)
        voxels = binary_fill_holes(voxels).astype(int)
        print('vox sum', np.sum(voxels))



    def show_voxels(self, zoom_factor = 0.2):

        fig_showVoxels = plt.figure()
        print(self.voxels.shape)
        voxels = zoom(self.voxels, (zoom_factor, zoom_factor, zoom_factor))
        print(voxels.shape)

        voxels = voxels.astype(bool)

        x, y, z = np.indices(voxels.shape)
        cols = np.empty(voxels.shape + (3,))
        cols[..., 0] = np.true_divide(z, 150)

        c = colors.hsv_to_rgb(cols)

        ax = fig_showVoxels.add_subplot(111, projection = '3d')

        ax.voxels(voxels, facecolors = c, edgecolors = 'white')


    def get_face_vertices(self, cell_idx, face_idx):

        face = self.faces[:, 5]
        face_vertices = [vertices[:, face[0]], vertices[:, face[1]], vertices[:, face[2]]]
        xs = [i[0] for i in face_vertices]
        ys = [i[1] for i in face_vertices]
        zs = [i[2] for i in face_vertices]
        return xs, ys, zs
