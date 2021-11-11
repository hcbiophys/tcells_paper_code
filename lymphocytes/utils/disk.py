import numpy as np
import nibabel as nib
import glob
import os
import h5py
from scipy.ndimage import zoom
import sys
import pyvista as pv
from pyvista import examples
import matplotlib.pyplot as plt
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
import pickle
import nrrd
pv.set_plot_theme("document")



def _modify_faces(faces):
    faces = faces.astype(np.intc).T
    faces = np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis = 1).flatten()
    return faces

def get_attribute_from_mat(mat_filename, zeiss_type, idx_cell = None, include_voxels = False):
    """
    returns attributes: frames, voxels, vertices, faces
    """
    f = h5py.File(mat_filename, 'r')

    frames_all, voxels_all, vertices_all, faces_all = [], [], [], []

    if zeiss_type == 'not_zeiss':
        OUT_group = f.get('OUT')

        frames = f['OUT/FRAME']

        frames = OUT_group.get('FRAME')
        frames = np.array(frames).flatten()

        for frame in frames:
            frames_all.append(frame)
            idx = np.where(frames == frame)

            if include_voxels:
                voxels = OUT_group.get('BINARY_MASK')
                voxels_ref = voxels[idx]
                voxels = f[voxels_ref[0][0]] # takes a long time
                voxels_all.append(voxels)
            else:
                voxels_all.append(None)


            vertices = OUT_group.get('VERTICES')
            vertices_ref = vertices[idx]
            vertices = np.array(f[vertices_ref[0][0]]).T
            vertices_all.append(vertices)

            faces = OUT_group.get('FACES')
            faces_ref = faces[idx]
            faces = np.array(f[faces_ref[0][0]]) - 1 # note: indexing for these starts at 1, so subtraction of 1 needed
            faces_all.append(_modify_faces(faces))


    elif zeiss_type == 'zeiss_single':

        OUT_group = f.get('DataOut')
        surf_refs = OUT_group.get('Surf')
        frame_refs = np.array(surf_refs[0, :]).flatten()
        frames = np.array([np.array(f[frame_ref])[0][0] for frame_ref in frame_refs])
        max_frame = int(np.max(frames))

        for frame in range(1, max_frame+1):
            frames_all.append(frame)
            idx = np.where(frames == frame)[0][0]
            if include_voxels:
                voxels = f[surf_refs[2, idx]] # takes a long time
                voxels_all.append(voxels)
            else:
                voxels_all.append(None)

            vertices = f[surf_refs[3, idx]]
            vertices = np.array(vertices).T
            vertices_all.append(vertices)

            faces = f[surf_refs[4, idx]]
            faces = np.array(faces) # note: indexing for these starts at 0, so no subtraction of 1 needed
            faces_all.append(_modify_faces(faces))

    elif zeiss_type == 'zeiss_many':
        OUT_group = f.get('DataOut')

        cell_group = f[OUT_group[idx_cell, 0]]

        dataset = cell_group['Surf']

        vertices = dataset[3, :]
        num = vertices.shape[0]


        for idx2 in range(num):
            surface_count = dataset[0, idx2]
            surface_index_imaris = dataset[1, idx2]
            frames = dataset[2, idx2]
            vertices = dataset[3, idx2]
            faces = dataset[4, idx2]
            if include_voxels:
                voxels = dataset[5, idx2]
                voxels_all.append(f[voxels])

            else:
                voxels_all.append(None)
            voxel_size = dataset[6, idx2]
            #print('voxel_size', np.array(f[voxel_size]))

            t_res = dataset[7, idx2]
            #print(np.array(f[t_res]))

            frames_all.append(int(np.array(f[frames])[0][0]))
            vertices_all.append(np.array(f[vertices]).T)
            faces = np.array(f[faces]) # note: indexing for these starts at 0, so no subtraction of 1 needed
            faces_all.append(_modify_faces(faces))

            #print('surface_count', np.array(f[surface_count]))
            #print('surface_index_imaris', np.array(f[surface_index_imaris]))
            #print('voxel_size', np.array(f[voxel_size]))
            #print('t_res', np.array(f[t_res]))

    return frames_all, voxels_all, vertices_all, faces_all



def _voxelize(vertices, faces):
    surf = pv.PolyData(vertices, faces)
    surf = surf.decimate(0.95)

    #surf = pv.voxelize(surf, density=0.5, check_surface=False)
    x_min, x_max, y_min, y_max, z_min, z_max = surf.bounds
    x = np.arange(x_min, x_max, 0.5)
    y = np.arange(y_min, y_max, 0.5)
    z = np.arange(z_min, z_max, 0.5)
    xx, yy, zz = np.meshgrid(x, y, z)
    xx = np.moveaxis(xx, 0, 1)
    yy = np.moveaxis(yy, 0, 1)
    zz = np.moveaxis(zz, 0, 1)

    # Create unstructured grid from the structured grid
    grid = pv.StructuredGrid(xx, yy, zz)
    ugrid = pv.UnstructuredGrid(grid)
    # get part of the mesh within the mesh's bounding surface.
    selection = ugrid.select_enclosed_points(surf.extract_surface(), tolerance=0.0, check_surface=False)
    mask = selection.point_arrays['SelectedPoints'].view(np.bool)
    mask = mask.astype(int)

    a = np.zeros_like(xx)
    for i in range(grid.points.shape[0]):
        idx_x = int(xx.shape[0]*(grid.points[i, 0]-x_min)/(x_max-x_min))
        idx_y = int(xx.shape[1]*(grid.points[i, 1]-y_min)/(y_max-y_min))
        idx_z = int(xx.shape[2]*(grid.points[i, 2]-z_min)/(z_max-z_min))
        a[idx_x, idx_y, idx_z] = mask[i]
    return a



def check_voxels(voxels):
    lw, num = measurements.label(voxels)
    area = measurements.sum(voxels, lw, index=np.arange(lw.max() + 1))
    area = list(area)
    if num != 1:
        print('normal')
        print('num: {}'.format(num), 'area: {}'.format(area))
        print('---')

    voxels = 1 - voxels

    lw, num = measurements.label(voxels)
    area = measurements.sum(voxels, lw, index=np.arange(lw.max() + 1))
    area = list(area)
    if num != 1:
        print('inverted')
        print('num: {}'.format(num), 'area: {}'.format(area))
        print('---')


def write_all_zoomed_niigz(mat_filename, save_format, voxelize = False, zoom_factor = 0.2, zeiss_type = False, idx_cell = None, xyz_res = None):
    frames_all, voxels_all, vertices_all, faces_all = get_attribute_from_mat(mat_filename=mat_filename, zeiss_type=zeiss_type, idx_cell=idx_cell, include_voxels = True)

    for idx in range(len(frames_all)):

        if not os.path.exists(save_format.format(int(frames_all[idx]))):

            print(idx)


            if voxelize:
                voxels_cleaned = _voxelize(vertices_all[idx], faces_all[idx])

                check_voxels(voxels_cleaned)



                xyz_res = np.array([0.5, 0.5, 0.5])
                coordinates = np.argwhere(voxels_cleaned == 1)*xyz_res


                plotter = pv.Plotter()

                point_cloud = pv.PolyData(coordinates)
                plotter.add_mesh(point_cloud)


                vertices = vertices_all[idx]

                coordinates += np.min(vertices, axis = 0) + np.array([5, 0, 0])
                point_cloud = pv.PolyData(coordinates)
                plotter.add_mesh(point_cloud)



                #vertices -= np.min(vertices, axis = 0)
                surf = pv.PolyData(vertices, faces_all[idx])
                plotter.add_mesh(surf, color = (1, 0, 1), opacity = 0.1)
                plotter.add_lines(np.array([[-10, 0, 0], [10, 0, 0]]), color = (0, 0, 0))
                plotter.add_lines(np.array([[0, -10, 0], [0, 10, 0]]), color = (0, 0, 0))
                plotter.add_lines(np.array([[0, 0, -10], [0, 0, 10]]), color = (0, 0, 0))
                plotter.show()




            else:
                voxels = np.array(voxels_all[idx])
                lw, num = measurements.label(voxels)
                area = measurements.sum(voxels, lw, index=np.arange(lw.max() + 1))
                area = list(area)

                voxels_cleaned = np.zeros_like(voxels)

                voxels_cleaned[lw == area.index(max(area))] = 1
                voxels_cleaned = binary_fill_holes(voxels_cleaned).astype(int)

                #lw, num = measurements.label(voxels_cleaned)
                #area = measurements.sum(voxels_cleaned, lw, index=np.arange(lw.max() + 1))

                voxels_cleaned = zoom(voxels_cleaned, (zoom_factor, zoom_factor, zoom_factor), order = 0) # order 0 means not interpolation

            #print(save_format.format(int(frames_all[idx])))
            #new_image = nib.Nifti1Image(voxels_cleaned, affine=np.eye(4))
            #nib.save(new_image, save_format.format(int(frames_all[idx])))



def copy_voxels_notDone(doneDir, toCopyDir):
    """
    Compare doneDir & toCopyDir, and copy discrepancies into ~/Desktop/RUNNING/in/
    """

    done_paths = glob.glob(doneDir+'*')

    done_idxs = [int(os.path.basename(i).split('_')[1]) for i in done_paths]

    for path in glob.glob(toCopyDir+'*'):
        idx = int(os.path.basename(path).split('_')[1].split('.')[0])

        if not idx in done_idxs:
            #copyfile(path, '/Users/harry/Desktop/RUNNING/in/' + os.path.basename(path))\

            voxels = read_niigz(path)

            lw, num = measurements.label(voxels)
            area = measurements.sum(voxels, lw, index=np.arange(lw.max() + 1))
            print('num', num, 'area', area)

            new_image = nib.Nifti1Image(voxels, affine=np.eye(4))
            nib.save(new_image, '/Users/harry/Desktop/RUNNING/in/' + os.path.basename(path))



def rm_done_from_inDir():

    for file_path1 in glob.glob('/Users/harry/Desktop/RUNNING/in/*'):
        base1 = os.path.basename(file_path1)[:-7]
        for file_path2 in glob.glob('/Users/harry/Desktop/RUNNING/out/Step1_SegPostProcess/*'):
            base2 = os.path.basename(file_path2)[:-8]
            if base1 == base2:
                if os.path.exists(file_path1):
                    print('Removing: {}'.format(file_path1))
                    os.remove(file_path1)



def copy_coefs_into_dir(outDir, idx_cell):
    for file in glob.glob('/Users/harry/Desktop/RUNNING/out/Step3_ParaToSPHARMMesh/{}_*SPHARM.coef'.format(idx_cell)):
        os.rename(file, outDir + os.path.basename(file))
        print(outDir + os.path.basename(file))





def save_calibrations(idx_cell, cell_no, xyz_res = (0.145, 0.145, 0.4)):
    """
    Save center of masses for zm data where voxels are floating
    """


    files = glob.glob('/Users/harry/OneDrive - Imperial College London/lymphocytes/good_seg_data_3/ZeissLLS/many/cell_{}/voxels_processed/*'.format(cell_no))


    calibrations = {}

    for file in files:
        voxels = nib.load(file)
        voxels = np.moveaxis(np.moveaxis(voxels.dataobj, 0, -1), 0, 1)
        coordinates = np.argwhere(voxels == 1)*np.array(xyz_res) + 0.5*np.array(xyz_res)
        x, y, z = coordinates.sum(0) / np.sum(voxels)
        voxels_centroid = np.array([x, y, z])
        calibrations[int(os.path.basename(file).split('_')[1].split('.')[0])] = voxels_centroid # 'centroid' is the mesh one

    pickle_out = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/calibrations/cell_{}.pickle'.format(idx_cell),'wb')
    pickle.dump(calibrations, pickle_out)




if __name__ == "__main__":

    """
    path = '/Users/harry/OneDrive - Imperial College London/lymphocytes/good_seg_data_3/ZeissLLS/many/cell_0/zoomedVoxels_0.2/0_178.nii.gz'
    voxels = nib.load(path)
    voxels = np.array(voxels.dataobj)

    lw, num = measurements.label(voxels)
    area = measurements.sum(voxels, lw, index=np.arange(lw.max() + 1))
    print('num', num, 'area', area)

    sys.exit()
    """



    """
    for idx_cell in range(10):
        print(idx_cell)
        save_calibrations(idx_cell = 'zm_3_2_{}'.format(idx_cell), cell_no = idx_cell, xyz_res = (0.145, 0.145, 0.4))
        #rm_done_from_inDir()
    sys.exit()
    """






    #mat_filename = '/Users/harry/OneDrive - Imperial College London/lymphocytes/good_seg_data_3/210428/'
    #mat_filename = '/Users/harry/OneDrive - Imperial College London/lymphocytes/good_seg_data_3/ZeissLLS/many/20210828_ot1gfp in collagen-01_clean.mat'
    mat_filename = '../../good_seg_data_3/ZeissLLS_2/20210828_ot1gfp in collagen-01-lattice lightsheet-11-3_clean_Export_Multi_Surf.mat'
    #save_format = '/Users/harry/OneDrive - Imperial College London/lymphocytes/good_seg_data_3/ZeissLLS_4/cell_{}/zoomedVoxels_dist0.5/{}_{{}}.nii.gz'.format(idx_cell, idx_cell)
    save_format = 'none'
    idx_cell = 0
    write_all_zoomed_niigz(mat_filename = mat_filename, save_format = save_format, zoom_factor = None, voxelize = True, zeiss_type = 'zeiss_many', idx_cell = idx_cell, xyz_res = [0.5, 0.5, 0.5])
    for idx_cell in [0, 1, 2]:
        outDir =  '/Users/harry/OneDrive - Imperial College London/lymphocytes/good_seg_data_3/ZeissLLS_3/cell_{}/coeffs/'.format(idx_cell)
        copy_coefs_into_dir(outDir = outDir, idx_cell=idx_cell)




    #rm_done_from_inDir('/Users/harry/Desktop/RUNNING/STACK2/', '/Users/harry/Desktop/RUNNING/out/Step1_SegPostProcess/')
