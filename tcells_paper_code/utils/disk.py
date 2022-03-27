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
    Returns attributes: frames, voxels, vertices, faces
    """
    f = h5py.File(mat_filename, 'r')

    frames_all, voxels_all, vertices_all, faces_all = [], [], [], []

    #if zeiss_type == 'not_zeiss':
    OUT_group = f.get('SURF')

    #frames = f['SURF/FRAME']

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
        if zeiss_type == 'not_zeiss':
            faces = np.array(f[faces_ref[0][0]]) - 1 # note: indexing for these starts at 1, so subtraction of 1 needed
        elif zeiss_type == 'zeiss':
            faces = np.array(f[faces_ref[0][0]])
        faces_all.append(_modify_faces(faces))

    return frames_all, voxels_all, vertices_all, faces_all



def _voxelize(vertices, faces, dx = 0.5, decimate = True):
    surf = pv.PolyData(vertices, faces)
    if decimate:
        surf = surf.decimate(0.95)

    #surf = pv.voxelize(surf, density=0.5, check_surface=False)
    x_min, x_max, y_min, y_max, z_min, z_max = surf.bounds
    x = np.arange(x_min, x_max, dx)
    y = np.arange(y_min, y_max, dx)
    z = np.arange(z_min, z_max, dx)
    xx, yy, zz = np.meshgrid(x, y, z)
    xx = np.moveaxis(xx, 0, 1)
    yy = np.moveaxis(yy, 0, 1)
    zz = np.moveaxis(zz, 0, 1)

    # Create  unstructured grid from the structured grid
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
    frames_all, voxels_all, vertices_all, faces_all = get_attribute_from_mat(mat_filename=mat_filename, idx_cell=idx_cell, include_voxels = True)

    for idx in range(len(frames_all)):

        if not os.path.exists(save_format.format(int(frames_all[idx]))):

            print(idx)


            if voxelize:
                voxels_cleaned = _voxelize(vertices_all[idx], faces_all[idx])

                check_voxels(voxels_cleaned)

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
            new_image = nib.Nifti1Image(voxels_cleaned, affine=np.eye(4))
            nib.save(new_image, save_format.format(int(frames_all[idx])))



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



def copy_coefs_into_dir(outDir, idx_cell):
    for file in glob.glob('/Users/harry/Desktop/RUNNING/out/Step3_ParaToSPHARMMesh/{}_*SPHARM.coef'.format(idx_cell)):
        os.rename(file, outDir + os.path.basename(file))
        print(outDir + os.path.basename(file))
