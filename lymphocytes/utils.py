import sys
import os
import numpy as np
import glob
from shutil import copyfile
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import measurements
from skimage import morphology
from scipy.ndimage.morphology import binary_fill_holes
import scipy.io
import vtk
from vtk import vtkSphereSource, vtkPolyData, vtkDecimatePro
import h5py
from vtk.util.numpy_support import vtk_to_numpy
import trimesh
from scipy.spatial import Delaunay
from collections import defaultdict
from skimage import measure
from scipy.ndimage.morphology import binary_dilation
import matplotlib


from lymphocyte_snap_class import *

#np.set_printoptions(threshold=sys.maxsize)


def alpha_shape_3D(pos,alpha):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos - np.array of shape (n,3) points.
        alpha - alpha value.
    return
        outer surface vertex indices, edge indices, and triangle indices
    """
    print(pos.shape)
    tetra = Delaunay(pos)
    tetra_vertices = tetra.vertices
    print(tetra_vertices[:4, :])


    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos,tetra_vertices,axis=0)
    normsq = np.sum(tetrapos**2,axis=2)[:,:,None]
    ones = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
    a = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
    Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
    Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
    c = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
    r = np.sqrt(Dx**2+Dy**2+Dz**2-4*a*c)/(2*np.abs(a))

    # Find tetrahedrals
    tetras = tetra_vertices[r<alpha,:]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:,TriComb].reshape(-1,3)
    Triangles = np.sort(Triangles,axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles:TrianglesDict[tuple(tri)] += 1
    Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
    #edges
    EdgeComb=np.array([(0, 1), (0, 2), (1, 2)])
    Edges=Triangles[:,EdgeComb].reshape(-1,2)
    Edges=np.sort(Edges,axis=1)
    Edges=np.unique(Edges,axis=0)

    Vertices = np.unique(Edges)
    return Vertices, Edges, Triangles


def mkVtkIdList(it):
  vil = vtk.vtkIdList()
  for i in it:
    vil.InsertNextId(int(i))
  return vil


def decimate(inputPoly, factor):



    print("Before decimation\n"
          "-----------------\n"
          "There are " + str(inputPoly.GetNumberOfPoints()) + "points.\n"
          "There are " + str(inputPoly.GetNumberOfPolys()) + "polygons.\n")

    decimate = vtkDecimatePro()
    decimate.SetInputData(inputPoly)
    decimate.SetTargetReduction(factor)
    decimate.Update()

    decimatedPoly = vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    print("After decimation \n"
          "-----------------\n"
          "There are " + str(decimatedPoly.GetNumberOfPoints()) + "points.\n"
          "There are " + str(decimatedPoly.GetNumberOfPolys()) + "polygons.\n")

    return decimatedPoly





def decimate_mat_mesh(mat_filename, idx_snap, decimation_factor, show, save_as):
    # '/Users/harry/Desktop/lymphocytes/batch2/Stack2.mat'

    f = h5py.File(mat_filename, 'r')
    dataset = f['DataOut/Surf']


    vertices = np.asarray(f[dataset[3, idx_snap]]).T
    faces = np.asarray(f[dataset[4, idx_snap]]).T
    voxels = f[dataset[2, idx_snap]]


    voxels = keep_only_largest_object(voxels)
    voxels = binary_fill_holes(voxels).astype(int)
    print('before', np.sum(voxels))

    #zf = 0.2
    #voxels = zoom(voxels, (zf, zf, zf))
    #voxels = binary_dilation(voxels)

    #vertices, faces, normals, values = measure.marching_cubes(voxels, 0)

    #idx_vertices, Edges, faces = alpha_shape_3D(pos = vertices, alpha = 0.6)
    #print(vertices.shape)
    #vertices = vertices[idx_vertices]
    #print(vertices.shape)


    #b = trimesh.base.Trimesh(vertices=vertices, faces=faces)
    mesh = trimesh.Trimesh(vertices = vertices, faces = faces, process = True)
    #trimesh.repair.fill_holes(mesh)
    #trimesh.repair.fix_inversion(mesh)
    #trimesh.repair.fix_normals(mesh)
    #trimesh.repair.fix_winding(mesh)
    print('idx', idx_snap)
    print('is_watertight', mesh.is_watertight)
    print('euler_number', mesh.euler_number)
    #mesh.show()


    mesh    = vtk.vtkPolyData()
    points  = vtk.vtkPoints()
    polys   = vtk.vtkCellArray()

    for i in range(vertices.shape[0]):
        points.InsertNextPoint(vertices[i, 0], vertices[i, 1], vertices[i, 2])
    for i in range(faces.shape[0]):
        polys.InsertNextCell(mkVtkIdList(faces[i, :]))

    mesh.SetPoints(points)
    mesh.SetPolys(polys)

    mesh = decimate(mesh, decimation_factor) # decimate

    if show:
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(mesh)
        actor = vtk.vtkActor()
        actor.GetProperty().SetOpacity(0.5)
        actor.SetMapper(mapper)

        renderer = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(renderer)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        iren.SetRenderWindow(renWin)
        renderer.AddActor(actor)
        renWin.SetSize(640, 480)
        renWin.Render()

        renWin.Render()
        iren.Start()

    vertices = vtk_to_numpy(mesh.GetPoints().GetData()).astype('double')


    cells = mesh.GetPolys()
    nCells = cells.GetNumberOfCells()
    array = cells.GetData()
    nCols = array.GetNumberOfValues()//nCells
    cells = vtk_to_numpy(array).astype('double')
    cells = cells.reshape((-1,nCols))[:, 1:]
    cells += 1


    # '/Users/harry/Desktop/lymphocytes/triangMeshes_forMatlab/test.mat'
    scipy.io.savemat(save_as, mdict={'faces': cells,
                                    'mins': np.array([0, 0, 0]),
                                    'vertices': vertices})



    return mesh


def decimate_mat_voxels(mat_filename, idx_snap, decimation_factor, show, save_as):

    f = h5py.File(mat_filename, 'r')
    dataset = f['DataOut/Surf']
    voxels = f[dataset[2, idx_snap]]

    voxels = zoom(voxels, (decimation_factor, decimation_factor, decimation_factor)).astype('double')
    print(voxels.shape)
    print(voxels.dtype)

    scipy.io.savemat(save_as, mdict={'bim': voxels,
                                    'origin': np.array([0, 0, 0]).astype('double'),
                                    'vxsize': np.array([1, 1, 1]).astype('double')})

def load_and_check_nib():

    '/Users/harry/Desktop/RUNNING/in/test.nii.gz'
    voxels = nib.load('/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom1_someStack10Missing/stack2/Stack2_0.nii.gz').get_fdata()

    voxels = 1-voxels

    lw, num = measurements.label(voxels)
    print(voxels.min(), voxels.max())
    print(voxels.shape)
    area = measurements.sum(voxels, lw, index=np.arange(lw.max() + 1))
    print('area', area)
    print('num', num)



def keep_only_largest_object(array):

        labels = morphology.label(array, connectivity = 1)
        labels_num = [len(labels[labels==each]) for each in np.unique(labels)]
        rank = np.argsort(np.argsort(labels_num))
        max_index = list(rank).index(len(rank)-2)
        new_array = np.zeros_like(array)
        new_array[labels == max_index] = 1
        return new_array



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





def plot_all_volumes(lymph_serieses):

    labels = ['2', '3', '4', '5', '7', '9', '10']
    colors = ['red', 'black', 'green', 'blue', 'orange', 'brown', 'pink']
    exit_idxs = [285, 90, 2, 50, 9, 39, 23]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for color, label, exit_idx, lymph_series in zip(colors, labels, exit_idxs, lymph_serieses):

        volumes = []
        for idx, lymph in enumerate(lymph_series.lymphSnaps_dict.values()):
            voxels = lymph.voxels
            volumes.append(np.sum(voxels))
            print('idx', idx)
        ax.plot([i for i in range(len(volumes[:exit_idx]))], volumes[:exit_idx], label = label, c = color)
        ax.plot([exit_idx+i for i in range(len(volumes[exit_idx:]))], volumes[exit_idx:], linestyle = ':', c = color)

    ax.set_xlim([0, 300])
    ax.set_ylim([0, 900000])
    plt.legend()
    plt.show()




def write_all_niigz(zoom_factor):

    # [ 'Stack4.mat', 'Stack10.mat', 'Stack3.mat', 'Stack5.mat', 'Stack9.mat', 'Stack2.mat', 'Stack7.mat']
    # [175, 210, 300, 115, 132, 300, 70]

    mat_filenames = ['/Users/harry/Desktop/lymphocytes/batch1/' + i for i in ['405s2/mat/405s2.mat', '405s3/mat/405s3.mat', '406s2/mat/406s2.mat']]
    lengths = [50, 60, 20]

    for mat_filename, length in zip(mat_filenames, lengths):
        print('mat_filename: ', mat_filename)
        for idx in range(length):
            print('idx :', idx)
            snap = LymphocyteSnap(mat_filename, 'dummy', 'dummy', idx)
            snap.ORIG_write_voxels_to_niigz('/Users/harry/Desktop/lymphocytes/batch1/niigz_zoom0.08', zoom_factor = zoom_factor)


def equal_axes(*axes):


    x_mids = []
    y_mids = []
    z_mids = []
    ax_ranges = []

    for ax in axes:
        x_min, x_max = ax.get_xlim()
        ax_ranges.append(x_max - x_min)
        x_mids.append((x_min + x_max)/2)
        y_min, y_max = ax.get_ylim()
        ax_ranges.append(y_max - y_min)
        y_mids.append((y_min + y_max)/2)
        z_min, z_max = ax.get_zlim()
        ax_ranges.append(z_max - z_min)
        z_mids.append((z_min + z_max)/2)

    max_range = np.array(ax_ranges).max()

    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(x_mids[ax_idx]-(max_range/2), x_mids[ax_idx]+(max_range/2))
        ax.set_ylim(y_mids[ax_idx]-(max_range/2), y_mids[ax_idx]+(max_range/2))
        #ax.set_xlim([-10, 10])
        #ax.set_ylim([-10, 10])
        ax.set_zlim(z_mids[ax_idx]-(max_range/2), z_mids[ax_idx]+(max_range/2))




def equal_axes_notSquare(*axes):

    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    z_mins = []
    z_maxs = []

    for ax in axes:
        x_min, x_max = ax.get_xlim()
        x_mins.append(x_min)
        x_maxs.append(x_max)
        y_min, y_max = ax.get_ylim()
        y_mins.append(y_min)
        y_maxs.append(y_max)
        z_min, z_max = ax.get_zlim()
        z_mins.append(z_min)
        z_maxs.append(z_max)

    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(min(x_mins), max(x_maxs))
        ax.set_ylim(min(y_mins), max(y_maxs))
        ax.set_zlim(min(z_mins), max(z_maxs))

def equal_axes_notSquare_2D(*axes):

    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []

    for ax in axes:
        x_min, x_max = ax.get_xlim()
        x_mins.append(x_min)
        x_maxs.append(x_max)
        y_min, y_max = ax.get_ylim()
        y_mins.append(y_min)
        y_maxs.append(y_max)

    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(min(x_mins), max(x_maxs))
        ax.set_ylim(min(y_mins), max(y_maxs))


def remove_ticks(*axes):
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

def find_voxel_ranges(niigz):

    voxels = read_niigz(niigz)

    start = 0
    end = 0
    for idx_slice in range(voxels.shape[0]):
        slice = voxels[idx_slice, :, :]
        if slice.max() == 1:
            start = idx_slice
            break
    for idx_slice in np.arange(start, voxels.shape[0]):
        slice = voxels[idx_slice, :, :]
        if slice.max() == 0:
            end = idx_slice
            break
    x_range = end-start

    start = 0
    end = 0
    for idx_slice in range(voxels.shape[1]):
        slice = voxels[:, idx_slice, :]
        if slice.max() == 1:
            start = idx_slice
            break
    for idx_slice in np.arange(start, voxels.shape[1]):
        slice = voxels[:, idx_slice, :]
        if slice.max() == 0:
            end = idx_slice
            break
    y_range = end-start

    start = 0
    end = 0
    for idx_slice in range(voxels.shape[2]):
        slice = voxels[:, :, idx_slice]
        if slice.max() == 1:
            start = idx_slice
            break
    for idx_slice in np.arange(start, voxels.shape[2]):
        slice = voxels[:, :, idx_slice]
        if slice.max() == 0:
            end = idx_slice
            break
    z_range = end-start

    return x_range, y_range, z_range





def find_optimal_3dview(niigz):

    x_range, y_range, z_range = find_voxel_ranges(niigz)

    ranges = [x_range, y_range, z_range]

    print('shortest: ', ranges.index(min(ranges)))

    """
    if ranges.index(max(ranges)) == 0:
        elev = 0
        azim = 0
    else:
        elev = 0
        azim = 90

    return elev, azim
    """

    if ranges.index(min(ranges)) == 0:
        azim = 0
    if ranges.index(min(ranges)) == 1:
        azim = 90
    else:
        azim = 90

    elev = 0
    return elev, azim


def voxel_volume(niigz):

    voxels = nib.load(niigz).get_fdata()

    voxels = keep_only_largest_object(voxels)
    voxels = binary_fill_holes(voxels).astype(int)

    return np.sum(voxels)

def read_niigz(niigz):

    voxels = nib.load(niigz).get_fdata()

    voxels = keep_only_largest_object(voxels)
    voxels = binary_fill_holes(voxels).astype(int)

    return voxels



def del_whereNone(nestedLists, attribute):

    nestedLists_new = []
    for list in nestedLists:

        list_new = []
        for item in list:
            if attribute == 'coeff_array':
                if item.coeff_array is not None:
                    list_new.append(item)
            elif attribute == 'speed':
                if item.speed is not None:
                    list_new.append(item)
            elif attribute == 'angle':
                if item.angle is not None:
                    list_new.append(item)

        nestedLists_new.append(list_new)

    return nestedLists_new







if __name__ == "__main__":



    decimate_mat_mesh('/Users/harry/Desktop/lymphocytes/batch2/Stack4.mat', 30, 0, True, '/Users/harry/Desktop/lymphocytes/mats_decimated/0_dec0.9.mat')
    #decimate_mat_voxels('/Users/harry/Desktop/lymphocytes/batch2/Stack2.mat', 15, 0.2, True, '/Users/harry/Desktop/lymphocytes/mats_decimated/voxels_dec0.2.mat')

    #copy_voxels_notDone('/Users/harry/Desktop/lymphocytes/batch2/zoom0.2_coeffs/stack2/', '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack2/')


    """
    for path in ['/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack2/Stack2_226.nii.gz', '/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack2/Stack2_227.nii.gz']:

        print('path :', path)
        voxels = nib.load(path).get_fdata()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.voxels(voxels[8:22, 45:80, 80:120])
        plt.show()

        labels = morphology.label(voxels, connectivity = 1)
        print(labels.min(), labels.max())

        #voxels = keep_only_largest_object(voxels)
        #voxels = binary_fill_holes(voxels).astype(int)

        lw, num = measurements.label(voxels)
        area = measurements.sum(voxels, lw, index=np.arange(lw.max() + 1))
        print('num', num, 'area', area)
    """


    #C_plot_cofms('/Users/harry/Desktop/lymphocytes/batch2/niigz_zoom0.2/stack2/')
