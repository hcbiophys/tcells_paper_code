import numpy as np
import vtk
from vtk import vtkSphereSource, vtkPolyData, vtkDecimatePro
import h5py
from vtk.util.numpy_support import vtk_to_numpy
import trimesh
from collections import defaultdict
from skimage import measure
import sys

import lymphocytes.utils.voxels as utils_voxels

def _mkVtkIdList(it):
  vil = vtk.vtkIdList()
  for i in it:
    vil.InsertNextId(int(i))
  return vil


def _decimate(inputPoly, factor):

    print(str(inputPoly.GetNumberOfPoints()), str(inputPoly.GetNumberOfPolys()))

    decimate = vtkDecimatePro()
    decimate.SetInputData(inputPoly)
    decimate.SetTargetReduction(factor)
    decimate.Update()

    decimatedPoly = vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    print(str(decimatedPoly.GetNumberOfPoints()), str(decimatedPoly.GetNumberOfPolys()))

    return decimatedPoly


def decimate_mat_mesh(mat_filename, idx_snap, decimation_factor, show):
    # '/Users/harry/Desktop/lymphocytes/batch2/Stack2.mat'

    f = h5py.File(mat_filename, 'r')
    OUT_group = f.get('OUT')

    frames = OUT_group.get('FRAME')
    frames = np.array(frames).flatten()
    idx = np.where(frames == idx_snap)

    voxels = OUT_group.get('BINARY_MASK')
    voxels_ref = voxels[idx]
    voxels = f[voxels_ref[0][0]]

    vertices = OUT_group.get('VERTICES')
    vertices_ref = vertices[idx]
    vertices = f[vertices_ref[0][0]]

    faces = OUT_group.get('FACES')
    faces_ref = faces[idx]
    faces = np.asarray(f[faces_ref[0][0]]) - 1


    voxels = utils_voxels.process_voxels(voxels)

    #zf = 0.2
    #voxels = zoom(voxels, (zf, zf, zf), order = 0)
    #voxels = binary_dilation(voxels)

    #vertices, faces, normals, values = measure.marching_cubes(voxels, 0)
    #idx_vertices, Edges, faces = alpha_shape_3D(pos = vertices, alpha = 0.6)


    #b = trimesh.base.Trimesh(vertices=vertices, faces=faces)
    """
    mesh = trimesh.Trimesh(vertices = vertices, faces = faces, process = True)
    """
    #trimesh.repair.fill_holes(mesh)
    #trimesh.repair.fix_inversion(mesh)
    #trimesh.repair.fix_normals(mesh)
    #trimesh.repair.fix_winding(mesh)

    mesh    = vtk.vtkPolyData()
    points  = vtk.vtkPoints()
    polys   = vtk.vtkCellArray()

    for i in range(vertices.shape[1]):
        points.InsertNextPoint(vertices[0, i], vertices[1, i], vertices[2, i])
    for i in range(faces.shape[1]):
        polys.InsertNextCell(_mkVtkIdList(faces[:, i]))

    mesh.SetPoints(points)
    mesh.SetPolys(polys)

    #mesh = _decimate(mesh, decimation_factor) # decimate

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
    #scipy.io.savemat(save_as, mdict={'faces': cells,
                                    #'mins': np.array([0, 0, 0]),
                                    #'vertices': vertices})

    return mesh




if __name__ == "__main__":
    frame = 31
    decimation_factor = 0.8
    decimate_mat_mesh('/Users/harry/Desktop/lymphocytes/good_seg_data_2/20190405_M101_1_5_mg_27_5_deg/stack4/Stack4_BC-T-corr-0_35um_Export_Surf_corr.mat', frame, decimation_factor, show = True)
