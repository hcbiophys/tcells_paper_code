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


class Utils:

    def mkVtkIdList(self, it):
      vil = vtk.vtkIdList()
      for i in it:
        vil.InsertNextId(int(i))
      return vil


    def decimate(self, inputPoly, factor):

        print(str(decimatedPoly.GetNumberOfPoints()), str(decimatedPoly.GetNumberOfPolys()))

        decimate = vtkDecimatePro()
        decimate.SetInputData(inputPoly)
        decimate.SetTargetReduction(factor)
        decimate.Update()

        decimatedPoly = vtkPolyData()
        decimatedPoly.ShallowCopy(decimate.GetOutput())

        print(str(decimatedPoly.GetNumberOfPoints()), str(decimatedPoly.GetNumberOfPolys()))

        return decimatedPoly


    def decimate_mat_mesh(self, mat_filename, idx_snap, decimation_factor, show, save_as):
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
            polys.InsertNextCell(self.mkVtkIdList(faces[i, :]))

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


    def decimate_mat_voxels(self, mat_filename, idx_snap, decimation_factor, show, save_as):

        f = h5py.File(mat_filename, 'r')
        dataset = f['DataOut/Surf']
        voxels = f[dataset[2, idx_snap]]

        voxels = zoom(voxels, (decimation_factor, decimation_factor, decimation_factor)).astype('double')
        print(voxels.shape)
        print(voxels.dtype)

        scipy.io.savemat(save_as, mdict={'bim': voxels,
                                        'origin': np.array([0, 0, 0]).astype('double'),
                                        'vxsize': np.array([1, 1, 1]).astype('double')})


    def write_all_niigz(self, zoom_factor):

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


    def del_whereNone(self, nestedLists, attribute):

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
