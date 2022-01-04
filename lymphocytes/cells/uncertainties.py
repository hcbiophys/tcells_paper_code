import numpy as np
import matplotlib
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
import os
from sklearn.decomposition import PCA
import scipy.stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import pickle
import random
from pykdtree.kdtree import KDTree
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from pyvista import examples
import time

from lymphocytes.cells.pca_methods import PCA_Methods
from lymphocytes.cells.single_cell_methods import Single_Cell_Methods
from lymphocytes.cells.centroid_variable_methods import Centroid_Variable_Methods
from lymphocytes.cell_frame.cell_frame_class import Cell_Frame
from lymphocytes.behavior_analysis.consecutive_frames_class import Consecutive_Frames
from lymphocytes.cells.curvature_lists import all_lists

import lymphocytes.utils.disk as utils_disk
import lymphocytes.utils.plotting as utils_plotting
import lymphocytes.utils.general as utils_general


def deg_to_rad(angle):
    return (6.283/360)*angle



def plane(lymph, num_points, color):

    surf = pv.PolyData(lymph.vertices, lymph.faces)
    idxs = surf.find_closest_point(lymph.uropod, n = num_points)
    vertices = [lymph.vertices[idx, :] for idx in idxs]

    pca_obj = PCA(n_components = 2)
    pca_obj.fit(np.array(vertices))
    normal = np.cross(pca_obj.components_[0, :], pca_obj.components_[1, :])


    #plane = pv.Plane(center=lymph.uropod, direction=normal, i_size=4, j_size=4)
    #plotter.add_mesh(plane, opacity = 0.5, color = color)

    vec1 = normal
    vec2 = lymph.uropod - lymph.centroid
    cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    if cos_angle < 0:
        vec1 = - normal
        cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

    angle_plane = 90 - (360/6.283)*np.arccos(cos_angle)

    #if plot:
    #    points = [surf.points[idx] for idx in idxs]
    #    for point in points:
    #        plotter.add_mesh(pv.Sphere(radius=0.05, center=point), color = color)

    return angle_plane





def get_curvature(lymph, angle_half_error, plot = False):

    surf = pv.PolyData(lymph.vertices, lymph.faces)

    surf = surf.smooth(n_iter=5000)
    surf = surf.decimate(0.98)
    curvature = surf.curvature()
    #surf_tree = KDTree(surf.points.astype(np.double))
    #dist, idx = surf_tree.query(np.array([[lymph.uropod[0], lymph.uropod[1], lymph.uropod[2]]]))

    curvatures = surf.curvature('Mean')

    idxs = surf.find_closest_point(lymph.uropod, n = 15)
    mean_curvature = np.mean([curvatures[idx] for idx in idxs])

    if angle_half_error is None:
        l_cord, delta_u = None, None
    else:
        l_cord = 2*(1/mean_curvature)*np.sin(angle_half_error/2)
        delta_u = (1/mean_curvature)*angle_half_error

    if plot:
        plotter = pv.Plotter()
        plotter.add_mesh(surf, scalars = curvatures, clim = [-0.5, 0.5], opacity = 0.5)
        #plotter.add_lines(np.array([lymph.uropod, lymph.centroid]), color = (0, 0, 1), width = 5)
        points = [surf.points[idx] for idx in idxs]
        for point in points:
            plotter.add_mesh(pv.Sphere(radius=0.05, center=point), color = (0, 0, 1))
        plotter.show()


    return mean_curvature, l_cord, delta_u



def save_curvatures(cells, idx_cells_orig):


    for idx_cell in idx_cells_orig:
        curvatures_dict = {}
        print(idx_cell)
        lymph_series = cells.cells[idx_cell]


        subsample = int(len(lymph_series)/10)
        lymph_series_sub = lymph_series[::subsample]
        frames = []
        curvatures = []
        for lymph in lymph_series_sub:
            mean_curvature, l_cord, delta_u = get_curvature(lymph, angle_half_error = None, plot = True)
            frames.append(lymph.frame)
            curvatures.append(mean_curvature)

        curvatures_dict[idx_cell] = zip(frames, curvatures)

        pickle_out = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/curvatures/cell_{}.pickle'.format(lymph_series[0].idx_cell),'wb')
        pickle.dump(curvatures_dict, pickle_out)

        del cells.cells[lymph_series[0].idx_cell]






def get_possible_points(lymph, l_cord):
    possible_points = []
    surf = pv.PolyData(lymph.vertices, lymph.faces)
    for point in surf.points:
        if np.linalg.norm(point - lymph.uropod) < l_cord:

            possible_points.append(point)
    return possible_points



def get_D0s(lymph, possible_points):
    UCs = []
    for point in possible_points:
        UC = np.linalg.norm(lymph.centroid - point)
        UCs.append(UC)

    D0s = [(3/2)*UC/np.cbrt(lymph.volume) for UC in UCs]
    return D0s






def get_mean_time_diff(idx_cell, lymph_series):
    angle_half_error = 6.283/16
    #angle = angle_half_error*2
    angle = angle_half_error


    frame_lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    dict = utils_general.get_frame_dict(lymph_series)
    frames = list(dict.keys())

    curvatures = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/curvatures/cell_{}.pickle'.format(idx_cell), "rb"))
    curvatures = curvatures[idx_cell]
    mean_curvature = np.mean([j for i,j in curvatures])
    l_cord = 2*(1/mean_curvature)*np.sin(angle/2)


    frame1s = []
    frame_diffs = []
    for idx in range(len(frames)):
        frame1 = frames[idx]
        for frame2 in frames[idx:]:
            if np.linalg.norm(dict[frame1].uropod - dict[frame2].uropod) > 2*l_cord:
                frame1s.append(frame1)
                frame_diffs.append(frame2-frame1)
                break

    mean_time_diff = np.mean(frame_diffs)
    mean_time_diff *= lymph_series[0].t_res
    return mean_time_diff





def save_PC_uncertainties(cells, idx_cells_orig):
    D0_percentage_uncertainties = []
    angle_half_error = 6.283/16


    PC_uncertainties_dict = {}


    for idx_cell in idx_cells_orig:
        lymph_series = cells.cells[idx_cell]
        dict = utils_general.get_frame_dict(lymph_series)
        frames = list(dict.keys())


        pc0_uncertainties, pc1_uncertainties, pc2_uncertainties = [], [], []
        for lymph in lymph_series[::10]:
            if not lymph.is_interpolation:

                print('Doing', lymph.idx_cell, lymph.frame)

                curvatures = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/curvatures/cell_{}.pickle'.format(lymph.idx_cell), "rb"))
                curvatures = curvatures[lymph.idx_cell]
                mean_curvature = np.mean([j for i,j in curvatures])
                l_cord = 2*(1/mean_curvature)*np.sin(angle_half_error/2)
                delta_u = (1/mean_curvature)*angle_half_error

                angle_plane = plane(lymph, num_points=50, color = (1, 0, 0)) # for original, not subsampled, mesh

                possible_points = get_possible_points(lymph, l_cord)
                if len(possible_points) == 0:
                    break


                D0s = get_D0s(lymph, possible_points)

                print('HERE', 100*np.std(D0s)/np.mean(D0s))
                D0_percentage_uncertainties.append(100*np.std(D0s)/np.mean(D0s))

                D0_vec = np.zeros(shape = cells.pca_obj.components_[0].shape)
                D0_vec[0] = 1

                count = 0
                for pc_vec, pc_uncertainty_list in zip(cells.pca_obj.components_, [pc0_uncertainties, pc1_uncertainties, pc2_uncertainties]):

                    cos_angle = np.dot(D0_vec, pc_vec)/(np.linalg.norm(D0_vec)*np.linalg.norm(pc_vec))
                    pc_uncertainty = abs(np.std(D0s)*cos_angle)
                    pc_uncertainty_list.append(pc_uncertainty)

                    print('pc{} uncertainty:'.format(count), pc_uncertainty)
                    count += 1


                """
                plotter = pv.Plotter()
                lymph.surface_plot(plotter=plotter, uropod_align=False, with_uropod = False)
                for point in possible_points:
                    plotter.add_mesh(pv.Sphere(radius=0.05, center=point), color = (0, 0, 1))
                plotter.show()
                """




        if len(lymph_series) > 0:
            PC_uncertainties_dict[lymph_series[0].idx_cell] = [np.mean(i) for i in [pc0_uncertainties, pc1_uncertainties, pc2_uncertainties]]

        del cells.cells[lymph_series[0].idx_cell]

    print('np.mean(D0_percentage_uncertainties)', np.mean(D0_percentage_uncertainties))


    pickle.dump(PC_uncertainties_dict, open('/Users/harry/OneDrive - Imperial College London/lymphocytes/PC_uncertainties.pickle', 'wb'))
