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

from tcells_paper_code.videos.pca_methods import PCA_Methods
from tcells_paper_code.videos.single_cell_methods import Single_Cell_Methods
from tcells_paper_code.videos.motion_methods import Motion_Methods
from tcells_paper_code.frames.frame_class import Frame
from tcells_paper_code.morphodynamics.consecutive_frames_class import Consecutive_Frames
from tcells_paper_code.videos.curvature_lists import all_lists

import tcells_paper_code.utils.disk as utils_disk
import tcells_paper_code.utils.plotting as utils_plotting
import tcells_paper_code.utils.general as utils_general


def deg_to_rad(angle):
    """
    Convert degrees to radians
    """
    return (6.283/360)*angle


def plane(frame, num_points, color):
    """
    To visualise the plane perpendicular to the uropod
    """

    surf = pv.PolyData(frame.vertices, frame.faces)
    idxs = surf.find_closest_point(frame.uropod, n = num_points)
    vertices = [frame.vertices[idx, :] for idx in idxs]

    pca_obj = PCA(n_components = 2)
    pca_obj.fit(np.array(vertices))
    normal = np.cross(pca_obj.components_[0, :], pca_obj.components_[1, :])


    #plane = pv.Plane(center=frame.uropod, direction=normal, i_size=4, j_size=4)
    #plotter.add_mesh(plane, opacity = 0.5, color = color)

    vec1 = normal
    vec2 = frame.uropod - frame.centroid
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


def get_curvature(frame, angle_half_error, plot = False):
    """
    Get the curvature around the uropod label
    Args:
    - frame: the frame object with surface segmentation to be analysed
    - angle_half_error: angular uncertainty
    - plot: whether to plot or not
    """

    surf = pv.PolyData(frame.vertices, frame.faces)

    surf = surf.smooth(n_iter=5000)
    surf = surf.decimate(0.98)
    curvature = surf.curvature()
    #surf_tree = KDTree(surf.points.astype(np.double))
    #dist, idx = surf_tree.query(np.array([[frame.uropod[0], frame.uropod[1], frame.uropod[2]]]))

    curvatures = surf.curvature('Mean')

    idxs = surf.find_closest_point(frame.uropod, n = 15)
    mean_curvature = np.mean([curvatures[idx] for idx in idxs])

    if angle_half_error is None:
        l_cord, delta_u = None, None
    else:
        l_cord = 2*(1/mean_curvature)*np.sin(angle_half_error/2)
        delta_u = (1/mean_curvature)*angle_half_error

    if plot:
        plotter = pv.Plotter()
        plotter.add_mesh(surf, scalars = curvatures, clim = [-0.5, 0.5], opacity = 0.5)
        #plotter.add_lines(np.array([frame.uropod, frame.centroid]), color = (0, 0, 1), width = 5)
        points = [surf.points[idx] for idx in idxs]
        for point in points:
            plotter.add_mesh(pv.Sphere(radius=0.05, center=point), color = (0, 0, 1))
        plotter.show()


    return mean_curvature, l_cord, delta_u



def save_curvatures(cells, idx_cells):
    """
    Save the curvatures of cells with indices idx_cells
    """


    for idx_cell in idx_cells:
        curvatures_dict = {}
        print(idx_cell)
        video = cells.cells[idx_cell]


        subsample = int(len(video)/10)
        video_sub = video[::subsample]
        frames = []
        curvatures = []
        for frame in video_sub:
            mean_curvature, l_cord, delta_u = get_curvature(frame, angle_half_error = None, plot = True)
            frames.append(frame.frame)
            curvatures.append(mean_curvature)

        curvatures_dict[idx_cell] = zip(frames, curvatures)

        pickle_out = open('../data/curvatures/{}.pickle'.format(video[0].idx_cell),'wb')
        pickle.dump(curvatures_dict, pickle_out)

        del cells.cells[video[0].idx_cell]






def get_possible_points(frame, l_cord):
    """
    Get the points within the uncertainty bound of the uropod label
    """

    possible_points = []
    surf = pv.PolyData(frame.vertices, frame.faces)
    for point in surf.points:
        if np.linalg.norm(point - frame.uropod) < l_cord:

            possible_points.append(point)
    return possible_points



def get_D0s(frame, possible_points):
    """
    Get the possible D_0 (first variable in the shape descriptor) values given the uropod uncertainty
    """

    UCs = []
    for point in possible_points:
        UC = np.linalg.norm(frame.centroid - point)
        UCs.append(UC)

    D0s = [(3/2)*UC/np.cbrt(frame.volume) for UC in UCs]
    return D0s


def get_tau_sig(idx_cell, video):
    """
    Get the timescale
    """
    angle_half_error = 6.283/16
    #angle = angle_half_error*2
    angle = angle_half_error


    frame_lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    dict = utils_general.get_frame_dict(video)
    frames = list(dict.keys())

    curvatures = pickle.load(open('../data/curvatures/{}.pickle'.format(idx_cell), "rb"))
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

    tau_sig = np.mean(frame_diffs)
    tau_sig *= video[0].t_res
    return tau_sig





def save_PC_uncertainties(cells, idx_cells):
    D0_percentage_uncertainties = []
    angle_half_error = 6.283/16


    PC_uncertainties_dict = {}


    for idx_cell in idx_cells:
        video = cells.cells[idx_cell]
        dict = utils_general.get_frame_dict(video)
        frames = list(dict.keys())


        pc0_uncertainties, pc1_uncertainties, pc2_uncertainties = [], [], []
        for frame in video[::10]:
            if not frame.is_interpolation:

                print('Doing', frame.idx_cell, frame.frame)

                curvatures = pickle.load(open('../data/curvatures/{}.pickle'.format(frame.idx_cell), "rb"))
                curvatures = curvatures[frame.idx_cell]
                mean_curvature = np.mean([j for i,j in curvatures])
                l_cord = 2*(1/mean_curvature)*np.sin(angle_half_error/2)
                delta_u = (1/mean_curvature)*angle_half_error

                angle_plane = plane(frame, num_points=50, color = (1, 0, 0)) # for original, not subsampled, mesh

                possible_points = get_possible_points(frame, l_cord)
                if len(possible_points) == 0:
                    break


                D0s = get_D0s(frame, possible_points)

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
                frame.surface_plot(plotter=plotter, uropod_align=False, with_uropod = False)
                for point in possible_points:
                    plotter.add_mesh(pv.Sphere(radius=0.05, center=point), color = (0, 0, 1))
                plotter.show()
                """




        if len(video) > 0:
            PC_uncertainties_dict[video[0].idx_cell] = [np.mean(i) for i in [pc0_uncertainties, pc1_uncertainties, pc2_uncertainties]]

        del cells.cells[video[0].idx_cell]

    print('np.mean(D0_percentage_uncertainties)', np.mean(D0_percentage_uncertainties))


    pickle.dump(PC_uncertainties_dict, open('../data/PC_uncertainties.pickle', 'wb'))
