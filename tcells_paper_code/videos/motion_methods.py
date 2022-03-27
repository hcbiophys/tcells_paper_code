import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import pickle
import sys
from scipy.special import sph_harm
from sklearn.decomposition import PCA
import glob
import os
from scipy.linalg import eig
import math

import lymphocytes.utils.plotting as utils_plotting
import lymphocytes.utils.general as utils_general
import lymphocytes.utils.disk as utils_disk


class Motion_Methods:
    """
    Methods involving e.g. the uropod and centroid
    """


    def _set_mean_uropod_and_centroid(self, idx_cell, time_either_side, max_time_either_side = 50):
        """
        set running means of uropod and centroid, based on tau_sig
        """

        video = self.cells[idx_cell]
        if time_either_side > max_time_either_side or np.isnan(time_either_side): # insignificant uropod motion, label so we can still include where required, e.g. graph of cumulatives
            for frame in video:
                frame.insignificant_uropod_motion = True
            time_either_side = max_time_either_side


        dict = utils_general.get_frame_dict(video)
        idx_frames = list(dict.keys())
        for frame in video:
            idx_frame = frame.idx_frame
            fs = [idx_frame-i for i in reversed(range(1, int(time_either_side//frame.t_res)+1))] + [idx_frame] + [idx_frame+i for i in range(1, int(time_either_side//frame.t_res)+1)]
            uropods = []
            centroids = []
            for f in fs:
                if f in idx_frames:
                    uropods.append(dict[f].uropod)
                    centroids.append(dict[f].centroid)

            if len(uropods) == len(fs):
                frame.mean_uropod = np.mean(np.array(uropods), axis = 0)
                frame.mean_centroid = np.mean(np.array(centroids), axis = 0)




    def _set_speed(self, idx_cell = None):
        """
        Set all the variables to do with speed: delta_uropod, speed_uropod; delta_centroid, speed_centroid
        """

        for video in self.cells.values():
            if video[0].idx_cell == idx_cell or idx_cell is None:
                dict = utils_general.get_frame_dict(video)
                idx_frames = list(dict.keys())

                for frame in video:

                    if frame.mean_centroid is not None and frame.mean_uropod is not None:
                        idx_frame_1 = frame.idx_frame
                        idx_frame_2 = frame.idx_frame + 1
                        if idx_frame_2 in idx_frames and dict[idx_frame_2].mean_centroid is not None and dict[idx_frame_2].mean_uropod is not None:
                            cbrt_vol_times_t_res = np.cbrt(dict[idx_frame_1].volume)*dict[idx_frame_1].t_res
                            vec2 = dict[idx_frame_1].mean_centroid - dict[idx_frame_1].mean_uropod

                            vec1 = dict[idx_frame_2].mean_uropod - dict[idx_frame_1].mean_uropod
                            dict[idx_frame_1].delta_uropod = vec1/cbrt_vol_times_t_res
                            cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                            run = np.linalg.norm(vec1*cos_angle)
                            run /= cbrt_vol_times_t_res
                            run *= np.sign(cos_angle)
                            dict[idx_frame_1].speed_uropod = run


                            vec1 = dict[idx_frame_2].mean_centroid - dict[idx_frame_1].mean_centroid
                            dict[idx_frame_1].delta_centroid = vec1/cbrt_vol_times_t_res
                            cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                            run = np.linalg.norm(vec1*cos_angle)
                            run /= cbrt_vol_times_t_res
                            run *= np.sign(cos_angle)
                            dict[idx_frame_1].speed_centroid = run




    def _set_speed_uropod_running_means(self, idx_cell = None, time_either_side = None):
        """
        Compute running mean of speed_uropod time series, for characterising as run or stop mode
        """

        for video in self.cells.values():
            for frame in video:
                frame.speed_uropod_running_mean = None

        for video in self.cells.values():
            if video[0].idx_cell == idx_cell or idx_cell is None:
                dict = utils_general.get_frame_dict(video)
                idx_frames = list(dict.keys())
                for frame in video:
                    idx_frame = frame.idx_frame
                    if time_either_side == -1:
                        frame.speed_uropod_running_mean = frame.speed_uropod
                    else:
                        fs = [idx_frame-i for i in reversed(range(1, int(time_either_side//frame.t_res)+1))] + [idx_frame] + [idx_frame+i for i in range(1, int(time_either_side//frame.t_res)+1)]
                        speed_uropods = []
                        for f in fs:
                            if f in idx_frames:
                                if dict[f].speed_uropod is not None:
                                    speed_uropods.append(dict[f].speed_uropod)
                        if len(speed_uropods) == len(fs):
                            frame.speed_uropod_running_mean = np.mean(np.array(speed_uropods), axis = 0)




    def unit_vector(self, vec):
        return vec/np.linalg.norm(vec)

    def _set_rotation(self, time_either_side):
        """
        Set variables to do with rotation: ellipsoid_vec (ellipsoid major axis),  ellipsoid_vec_smoothed (ellipsoid major axis running mean), spin_vec ('spin' vector describing the ellipsoidal turning), turning (rotation angle)
        """

        for video in self.cells.values():
            ellipsoids_dict = {}
            latest_vec0 = None
            for frame in video:

                if frame.coeffPathFormat is not None:
                    #frame.surface_plot(plotter=plotter, opacity = 0.5) # plot cell surface
                    #frame.plotRecon_singleDeg(plotter=plotter, max_l = 1, opacity = 0.5) # plot ellipsoid (i.e. l_max = 1)

                    x, y, z, = 0, 1, 2


                    A = np.array([[frame._get_clm(x, 1, 0), frame._get_clm(y, 1, 0), frame._get_clm(z, 1, 0)],
                                    [frame._get_clm(x, 1, 1).real, frame._get_clm(y, 1, 1).real, frame._get_clm(z, 1, 1).real],
                                    [frame._get_clm(x, 1, 1).imag, frame._get_clm(y, 1, 1).imag, frame._get_clm(z, 1, 1).imag]])
                    A = A.real
                    M = np.array([[0, -0.345494, 0],
                                    [0, 0, -0.345494],
                                    [0.488603, 0, 0]])
                    MA = np.matmul(M, A)
                    MA = MA.T
                    to_eig = np.matmul(MA, MA.T)

                    vals, vecs = np.linalg.eig(to_eig)
                    lambdas = [np.sqrt(i) for i in vals]
                    lengths = np.array([i*2 for i in lambdas])

                    new_order = np.argsort(lengths)[::-1]
                    lengths = lengths[new_order]
                    vecs = vecs[:, new_order]


                    frame.ellipsoid_length = lengths[0]
                    frame.ellipsoid_vec = vecs[:, 0]


                    if latest_vec0 is not None:
                        cos_angle = np.dot(latest_vec0, frame.ellipsoid_vec)/(np.linalg.norm(latest_vec0)*np.linalg.norm(frame.ellipsoid_vec))
                        if cos_angle < 0:
                            frame.ellipsoid_vec = -frame.ellipsoid_vec


                    latest_vec0 = frame.ellipsoid_vec



            # interpolate if 1 missing
            dict = utils_general.get_frame_dict(video)
            idx_frames = list(dict.keys())
            for idx_frame in range(int(min(idx_frames)), int(max(idx_frames))):
                if idx_frame in idx_frames and idx_frame-1 in idx_frames and idx_frame+1 in idx_frames:
                    if dict[idx_frame].ellipsoid_length is None and dict[idx_frame-1].ellipsoid_length is not None and dict[idx_frame+1].ellipsoid_length is not None:
                        dict[idx_frame].ellipsoid_length = (dict[idx_frame-1].ellipsoid_length + dict[idx_frame+1].ellipsoid_length)/2
                        dict[idx_frame].ellipsoid_vec = (dict[idx_frame-1].ellipsoid_vec + dict[idx_frame+1].ellipsoid_vec)/2
                        dict[idx_frame].ellipsoid_vec = [self.unit_vector(vec) for vec in dict[idx_frame].ellipsoid_vec]




        if time_either_side == -1:
            for video in self.cells.values():
                for frame in video:
                    frame.ellipsoid_vec_smoothed = frame.ellipsoid_vec

        # if wanting to find ellipsoid_vec_smoothed
        else:

            for video in self.cells.values():
                dict = utils_general.get_frame_dict(video)
                idx_frames = list(dict.keys())

                for frame in video:

                    idx_frame = frame.idx_frame
                    fs = [idx_frame-i for i in reversed(range(1, int(time_either_side//frame.t_res)+1))] + [idx_frame] + [idx_frame+i for i in range(1, int(time_either_side//frame.t_res)+1)]
                    ellipsoid_vecs = []
                    for f in fs:
                        if f in idx_frames and dict[f].ellipsoid_vec is not None:
                            ellipsoid_vecs.append(dict[f].ellipsoid_vec)

                    if len(ellipsoid_vecs) == len(fs):
                        vecs_stacked = np.vstack(ellipsoid_vecs)
                        vec_mean = np.mean(vecs_stacked, axis = 0)
                        dict[idx_frame].ellipsoid_vec_smoothed = self.unit_vector(vec_mean)

                    else:
                        dict[idx_frame].ellipsoid_vec_smoothed = None


        # find spin_vec & turning using the smoothed ellipsoidal vectors
        for video in self.cells.values():
            dict = utils_general.get_frame_dict(video)
            idx_frames = list(dict.keys())


            for frame in video:
                idx_frame = frame.idx_frame

                if idx_frame+1 in idx_frames and dict[idx_frame].ellipsoid_vec_smoothed is not None and dict[idx_frame+1].ellipsoid_vec_smoothed is not None:
                    vec1 = dict[idx_frame].ellipsoid_vec_smoothed
                    vec2 = dict[idx_frame+1].ellipsoid_vec_smoothed


                    cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                    if cos_angle < 0:
                        vec1  = - vec1


                    cross_norm = np.linalg.norm(np.cross(vec1, vec2))
                    angle = np.arcsin(cross_norm/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
                    angle /= dict[idx_frame].t_res

                    dict[idx_frame].spin_vec = angle*np.cross(vec1, vec2) /cross_norm
                    dict[idx_frame].turning = np.linalg.norm(dict[idx_frame].spin_vec)





    def _set_morph_derivs(self, time_either_side = 12):
        """
        Describes how much the morphology is changing
        """

        for video in self.cells.values():
            dict = utils_general.get_frame_dict(video)
            idx_frames = list(dict.keys())
            for frame in video:
                idx_frame = frame.idx_frame
                fs = [idx_frame-i for i in reversed(range(1, int(time_either_side//frame.t_res)+1))] + [idx_frame] + [idx_frame+i for i in range(1, int(time_either_side//frame.t_res)+1)]
                morphs = []
                for f in fs:
                    if f in idx_frames:
                        morphs.append(dict[f].RI_vector[1:])
                morph_derivs = []
                morph_derivs_low = []
                morph_derivs_high = []
                for idx in range(1, len(morphs)):
                    morph_derivs.append(np.linalg.norm(morphs[idx]-morphs[idx-1]))
                    morph_derivs_low.append(np.linalg.norm(morphs[idx][:2]-morphs[idx-1][:2]))
                    morph_derivs_high.append(np.linalg.norm(morphs[idx][2:]-morphs[idx-1][2:]))
                if len(morph_derivs) == len(fs)-1: # since derivative chops of 1 element
                    frame.morph_deriv = np.mean(np.array(morph_derivs), axis = 0)
                    frame.morph_deriv /= frame.t_res # DOES THIS ALSO NEED TO BE NORMALIZED BY VOLUME? No, because RI_vector already is
                    frame.morph_deriv_low = np.mean(np.array(morph_derivs_low), axis = 0)
                    frame.morph_deriv_low /= frame.t_res
                    frame.morph_deriv_high = np.mean(np.array(morph_derivs_high), axis = 0)
                    frame.morph_deriv_high /= frame.t_res
