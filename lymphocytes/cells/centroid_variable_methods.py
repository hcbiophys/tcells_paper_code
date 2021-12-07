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


class Centroid_Variable_Methods:



    def _set_centroid_attributes(self, attribute, time_either_side = None, time_either_side_2 = 50, idx_cell = None):

        if attribute[:3] == 'run':
            self._set_run(idx_cell = idx_cell)
        elif attribute == 'searching':
             self._set_searching(time_either_side, time_either_side_2)


    def _set_mean_uropod_and_centroid(self, idx_cell, time_either_side):
        lymph_series = self.cells[idx_cell]

        times = [(lymph.frame-lymph_series[0].frame)*lymph.t_res for lymph in lymph_series]

        if np.isnan(time_either_side):
            for lymph in lymph_series:
                lymph.mean_uropod = np.array([np.nan, np.nan, np.nan])
                lymph.mean_centroid = np.array([np.nan, np.nan, np.nan])
        else:
            dict = utils_general.get_frame_dict(lymph_series)
            frames = list(dict.keys())
            for lymph in lymph_series:
                frame = lymph.frame
                fs = [frame-i for i in reversed(range(1, int(time_either_side//lymph.t_res)+1))] + [frame] + [frame+i for i in range(1, int(time_either_side//lymph.t_res)+1)]
                uropods = []
                centroids = []
                for f in fs:
                    if f in frames:
                        uropods.append(dict[f].uropod)
                        centroids.append(dict[f].centroid)
                if len(uropods) >= len(fs)-1:
                    lymph.mean_uropod = np.mean(np.array(uropods), axis = 0)
                    lymph.mean_centroid = np.mean(np.array(centroids), axis = 0)

        if len([i for i in lymph_series if i.mean_uropod is None]) == len(lymph_series):
            for lymph in lymph_series:
                lymph.mean_uropod = np.array([np.nan, np.nan, np.nan])
                lymph.mean_centroid = np.array([np.nan, np.nan, np.nan])



    def _set_run(self, idx_cell = None):

        idx_cells_done = []
        for idx, lymph_series in self.cells.items():
            for lymph in lymph_series:
                if lymph.mean_uropod is not None:
                    if np.isnan(lymph.mean_uropod[0]): # i.e. run_uropod should be set to 0 as there was no significant movement
                        set_to = 0
                        idx_cells_done.append(idx)
                    else:
                        set_to = None # otherwise reset

                    lymph.run_uropod = set_to
                    lymph.delta_uropod = set_to
                    lymph.run_centroid = set_to
                    lymph.delta_centroid = set_to

        idx_cells_done = list(set(idx_cells_done))


        for lymph_series in self.cells.values():
            if lymph_series[0].idx_cell not in idx_cells_done:
                if lymph_series[0].idx_cell == idx_cell or idx_cell is None:
                    dict = utils_general.get_frame_dict(lymph_series)
                    frames = list(dict.keys())



                    for lymph in lymph_series:

                        if lymph.mean_centroid is not None and lymph.mean_uropod is not None:
                            frame_1 = lymph.frame
                            frame_2 = lymph.frame + 1
                            if frame_2 in frames and dict[frame_2].mean_centroid is not None and dict[frame_2].mean_uropod is not None:
                                cbrt_vol_times_t_res = np.cbrt(dict[frame_1].volume)*dict[frame_1].t_res
                                vec2 = dict[frame_1].mean_centroid - dict[frame_1].mean_uropod

                                vec1 = dict[frame_2].mean_uropod - dict[frame_1].mean_uropod
                                dict[frame_1].delta_uropod = vec1/cbrt_vol_times_t_res
                                cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                                run = np.linalg.norm(vec1*cos_angle)
                                run /= cbrt_vol_times_t_res
                                run *= np.sign(cos_angle)
                                dict[frame_1].run_uropod = run


                                vec1 = dict[frame_2].mean_centroid - dict[frame_1].mean_centroid
                                dict[frame_1].delta_centroid = vec1/cbrt_vol_times_t_res
                                cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
                                run = np.linalg.norm(vec1*cos_angle)
                                run /= cbrt_vol_times_t_res
                                run *= np.sign(cos_angle)
                                dict[frame_1].run_centroid = run




    def _set_run_uropod_running_means(self, idx_cell = None, time_either_side = None):

        for lymph_series in self.cells.values():
            for lymph in lymph_series:
                lymph.run_uropod_running_mean = None

        for lymph_series in self.cells.values():
            if lymph_series[0].idx_cell == idx_cell or idx_cell is None:

                dict = utils_general.get_frame_dict(lymph_series)
                frames = list(dict.keys())
                for lymph in lymph_series:
                    frame = lymph.frame
                    if time_either_side == -1:
                        lymph.run_uropod_running_mean = lymph.run_uropod
                    else:
                        fs = [frame-i for i in reversed(range(1, int(time_either_side//lymph.t_res)+1))] + [frame] + [frame+i for i in range(1, int(time_either_side//lymph.t_res)+1)]
                        run_uropods = []
                        for f in fs:
                            if f in frames:
                                if dict[f].run_uropod is not None:
                                    run_uropods.append(dict[f].run_uropod)
                        if len(run_uropods) == len(fs):
                            lymph.run_uropod_running_mean = np.mean(np.array(run_uropods), axis = 0)






    def _set_searching(self, time_either_side, time_either_side_2):

        """
        for lymph_series in self.cells.values():
            ellipsoids_dict = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/ellipsoids_all_3/cell_{}.pickle'.format(lymph_series[0].idx_cell), "rb"))
            for lymph in lymph_series:
                lymph.ellipsoid = ellipsoids_dict[lymph.frame][0]
                lymph.ellipsoid_all_3 = ellipsoids_dict[lymph.frame]
        """


        for lymph_series in self.cells.values():
            ellipsoids_dict = {}
            for lymph in lymph_series:

                if lymph.coeffPathFormat is not None:

                    #lymph.surface_plot(plotter=plotter, opacity = 0.5) # plot cell surface
                    #lymph.plotRecon_singleDeg(plotter=plotter, max_l = 1, opacity = 0.5) # plot ellipsoid (i.e. l_max = 1)

                    x, y, z, = 0, 1, 2

                    """
                    neg_coeff_x = -lymph._get_clm(x, 1, 1).real + 1j*lymph._get_clm(x, 1, 1).imag
                    neg_coeff_y = -lymph._get_clm(y, 1, 1).real + 1j*lymph._get_clm(y, 1, 1).imag
                    neg_coeff_z = -lymph._get_clm(z, 1, 1).real + 1j*lymph._get_clm(z, 1, 1).imag


                    A = np.array([[neg_coeff_x-lymph._get_clm(x, 1, 1), 1j*(neg_coeff_x+lymph._get_clm(x, 1, 1)), np.sqrt(2)*lymph._get_clm(x, 1, 0)],
                                        [neg_coeff_y-lymph._get_clm(y, 1, 1), 1j*(neg_coeff_y+lymph._get_clm(y, 1, 1)), np.sqrt(2)*lymph._get_clm(y, 1, 0)],
                                    [neg_coeff_z-lymph._get_clm(z, 1, 1), 1j*(neg_coeff_z+lymph._get_clm(z, 1, 1)), np.sqrt(2)*lymph._get_clm(z, 1, 0)]])

                    A *= np.sqrt(3)/(2*np.sqrt(2*math.pi))
                    A = A.real # imaginary part is zero anyway

                    print('A1', A)
                    """


                    A = np.array([[lymph._get_clm(x, 1, 0), lymph._get_clm(y, 1, 0), lymph._get_clm(z, 1, 0)],
                                    [lymph._get_clm(x, 1, 1).real, lymph._get_clm(y, 1, 1).real, lymph._get_clm(z, 1, 1).real],
                                    [lymph._get_clm(x, 1, 1).imag, lymph._get_clm(y, 1, 1).imag, lymph._get_clm(z, 1, 1).imag]])
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
                    lengths = lengths[np.argsort(lengths)[::-1]]
                    vecs = vecs[:, np.argsort(lengths)[::-1]]

                    lymph.ellipsoid_lengths = lengths
                    lymph.ellipsoid_vecs = [vecs[:, 0], vecs[:, 0], vecs[:, 0]]

            dict = utils_general.get_frame_dict(lymph_series)
            frames = list(dict.keys())
            for frame in range(int(min(frames)), int(max(frames))):
                if frame in frames and frame-1 in frames and frame+1 in frames:
                    if dict[frame].ellipsoid_lengths is None and dict[frame-1].ellipsoid_lengths is not None and dict[frame+1].ellipsoid_lengths is not None:
                        dict[frame].ellipsoid_lengths = (dict[frame-1].ellipsoid_lengths + dict[frame+1].ellipsoid_lengths)/2
                        lymph.ellipsoid_vecs = [(dict[frame-1].ellipsoid_vecs[idx] + dict[frame+1].ellipsoid_vecs[idx])/2 for idx in range(3)]






        """
        for lymph_series in self.cells.values():
            dict = utils_general.get_frame_dict(lymph_series)
            frames = list(dict.keys())


            for lymph in lymph_series:

                frame = lymph.frame
                fs = [frame-i for i in reversed(range(1, int(time_either_side//lymph.t_res)+1))] + [frame] + [frame+i for i in range(1, int(time_either_side//lymph.t_res)+1)]
                spin_vecs = []
                angles = []
                for idx_f in range(len(fs)-1):

                    if fs[idx_f] in frames and fs[idx_f+1] in frames:

                        vec1 = dict[fs[idx_f]].ellipsoid_vecs[0]
                        vec2 = dict[fs[idx_f+1]].ellipsoid_vecs[0]

                        cross_norm = np.linalg.norm(np.cross(vec2, vec1))
                        angle = np.arcsin(cross_norm/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
                        angle /= dict[fs[idx_f]].t_res

                        dict[fs[idx_f]].spin_vec = angle*np.cross(vec2, vec1) /cross_norm
                        spin_vecs.append(dict[fs[idx_f]].spin_vec)
                        dict[fs[idx_f]].angle = np.linalg.norm(dict[fs[idx_f]].spin_vec)
                        angles.append(dict[fs[idx_f]].angle)

                if len(spin_vecs) >= len(fs)-3:
                    lymph.spin_vec_std = np.sum(np.var(np.array(spin_vecs), axis = 0)) # highlights changes in turning direction & magnitude
                    lymph.angle_mean = np.mean(angles) # highlights  turning magnitude
        """







    def _set_morph_derivs(self, time_either_side = 12):
        """
        Set lymph.morph_deriv attribute
        this is the mean derivative of RI_vector, showing how much the morphology is changing (ignoring e.g. rotations)
        """

        for lymph_series in self.cells.values():
            dict = utils_general.get_frame_dict(lymph_series)
            frames = list(dict.keys())
            for lymph in lymph_series:
                frame = lymph.frame
                fs = [frame-i for i in reversed(range(1, int(time_either_side//lymph.t_res)+1))] + [frame] + [frame+i for i in range(1, int(time_either_side//lymph.t_res)+1)]
                morphs = []
                for f in fs:
                    if f in frames:
                        morphs.append(dict[f].RI_vector[1:])
                morph_derivs = []
                morph_derivs_low = []
                morph_derivs_high = []
                for idx in range(1, len(morphs)):
                    morph_derivs.append(np.linalg.norm(morphs[idx]-morphs[idx-1]))
                    morph_derivs_low.append(np.linalg.norm(morphs[idx][:2]-morphs[idx-1][:2]))
                    morph_derivs_high.append(np.linalg.norm(morphs[idx][2:]-morphs[idx-1][2:]))
                if len(morph_derivs) == len(fs)-1: # since derivative chops of 1 element
                    lymph.morph_deriv = np.mean(np.array(morph_derivs), axis = 0)
                    lymph.morph_deriv /= lymph.t_res # DOES THIS ALSO NEED TO BE NORMALIZED BY VOLUME? No, because RI_vector already is
                    lymph.morph_deriv_low = np.mean(np.array(morph_derivs_low), axis = 0)
                    lymph.morph_deriv_low /= lymph.t_res
                    lymph.morph_deriv_high = np.mean(np.array(morph_derivs_high), axis = 0)
                    lymph.morph_deriv_high /= lymph.t_res
