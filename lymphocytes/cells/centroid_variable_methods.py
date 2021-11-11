import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import pickle
import sys

import lymphocytes.utils.plotting as utils_plotting
import lymphocytes.utils.general as utils_general

class Centroid_Variable_Methods:


    def _get_frame_dict(self, lymph_series):
        """
        self.cells[idx_cell] is a list of lymphs, ordered by frame
        this function returns a dict so lymphs can easily be acccessed by frame, like dict[frame] = lymphs
        """
        frames = [lypmh.frame for lypmh in lymph_series]
        dict = {}
        for frame, lymph in zip(frames, lymph_series):
            dict[frame] = lymph
        return dict


    def _set_centroid_attributes(self, attribute, time_either_side = None, idx_cell = None):
        """
        Set delta_centroid or delta_sensing_direction
        - attribute: which to set
        - time_either_side: sets the running mean window size
        """

        if attribute not in self.attributes_set:

            self._set_mean_uropod_and_centroid(time_either_side = 12)
            if attribute == 'delta_centroid':
                self._set_delta_centroid()
            elif attribute == 'delta_sensing_direction':
                self._set_delta_sensing_directions()
            elif attribute == 'run':
                self._set_run(time_either_side, set_as_mean = False, idx_cell = idx_cell)
            elif attribute == 'run_mean':
                self._set_run(time_either_side, set_as_mean = True)
            elif attribute == 'searching':
                 self._set_searching(time_either_side)

            self.attributes_set.append(attribute)

    def _set_centroid_attributes_to_NONE(self):
        for lymph in utils_general.list_all_lymphs(self):
            lymph.run = None
            lymph.run_mean = None
            lymph.spin_vec = None
            lymph.spin_vec_magnitude = None
            lymph.spin_vec_magnitude_mean = None
            lymph.direction = None
            lymph.spin_vec_std = None
            lymph.direction_std = None


    def _set_mean_uropod_and_centroid(self, time_either_side):
        """
        If there are enough surrounding frames, set lymph.mean_uropod & lymph.mean_centroid
        """
        for lymph_series in self.cells.values():
            dict = self._get_frame_dict(lymph_series)
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



    def _set_delta_centroid(self):
        """
        Set lymph.delta_centroid based on lymph.mean_centroid
        """

        for lymph_series in self.cells.values():
            dict = self._get_frame_dict(lymph_series)
            frames = list(dict.keys())

            # set delta_centroid
            for lymph in lymph_series:
                frame = lymph.frame
                if lymph.mean_centroid is not None:
                    if frame-1 in frames and dict[frame-1].mean_centroid is not None:
                        lymph.delta_centroid = np.linalg.norm(lymph.mean_centroid-dict[frame-1].mean_centroid)
                        lymph.delta_centroid /= np.cbrt(lymph.volume)
                        lymph.delta_centroid /= lymph.t_res
                    elif frame+1 in frames and dict[frame+1].mean_centroid is not None:
                        lymph.delta_centroid = np.linalg.norm(dict[frame+1].mean_centroid-lymph.mean_centroid)
                        lymph.delta_centroid /= np.cbrt(lymph.volume)
                        lymph.delta_centroid /= lymph.t_res



    def _set_delta_sensing_directions(self):
        """
        Set lymph.delta_sensing_direction
        """

        for lymph_series in self.cells.values():
            dict = self._get_frame_dict(lymph_series)
            frames = list(dict.keys())

            # set delta_sensing_direction
            for lymph in lymph_series:
                if lymph.mean_centroid is not None and lymph.mean_uropod is not None:
                    frame = lymph.frame
                    vec1 = lymph.mean_centroid - lymph.mean_uropod
                    if frame-1 in frames and dict[frame-1].mean_centroid is not None and dict[frame-1].mean_uropod is not None:
                        vec2 = dict[frame-1].mean_centroid - dict[frame-1].mean_uropod
                        angle = np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
                        lymph.delta_sensing_direction = angle
                        lymph.delta_sensing_direction /= lymph.t_res
                    elif frame+1 in frames and dict[frame+1].mean_centroid is not None and dict[frame+1].mean_uropod is not None:
                        vec2 = dict[frame+1].mean_centroid - dict[frame+1].mean_uropod
                        angle = np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
                        lymph.delta_sensing_direction = angle
                        lymph.delta_sensing_direction /= lymph.t_res


    def _set_searching(self, time_either_side):
        """
        Set lymph.delta_sensing_direction
        """


        for lymph_series in self.cells.values():
            """
            ellipsoid_centroids_dict = {}
            for lymph in lymph_series:
                print(lymph.idx_cell)
                vertices, faces, uropod = lymph._get_vertices_faces_plotRecon_singleDeg(max_l = 1, uropod_align = False)
                surf = pv.PolyData(vertices, faces)
                ellipsoid_centroids_dict[lymph.frame] = surf.center_of_mass()
            pickle_out = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/ellipsoid_centroids/cell_{}.pickle'.format(lymph_series[0].idx_cell),'wb')
            pickle.dump(ellipsoid_centroids_dict, pickle_out)
            """

            ellipsoid_centroids_dict =   pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/ellipsoid_centroids/cell_{}.pickle'.format(lymph_series[0].idx_cell), "rb"))
            for lymph in lymph_series:
                lymph.centroid_chosen = ellipsoid_centroids_dict[lymph.frame]





        for lymph_series in self.cells.values():
            dict = self._get_frame_dict(lymph_series)
            frames = list(dict.keys())

            # set delta_sensing_direction
            for lymph in lymph_series:

                if lymph.centroid_chosen is not None and lymph.mean_uropod is not None:
                    frame = lymph.frame
                    fs = [frame-i for i in reversed(range(1, int(time_either_side//lymph.t_res)+1))] + [frame] + [frame+i for i in range(1, int(time_either_side//lymph.t_res)+1)]
                    spin_vecs = []
                    directions = []
                    spin_vec_magnitudes = []
                    for idx_f in range(len(fs)-1):

                        if fs[idx_f] in frames and fs[idx_f+1] in frames:
                            if dict[fs[idx_f]].mean_uropod is not None and dict[fs[idx_f+1]].mean_uropod is not None:

                                vec1 = dict[fs[idx_f]].centroid_chosen - dict[fs[idx_f]].mean_uropod

                                dict[fs[idx_f]].direction = vec1/np.linalg.norm(vec1)
                                directions.append(dict[fs[idx_f]].direction)


                                vec2 = dict[fs[idx_f+1]].centroid_chosen - dict[fs[idx_f+1]].mean_uropod
                                cross_norm = np.linalg.norm(np.cross(vec2, vec1))
                                angle = np.arcsin(cross_norm/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
                                angle /= dict[fs[idx_f]].t_res
                                dict[fs[idx_f]].spin_vec = angle*np.cross(vec2, vec1) /cross_norm
                                spin_vecs.append(dict[fs[idx_f]].spin_vec)
                                dict[fs[idx_f]].spin_vec_magnitude = np.linalg.norm(dict[fs[idx_f]].spin_vec)
                                spin_vec_magnitudes.append(dict[fs[idx_f]].spin_vec_magnitude)

                    if len(spin_vecs) == len(fs)-1:
                        lymph.spin_vec_std = np.sum(np.std(np.array(spin_vecs), axis = 0)) # highlights changes in turning direction & magnitude
                        lymph.spin_vec_magnitude_mean = np.mean(spin_vec_magnitudes) # highlights  turning magnitude
                        lymph.direction_std = np.sum(np.std(np.array(directions), axis = 0)) # highlights direction change (e.g. turning could be oscilatory)











    def _set_run(self, time_either_side, idx_cell = None, set_as_mean = False):

        for lymph in utils_general.list_all_lymphs(self):
            if lymph.idx_cell == idx_cell or idx_cell is None:
                lymph.centroid_chosen = lymph.mean_centroid
                #vertices, faces, uropod = lymph._get_vertices_faces_plotRecon_singleDeg(max_l = 1, uropod_align = False)
                #surf = pv.PolyData(vertices, faces)
                #lymph.centroid_chosen = surf.center_of_mass()


        for lymph_series in self.cells.values():
            print(lymph_series[0].idx_cell, idx_cell)
            if lymph_series[0].idx_cell == idx_cell or idx_cell is None:
                dict = self._get_frame_dict(lymph_series)
                frames = list(dict.keys())

                """
                for lymph in lymph_series:
                    if lymph.mean_uropod is not None:
                        frame = lymph.frame
                        if frame-1 in frames and dict[frame-1].mean_centroid is not None:
                            uropod_vec = lymph.mean_uropod-dict[frame-1].mean_uropod
                        #elif frame+1 in frames and dict[frame+1].mean_uropod is not None:
                        #    uropod_vec = dict[frame+1].mean_uropod-lymph.mean_uropod
                            vec2 = lymph.mean_centroid - lymph.mean_uropod
                            cos_angle = np.dot(uropod_vec, vec2)/(np.linalg.norm(uropod_vec)*np.linalg.norm(vec2))

                            run = np.linalg.norm(uropod_vec*cos_angle)
                            run /= np.cbrt(lymph.volume)
                            run /= lymph.t_res
                            run *= np.sign(cos_angle)

                            lymph.run = run
                            lymph.run_initial = run


                """

                for lymph in lymph_series:

                    if lymph.centroid_chosen is not None and lymph.mean_uropod is not None:
                        frame = lymph.frame

                        fs = [frame-i for i in reversed(range(1, int(time_either_side//lymph.t_res)+1))] + [frame] + [frame+i for i in range(1, int(time_either_side//lymph.t_res)+1)]
                        runs = []

                        for idx_f in range(len(fs)-1):

                            if fs[idx_f] in frames and fs[idx_f+1] in frames:
                                if dict[fs[idx_f]].mean_uropod is not None and dict[fs[idx_f+1]].mean_uropod is not None:

                                    uropod_vec = dict[fs[idx_f+1]].mean_uropod-dict[fs[idx_f]].mean_uropod
                                    vec2 = dict[fs[idx_f]].centroid_chosen - dict[fs[idx_f]].mean_uropod

                                    cos_angle = np.dot(uropod_vec, vec2)/(np.linalg.norm(uropod_vec)*np.linalg.norm(vec2))

                                    run = np.linalg.norm(uropod_vec*cos_angle)
                                    run /= np.cbrt(dict[fs[idx_f]].volume)
                                    run /= dict[fs[idx_f]].t_res
                                    run *= np.sign(cos_angle)
                                    runs.append(run)

                        if len(runs) == len(fs)-1:

                            if set_as_mean:
                                lymph.run_mean = np.mean(runs)
                            else:
                                lymph.run = np.mean(runs)








    def _set_morph_derivs(self, time_either_side = 12):
        """
        Set lymph.morph_deriv attribute
        this is the mean derivative of RI_vector, showing how much the morphology is changing (ignoring e.g. rotations)
        """

        for lymph_series in self.cells.values():
            dict = self._get_frame_dict(lymph_series)
            frames = list(dict.keys())
            for lymph in lymph_series:
                frame = lymph.frame
                fs = [frame-i for i in reversed(range(1, int(time_either_side//lymph.t_res)+1))] + [frame] + [frame+i for i in range(1, int(time_either_side//lymph.t_res)+1)]
                morphs = []
                for f in fs:
                    if f in frames:
                        morphs.append(dict[f].RI_vector)
                morph_derivs = []
                for idx in range(1, len(morphs)):
                    morph_derivs.append(np.linalg.norm(morphs[idx]-morphs[idx-1]))
                if len(morph_derivs) == len(fs)-1: # since derivative chops of 1 element
                    lymph.morph_deriv = np.mean(np.array(morph_derivs), axis = 0)
                    lymph.morph_deriv /= lymph.t_res
