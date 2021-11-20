import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import pickle
import sys
from scipy.special import sph_harm
from sklearn.decomposition import PCA



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


    def _set_centroid_attributes(self, attribute, time_either_side = None, time_either_side_2 = 50, idx_cell = None):
        """
        Set delta_centroid or delta_sensing_direction
        - attribute: which to set
        - time_either_side: sets the running mean window size
        """

        if attribute not in self.attributes_set:

            self._set_mean_uropod_and_centroid(time_either_side = 12)
            if attribute == 'delta_centroid_uropod':
                self._set_delta_centroid_and_uropod()
            elif attribute == 'delta_sensing_direction':
                self._set_delta_sensing_directions()
            elif attribute == 'run':
                self._set_run(time_either_side, set_as_mean = False, idx_cell = idx_cell)
            elif attribute == 'run_mean':
                self._set_run(time_either_side, set_as_mean = True)
            elif attribute == 'searching':
                 self._set_searching(time_either_side, time_either_side_2)

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




    def _set_delta_centroid_and_uropod(self):
        """
        Set lymph.delta_centroid based on lymph.mean_centroid
        """

        for attributes_pair in [('mean_centroid', 'delta_centroid'), ('mean_uropod', 'delta_uropod')]:

            for lymph_series in self.cells.values():
                dict = self._get_frame_dict(lymph_series)
                frames = list(dict.keys())

                # set delta_centroid
                for lymph in lymph_series:
                    frame = lymph.frame
                    if getattr(lymph, attributes_pair[0]) is not None:
                        if frame-1 in frames and getattr(dict[frame-1], attributes_pair[0]) is not None:
                            val = np.linalg.norm(getattr(lymph, attributes_pair[0])-getattr(dict[frame-1], attributes_pair[0]))
                            val /= np.cbrt(lymph.volume)
                            val /= lymph.t_res
                            setattr(lymph, attributes_pair[1], val)
                        elif frame+1 in frames and getattr(dict[frame+1], attributes_pair[0])  is not None:
                            val = np.linalg.norm(getattr(dict[frame+1], attributes_pair[0])-getattr(lymph, attributes_pair[0]) )
                            val /= np.cbrt(lymph.volume)
                            val /= lymph.t_res
                            setattr(lymph, attributes_pair[1], val)
                        print(lymph.delta_centroid, lymph.delta_uropod)






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


    def _set_searching(self, time_either_side, time_either_side_2):

        for lymph_series in self.cells.values():

            #ellipsoid_centroids_dict = {}
            #for lymph in lymph_series:
            #    print(lymph.idx_cell)
            #    vertices, faces, uropod = lymph._get_vertices_faces_plotRecon_singleDeg(max_l = 1, uropod_align = False)
            #    surf = pv.PolyData(vertices, faces)
            #    ellipsoid_centroids_dict[lymph.frame] = surf.center_of_mass()
            #pickle_out = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/ellipsoid_centroids/cell_{}.pickle'.format(lymph_series[0].idx_cell),'wb')
            #pickle.dump(ellipsoid_centroids_dict, pickle_out)

            #ellipsoid_centroids_dict =   pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/ellipsoid_centroids/cell_{}.pickle'.format(lymph_series[0].idx_cell), "rb"))
            #for lymph in lymph_series:
            #    lymph.centroid_chosen = ellipsoid_centroids_dict[lymph.frame]

            """
            furthest_points_dict = {}
            for lymph in lymph_series:
                dists = [np.linalg.norm(lymph.uropod-lymph.vertices[i, :]) for i in range(lymph.vertices.shape[0])]
                idx_furthest = dists.index(max(dists))
                furthest_point = lymph.vertices[idx_furthest, :]
                furthest_points_dict[lymph.frame] = furthest_point
                lymph.centroid_chosen = furthest_point
            pickle_out = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/furthest_points/cell_{}.pickle'.format(lymph_series[0].idx_cell),'wb')
            pickle.dump(furthest_points_dict, pickle_out)
            """

            furthest_points_dict =   pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/furthest_points/cell_{}.pickle'.format(lymph_series[0].idx_cell), "rb"))
            for lymph in lymph_series:
                #lymph.centroid_chosen = furthest_points_dict[lymph.frame]
                lymph.centroid_chosen = lymph.mean_centroid


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

                    if len(spin_vecs) >= len(fs)-3:
                        lymph.spin_vec_std = np.sum(np.var(np.array(spin_vecs), axis = 0)) # highlights changes in turning direction & magnitude
                        lymph.spin_vec_magnitude_mean = np.mean(spin_vec_magnitudes) # highlights  turning magnitude
                        lymph.direction_std = np.sum(np.var(np.array(directions), axis = 0)) # highlights direction change (e.g. turning could be oscilatory)
                        lymph.direction_mean = np.mean(np.array(directions), axis = 0)
                        lymph.direction_mean /= np.linalg.norm(lymph.direction_mean)



        for lymph_series in self.cells.values():
            dict = self._get_frame_dict(lymph_series)
            frames = list(dict.keys())

            # set delta_sensing_direction
            for lymph in lymph_series:

                frame = lymph.frame
                fs = [frame-i for i in reversed(range(1, int(time_either_side_2//lymph.t_res)+1))] + [frame] + [frame+i for i in range(1, int(time_either_side_2//lymph.t_res)+1)]
                spin_vecs_2 = []
                directions_2 = []
                spin_vec_magnitudes_2 = []
                for idx_f in range(len(fs)-1):

                    if fs[idx_f] in frames and fs[idx_f+1] in frames:

                        if dict[fs[idx_f]].direction_mean is not None and dict[fs[idx_f+1]].direction_mean is not None:

                            vec1 = dict[fs[idx_f]].direction_mean
                            directions_2.append(dict[fs[idx_f]].direction_mean)
                            vec2 = dict[fs[idx_f+1]].direction_mean


                            cross_norm = np.linalg.norm(np.cross(vec2, vec1))
                            angle = np.arcsin(cross_norm/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
                            angle /= dict[fs[idx_f]].t_res
                            dict[fs[idx_f]].spin_vec_2 = angle*np.cross(vec2, vec1) /cross_norm

                            dict[fs[idx_f]].spin_vec_2 /= np.linalg.norm(dict[fs[idx_f]].spin_vec_2)


                            spin_vecs_2.append(dict[fs[idx_f]].spin_vec_2)
                            dict[fs[idx_f]].spin_vec_magnitude_2 = np.linalg.norm(dict[fs[idx_f]].spin_vec_2)
                            spin_vec_magnitudes_2.append(dict[fs[idx_f]].spin_vec_magnitude_2)

                if len(spin_vecs_2) >= len(fs)-3:
                    lymph.spin_vec_std_2 = np.sum(np.var(np.array(spin_vecs_2), axis = 0)) # highlights changes in turning direction & magnitude
                    lymph.spin_vec_magnitude_mean_2 = np.mean(spin_vec_magnitudes_2) # highlights  turning magnitude
                    lymph.direction_std_2 = np.sum(np.var(np.array(directions_2), axis = 0)) # highlights direction change (e.g. turning could be oscilatory)





    """
    def _set_searching(self, time_either_side):

        for lymph_series in self.cells.values():
            ellipsoids_dict = {}
            for lymph in lymph_series:
                vertices, faces, uropod = lymph._get_vertices_faces_plotRecon_singleDeg(max_l = 1, uropod_align = False)
                vertices_split = [vertices[row, :] for row in range(vertices.shape[0])]
                pca_obj = PCA(n_components = 1)
                pca_obj.fit_transform(vertices - np.mean(vertices, axis = 0))
                pc_vals = [pca_obj.transform(i.reshape(1, -1)) for i in vertices_split]
                idx_min = pc_vals.index(min(pc_vals))
                idx_max = pc_vals.index(max(pc_vals))
                min_vert = vertices[idx_min, :]
                max_vert = vertices[idx_max, :]
                if np.linalg.norm(lymph.uropod - min_vert) < np.linalg.norm(lymph.uropod - max_vert):
                    a = min_vert
                    b = max_vert
                else:
                    a = max_vert
                    b = min_vert
                ellipsoids_dict[lymph.frame] = [a,b]
            pickle_out = open('/Users/harry/OneDrive - Imperial College London/lymphocytes/ellipsoids/cell_{}.pickle'.format(lymph_series[0].idx_cell),'wb')
            pickle.dump(ellipsoids_dict, pickle_out)


        for lymph_series in self.cells.values():
            ellipsoids_dict = pickle.load(open('/Users/harry/OneDrive - Imperial College London/lymphocytes/ellipsoids/cell_{}.pickle'.format(lymph_series[0].idx_cell), "rb"))
            for lymph in lymph_series:
                lymph.ellipsoid_ab = ellipsoids_dict[lymph.frame]

        for lymph_series in self.cells.values():
            dict = self._get_frame_dict(lymph_series)
            frames = list(dict.keys())

            for lymph in lymph_series:


                frame = lymph.frame
                fs = [frame-i for i in reversed(range(1, int(time_either_side//lymph.t_res)+1))] + [frame] + [frame+i for i in range(1, int(time_either_side//lymph.t_res)+1)]
                spin_vecs = []
                directions = []
                spin_vec_magnitudes = []
                for idx_f in range(len(fs)-1):

                    if fs[idx_f] in frames and fs[idx_f+1] in frames:


                        vec1 = dict[fs[idx_f]].ellipsoid_ab

                        dict[fs[idx_f]].direction = vec1/np.linalg.norm(vec1)
                        directions.append(dict[fs[idx_f]].direction)

                        vec2 = dict[fs[idx_f+1]].ellipsoid_ab
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
                    lymph.direction_mean = np.mean(np.array(directions), axis = 0)
    """









    def _set_run(self, time_either_side, idx_cell = None, set_as_mean = False):


        for lymph_series in self.cells.values():
            if lymph_series[0].idx_cell == idx_cell or idx_cell is None:
                dict = self._get_frame_dict(lymph_series)
                frames = list(dict.keys())

                for lymph in lymph_series:

                    if lymph.mean_centroid is not None and lymph.mean_uropod is not None:
                        frame = lymph.frame

                        fs = [frame-i for i in reversed(range(1, int(time_either_side//lymph.t_res)+1))] + [frame] + [frame+i for i in range(1, int(time_either_side//lymph.t_res)+1)]
                        runs = []
                        runs_centroids = []

                        for idx_f in range(len(fs)-1):

                            for (attribute,which_list) in [('mean_uropod', runs), ('mean_centroid', runs_centroids)]:

                                if fs[idx_f] in frames and fs[idx_f+1] in frames:
                                    if dict[fs[idx_f]].mean_uropod is not None and dict[fs[idx_f+1]].mean_uropod is not None:

                                        vec1 = getattr(dict[fs[idx_f+1]], attribute)-getattr(dict[fs[idx_f]], attribute)
                                        vec2 = dict[fs[idx_f]].mean_centroid - dict[fs[idx_f]].mean_uropod

                                        cos_angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

                                        dict[fs[idx_f]].run_theta = np.arccos(cos_angle)


                                        run = np.linalg.norm(vec1*cos_angle)
                                        run /= np.cbrt(dict[fs[idx_f]].volume)
                                        run /= dict[fs[idx_f]].t_res
                                        run *= np.sign(cos_angle)
                                        which_list.append(run)

                        if len(runs) == len(fs)-1:
                            if set_as_mean:
                                lymph.run_mean = np.mean(runs)
                            else:
                                lymph.run = np.mean(runs)

                        if len(runs_centroids) == len(fs)-1:
                            lymph.run_centroid = np.mean(runs_centroids)




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
                    lymph.morph_deriv /= lymph.t_res # DOES THIS ALSO NEED TO BE NORMALIZED BY VOLUME?
                    lymph.morph_deriv_low = np.mean(np.array(morph_derivs_low), axis = 0)
                    lymph.morph_deriv_low /= lymph.t_res
                    lymph.morph_deriv_high = np.mean(np.array(morph_derivs_high), axis = 0)
                    lymph.morph_deriv_high /= lymph.t_res
