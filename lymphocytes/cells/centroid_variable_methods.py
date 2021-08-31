import numpy as np
import matplotlib.pyplot as plt
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


    def _set_centroid_attributes(self, attribute, num_either_side = 2):
        """
        Set delta_centroid or delta_sensing_direction
        - attribute: which to set
        - num_either_side: sets the running mean window size
        """
        self._set_mean_uropod_and_centroid(num_either_side)
        if attribute == 'delta_centroid':
            self._set_delta_centroid()
        elif attribute == 'delta_sensing_direction':
            self._set_delta_sensing_directions()


    def _set_mean_uropod_and_centroid(self, num_either_side):
        """
        If there are enough surrounding frames, set lymph.mean_uropod & lymph.mean_centroid
        """
        for lymph_series in self.cells.values():
            dict = self._get_frame_dict(lymph_series)
            frames = list(dict.keys())
            for lymph in lymph_series:
                frame = lymph.frame
                fs = [frame-i for i in range(1, num_either_side+1)] + [frame] + [frame+i for i in range(1, num_either_side+1)]
                uropods = []
                centroids = []
                for f in fs:
                    if f in frames:
                        uropods.append(dict[f].uropod)
                        centroids.append(dict[f].centroid)
                if len(uropods) == 2*num_either_side + 1:
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
                        lymph.delta_centroid /= lymph.t_res
                    elif frame+1 in frames and dict[frame+1].mean_centroid is not None:
                        lymph.delta_centroid = np.linalg.norm(dict[frame+1].mean_centroid-lymph.mean_centroid)
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



    def _set_morph_derivs(self, num_either_side = 2):
        """
        Set lymph.morph_deriv attribute
        this is the mean derivative of RI_vector, showing how much the morphology is changing (ignoring e.g. rotations)
        """

        for lymph_series in self.cells.values():
            dict = self._get_frame_dict(lymph_series)
            frames = list(dict.keys())
            for lymph in lymph_series:
                frame = lymph.frame
                fs = [frame-i for i in range(1, num_either_side+1)] + [frame] + [frame+i for i in range(1, num_either_side+1)]
                morphs = []
                for f in fs:
                    if f in frames:
                        morphs.append(dict[f].RI_vector)
                morph_derivs = []
                for idx in range(1, len(morphs)):
                    morph_derivs.append(np.linalg.norm(morphs[idx]-morphs[idx-1]))
                if len(morph_derivs) == 2*num_either_side: # since derivative chops of 1 element
                    lymph.morph_deriv = np.mean(np.array(morph_derivs), axis = 0)
                    lymph.morph_deriv /= lymph.t_res
