import numpy as np
import matplotlib.pyplot as plt
import lymphocytes.utils.general as utils_general
from sklearn.decomposition import PCA
import sys
from sklearn.preprocessing import StandardScaler
import copy
import pyvista as pv

import lymphocytes.utils.general as utils_general



class PCA_Methods:
    """
    Inherited by Lymph_Seriese class.
    Contains methods that involve PCA.
    """


    def _edit_serieses(self, removeLymphNones = False, removeSpeedNone = False, removeAngleNone = False):

        if removeLymphNones:
            self.lymph_serieses = utils_general.del_whereNone(self.lymph_serieses, 'lymph')
        if removeSpeedNone:
            self.lymph_serieses = utils_general.del_whereNone(self.lymph_serieses, 'speed')
        if removeAngleNone:
            self.lymph_serieses = utils_general.del_whereNone(self.lymph_serieses, 'angle')


    def _set_pca(self, n_components, removeSpeedNone = False, removeAngleNone = False):
        """
        Args:
        - n_components: dimensionailty of low dimensional representation.
        - removeSpeedNone: whether to remove frames with speed None.
        - removeAngleNone: whether to remove frames with angle None.
        """

        self._edit_serieses(removeLymphNones = True, removeSpeedNone = removeSpeedNone, removeAngleNone = removeAngleNone)
        lymphs = utils_general.list_all_lymphs(self)
        RI_vectors = np.array([lymph.RI_vector for lymph in lymphs])

        pca_obj = PCA(n_components = n_components)
        pca_obj.fit_transform(RI_vectors)
        print('EXPLAINED VARIANCE RATIO: ', pca_obj.explained_variance_ratio_)

        for lymph in lymphs:
            lymph.pca = pca_obj.transform(lymph.RI_vector.reshape(1, -1))
            lymph.pca = np.squeeze(lymph.pca, axis = 0)

        return pca_obj

    def PC_sampling(self, n_components):


        pca_obj = self._set_pca(n_components)
        lymphs = utils_general.list_all_lymphs(self)
        PCs = np.array([lymph.pca for lymph in lymphs])
        mins = np.min(PCs, axis = 0)
        maxs = np.max(PCs, axis = 0)
        mean = np.mean(PCs, axis = 0)
        print('PC mean', mean)

        fig_sampling = plt.figure()

        for idx_PC in range(PCs.shape[1]):
            for idx_sample, sample in enumerate([-2, -1, 0, 1, 2]):


                mean_copy = copy.deepcopy(mean)
                mean_copy[idx_PC] += sample*(maxs[idx_PC] - mins[idx_PC])/4
                inverted = pca_obj.inverse_transform(mean_copy)



                ax = fig_sampling.add_subplot(n_components, 5, 5*idx_PC+idx_sample+1)
                ax.bar(range(len(inverted)), inverted)
                ax.set_ylim([-1, 3.5])






    """
    def PC_sampling(self, n_components):

        plotter = pv.Plotter(shape=(3, 8))

        pca_obj = self._set_pca(n_components)
        lymphs = utils_general.list_all_lymphs(self)
        PCs = np.array([lymph.pca for lymph in lymphs])
        mins = np.min(PCs, axis = 0)
        maxs = np.max(PCs, axis = 0)
        mean = np.mean(PCs, axis = 0)
        print('PC mean', mean)

        fig_sampling = plt.figure()

        lymphs = utils_general.list_all_lymphs(self)
        for idx_PC in range(PCs.shape[1]):
            for idx_sample, sample in enumerate([-2, -1, 0, 1, 2]):


                mean_copy = copy.deepcopy(mean)
                mean_copy[idx_PC] += sample*(maxs[idx_PC] - mins[idx_PC])/4
                inverted = pca_obj.inverse_transform(mean_copy)


                dists = [np.linalg.norm(lymph.RI_vector - inverted) for lymph in lymphs]
                closest = lymphs[dists.index(min(dists))]
                lymphs.remove(closest)

                #plotter = pv.Plotter()
                #closest.plotRecon_singleDeg(plotter=plotter, max_l = 3, uropod_align = False)
                #plotter.show()

                diffs = inverted - closest.RI_vector

                closest.max_l = 3
                closest._set_vector()
                coeffs_per_coord = int(closest.vector.shape[0]/3.)

                # make vector uropod-based
                closest.vector[0*coeffs_per_coord] = (closest.centroid - closest.uropod)[0]
                closest.vector[1*coeffs_per_coord] = (closest.centroid - closest.uropod)[1]
                closest.vector[2*coeffs_per_coord] = (closest.centroid - closest.uropod)[2]

                # get l-based factors for noise scales
                dict = utils_general.get_idxs_l_in_vector(closest.vector, closest.max_l)
                to_multiply = []
                for _ in range(3): # for each coordinate
                    for l, l_list in dict.items():
                        for part in l_list:
                            to_multiply.append(np.cbrt(closest.volume)*diffs[l]/len(l_list))



                vectors, dists = [], []

                initial_vector = closest.vector
                for _ in range(50):
                    noise = abs(np.random.rand(closest.vector.shape[0]))
                    noise = noise*np.array(to_multiply)*2
                    closest.vector = initial_vector + noise

                    closest.coeff_array = np.concatenate([i[:, None] for i in np.split(closest.vector, 3)], axis = 1)
                    closest._set_RIvector()
                    closest.RI_vector[0] = np.linalg.norm(closest.coeff_array[0, :])/np.cbrt(closest.volume)
                    dist = np.linalg.norm(closest.RI_vector - inverted[:4])

                    vectors.append(closest.vector)
                    dists.append(dist)

                closest.vector = vectors[dists.index(min(dists))]
                plotter.subplot(idx_PC, idx_sample)
                closest.plotRecon_singleDeg(plotter=plotter, max_l = 3, uropod_align = False)





                ax = fig_sampling.add_subplot(n_components, 5, 5*idx_PC+idx_sample+1)
                ax.bar(range(len(inverted)), inverted)
                ax.set_ylim([-1, 3.5])

        plotter.show()
        """
