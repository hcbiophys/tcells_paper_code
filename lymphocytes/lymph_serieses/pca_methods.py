import numpy as np
import matplotlib.pyplot as plt
import lymphocytes.utils.general as utils_general
from sklearn.decomposition import PCA
import sys
from sklearn.preprocessing import StandardScaler
import copy

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

        fig = plt.figure()
        for idx_PC in range(PCs.shape[1]):
            for idx_sample, sample in enumerate([-2, -1, 0, 1, 2]):
                mean_copy = copy.deepcopy(mean)
                mean_copy[idx_PC] += sample*(maxs[idx_PC] - mins[idx_PC])/4
                inverted = pca_obj.inverse_transform(mean_copy)
                ax = fig.add_subplot(n_components, 5, 5*idx_PC+idx_sample+1)
                ax.bar(range(len(inverted)), inverted)
                ax.set_ylim([-1, 3.5])
