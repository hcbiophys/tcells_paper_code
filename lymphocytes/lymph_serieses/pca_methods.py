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

        RI_vectors = []

        for lymph_series in self.lymph_serieses:
            for lymph in lymph_series:
                if lymph is not None:
                    RI_vectors.append(lymph.RI_vector)

        RI_vectors = np.array(RI_vectors)

        pca_obj = PCA(n_components = n_components)
        pca_obj.fit_transform(RI_vectors)
        print('EXPLAINED VARIANCE RATIO: ', pca_obj.explained_variance_ratio_)

        for lymph_series in self.lymph_serieses:
            for lymph in lymph_series:
                    lymph.pca = pca_obj.transform(np.array(lymph.RI_vector).reshape(1, -1))
                    lymph.pca = np.squeeze(lymph.pca, axis = 0)

        return pca_obj

    def PC_sampling(self, n_components):

        pca_obj = self._set_pca(n_components)

        PCs = np.array([lymph.pca for lymph_series in self.lymph_serieses for lymph in lymph_series])
        mins = np.min(PCs, axis = 0)
        maxs = np.max(PCs, axis = 0)
        mean = np.mean(PCs, axis = 0)

        fig = plt.figure()
        for idx_PC in range(PCs.shape[1]):
            for idx_sample, sample in enumerate([-2, -1, 0, 1, 2]):
                mean_copy = copy.deepcopy(mean)
                mean_copy[idx_PC] += sample*(maxs[idx_sample] - mins[idx_sample])/4
                inverted = pca_obj.inverse_transform(mean_copy)
                ax = fig.add_subplot(5, n_components, idx_PC*PCs.shape[1]+idx_sample+1)
                ax.bar(range(len(inverted)), inverted)
                if sample in [-2, 2]:
                    print([np.round(i, 4) for i in inverted])
            print('--')
