import numpy as np
import matplotlib.pyplot as plt
import lymphocytes.utils.general as utils_general
from sklearn.decomposition import PCA
import sys
from sklearn.preprocessing import StandardScaler

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
                if lymph is not None:
                    lymph.pca = pca_obj.transform(np.array(lymph.RI_vector).reshape(1, -1))
                    lymph.pca = np.squeeze(lymph.pca, axis = 0)
