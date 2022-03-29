import sys
import numpy as np
from tcells_paper_code.morphodynamics.cwt import CWT, show_cell_series_clustered


filename = sys.argv[1]
load_or_save_or_run = sys.argv[2]



cwt = CWT(filename = filename, load_or_save_or_run = load_or_save_or_run, idx_cfs= 'all')

#cwt.plot_series()
#cwt.plot_wavelets(wavelet = 'mexh', scales = [0.5*i for i in np.linspace(2, 12, 6)])
#cwt.plot_wavelets(wavelet = 'gaus1', scales = [0.4*i for i in np.linspace(2, 27, 6)])
#cwt.plot_spectrograms()
#cwt.plot_embeddings(path_of = 'CELL21a')
#cwt.set_kde(plot = True)

#cwt.transition_matrix(grid = True, plot = True)

#show_cell_series_clustered(['CELL21a-72'])
#sys.exit()
