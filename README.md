# Code for the paper: T Cell Morphodynamics Reveal Periodic Shape Oscillations in 3D Migration


## Setup

* Tested on Mac OS Catalina 10.15.3

* Requirements (can be installed via e.g. python -m pip install numpy==1.18.4 once the desired python environment has been loaded):\
Python 3.7.3\
numpy==1.18.4\
opencv-python==4.1.2.30\
scikit-learn==0.23.0\
matplotlib==3.2.1\
PyWavelets==1.1.1\
pyvista==0.32.1\
scipy==1.7.2\


* To set up as a package so imports of internal modules work:\
*python3 setup.py develop*

## Section 1: Morphology analysis (shape descriptor, PCA etc.)

In all cases, run the following command while in the *scripts* folder within *tcells_paper_code*, and uncomment the required function (details below):\
*python3 main_morphology.py all_run_stop*\
Note: *all_run_stop* can be *all*, *run*, or *stop* and these correspond to including all cells, the long videos from the run mode, and the long videos from the stop mode, respectively.


### Section 1a: Single-cell functions
Running functions with a single cell video (for example, plotting its time series). Note: for these functions, change *idx_cell = None* in *main_morphology.py* from *None* to the required cell code (the possible codes are shown in *main_morphology.py*).

* (Fig. 1d) To plot how increasing the number of spherical harmonics (by including more degrees, l) decreases the smoothing, uncomment the function:\
*cells.plot_l_truncations(idx_cell=idx_cell)*

* To plot the subsampled frames in a video, uncomment the function:\
*cells.plot_orig_series(idx_cell = idx_cell, uropod_align = False, color_by = 'pca1', plot_every = 6, plot_flat = False)*\
Here, cells will be coloured by PC (principal component) 2. Note: pca0, pca1, pca2 correspond to PCs 1, 2, and 3, respectively. Every 6 frames are plotted.

* To plot the same as above, but for reconstructions with truncated spherical harmonic representations, uncomment the function:\
*cells.plot_recon_series(idx_cell = idx_cell, max_l = 1, color_by = None, plot_every=6)*\
Here, the maximum spherical harmonic degree, l, is 1, there is no special colouring, and ever 6 frames are plotted.

* To plot the translucent surface segmentation of each frame in space as the cell migrates (Fig. 3c), uncomment the function:\
*cells.plot_migratingCell(idx_cell=idx_cell, opacity = 0.2, plot_every = 40)*\
Here, opacity determines the translucency, and every 40 frames are plotted.

* To plot the line joining the uropod label and centroid over the migration visualisation of the previous bullet point, uncomment the function:\
*cells.plot_uropod_centroid_line(idx_cell = idx_cell, plot_every = 1)*

* To plot the PC series of a single cell, uncomment the function:\
*cells.plot_series_PCs(idx_cell=idx_cell, plot_every=5)*



### Section 1b: Many-cell functions
Running functions using many cell videos (for example, plotting attributes across all cells).

* To plot the histogram comparing uropod-centroid (UC) axis and ellipsoid axis (Supplementary Fig. 4b), uncomment the function:\
*cells.alignments(min_length = 0.0025)*

* (Supplementary Fig. 4d) To plot the histograms of speed_uropod and speed_centroid, uncomment the function:\
*cells.speeds_histogram()*

* (Fig. 3b) To plot the cumulative retraction speeds (i.e. speed_uropod), uncomment the function:
*cells.plot_cumulatives()*

* (Fig. 3d) To plot the figure showing the emergence of bimodal run-and-stop behaviour, uncomment the function:\
*cells.bimodality_emergence()*

* (Fig. 2b, Supplementary Fig. 2a) To plot cell surfaces sampled over the PCs, uncomment the function:\
*cells.plot_component_frames(bin_size=7, pca=True, plot_original = False, max_l = 3)*\
Here, *plot_original = True / False* is for Supplementary Fig. 2a (original surfaces) and Fig. 2b (using truncated, i.e. smoothed, representations), respectively. The bin size gives the number of frames shown, and max_l gives the degree of truncation for the spherical harmonic descriptor.

* (Fig. 2c) To sample the min, mean and max along each PC in the spherical harmonic descriptor space, uncomment the function:\
*cells.PC_sampling()*

* (Fig. 2d) To plot cell surfaces at their locations in the 3D PCA shape space, or 'morphospace', uncomment the function:\
*cells.plot_PC_space(plot_original = False, max_l = 3)*\
Here, *plot_original = True / False* again dermines whether full or smoothed meshes are plotted, and max_l gives the degree of truncation for the spherical harmonic descriptor.


* To plot all time series and histograms of attributes across all cells (example attributes given), uncomment the function:\
*cells.plot_attributes(attributes = ['volume', 'pca0', 'pca1', 'pca2'])*

* To plot pairwise correlations between attributes (e.g. the examples given), uncomment the function:\
*cells.correlation(attributes = ['pca0', 'pca1', 'pca2', 'speed_uropod'])*\
Note: pairwise correlations between the PCs will show up blank.

* To scatter some attributes (e.g. the examples given) such that hovering over a point gives the cell ID and frame number, uncomment the function:\
*cells.scatter_annotate('speed_uropod', 'speed_centroid')*

* (Supplementary Fig. 2d) To plot how dimensionality varies along PC 1, the transition from spherical to polarised, uncomment the function:\
*cells.expl_var_bar_plot()*

* (Supplementary Fig. 2e) To plot the difference in spherical harmonic spectra between subpopulations across PC 1, uncomment the function:\
*cells.low_high_PC1_vecs()*


## Section 2: Morphodynamics analysis (autocorrelation functions (ACFs), morphodynamic spaces etc.)

### Section 2a: Global morphodynamic parameters (ACFs, power spectra)

In all cases, run the following command while in the *scripts* folder within *tcells_paper_code*, and uncomment the required function (details below):\
*python3 main_morphodynamics_global.py*

* (Supplementary Fig. 5b) To plot the ACF decay timescales, uncomment the function:\
*ACF()*

* (Supplementary Fig. 5c) To plot the power spetrum for each variable, uncomment the function:\
*run_power_spectrum()*  

### Section 2b: Continuous wavelet analysis for analysing local behaviours

In all cases, run the following command while in the *scripts* folder within *tcells_paper_code*, and uncomment the required function (details below):\
*python3 main_morphodynamics_cwt.py filename load_or_save_or_run*\
Here, *filename* is the name to load or save. Options for loading are (as stored in */data/cwt_saved*): *150* (all PCs), *150_PC1_run* (marginal PC1 dynamics for run mode), *150_PC2_run* (marginal PC2 dynamics for run mode), *150_PC3_run* (marginal PC3 dynamics for run mode). *load_or_save_or_run* is whether to load pre-saved data, save new ones, or simply run without saving.\
Note: as described in the main text (Methods 4.7), the maximum Gaussian wavelet with was reduced for the final representation of marginal PC 2 dynamics. This can be changed by selecting which *self.gaus1_scales* to use in */morphodynamics/cwt.py*

* To plot some of the longer time series (PCs 1-3 and retraction speed), uncomment the function:\
*cwt.plot_series()*

* To plot the wavelets, uncomment one of the following functions:\
*cwt.plot_wavelets(wavelet = 'mexh', scales = [0.5*i for i in np.linspace(2, 12, 6)])*
*cwt.plot_wavelets(wavelet = 'gaus1', scales = [0.4*i for i in np.linspace(2, 27, 6)])*

* To plot the spectrograms, uncomment the function:\
*cwt.plot_spectrograms()*

* (Fig. 5b, Supplementary Fig. 6b) To plot the t-SNE embeddings, uncomment the function:\
*cwt.plot_embeddings(path_of = 'CELL21a')*\
Here, *path_of=None* yields all embeddings (Supplementary Fig. 6b), coloured by cell. Setting *path_of* to a cell section index (e.g. *'CELL21a'*) plots the trajectory of that section (Fig. 5b).
Note on the cell section indices: these are the cell indices (e.g. CELL1, CELL2, CELL3...) but with a letter (a, b, c...) appended for each continuous section, since some videos have long gaps.
For example if cell 'CELL1' has two continuous sections with a gap, these have indices 'CELL1a' and 'CELL1b'. If the time series has no gaps, it will be 'CELL1a'

* (Fig. 4a) To plot the probability density function (PDF) over the morphodynamic space (found using kernel density estimation, KDE), uncomment the function:\
*cwt.set_kde(plot = True)*


* (Fig. 4b) To visualise the cell frame series, PC series, and spectrogram surrounding a certain frame (i.e. in the window where the wavelets are picking up information), uncomment the function:\
*show_cell_series_clustered(['CELL21a-72'])*\
Here, the input is a list of cell section indices. Note, the function that plots the t-SNE embeddings is interactive, so hovering the mouse over a point will show its index, and these can then be inputted here for the visualisation.


* (Fig. 4d) To plot the transition probability matrix, uncomment the function:\
*cwt.transition_matrix(grid = True, plot = True)*\
Here, *grid = True / False* means it will be computed over a grid, or over the stereotyped high PDF peaks, respectively.
