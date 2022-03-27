# Code for the paper: T Cell Morphodynamics Reveal Periodic Shape Oscillations in 3D Migration

In all cases, run in scripts/

*python3 main.py all_run_stop*

all_run_stop can be all, run, or stop and these correspond to including all cells, the long videos from the run mode, and the long videos from the stop mode, respectively.

Uncomment the required function within main.py to run it



## Single Cell Functions
Running functions with a single cell video (for example, plotting its time series)
For the single cell functions, change *idx_cell = None* in main.py from *None* to the required cell code (the possible codes are shown in main.py)

* To plot how increasing the number of spherical harmonics (by including more degrees, l) decreases the smoothing (Fig. 1d), uncomment the function:

*cells.plot_l_truncations(idx_cell=idx_cell)*

* To plot the subsampled frames in a video, uncomment the function:
*cells.plot_orig_series(idx_cell = idx_cell, uropod_align = False, color_by = 'pca1', plot_every = 6, plot_flat = False)*
Here, cells will be coloured by PC (principal component) 2. Note: pca0, pca1, pca2 correspond to PCs 1, 2, and 3, respectively. Every 6 frames are plotted.

* To plot the same as above, but for reconstructions with truncated spherical harmonic representations, uncomment the function:
*cells.plot_recon_series(idx_cell = idx_cell, max_l = 1, color_by = None, plot_every=6)*
Here, the maximum spherical harmonic degree, l, is 1, there is no special colouring, and ever 6 frames are plotted.

* To plot the translucent surface segmentation of each frame in space as the cell migrates (Fig. 3c), uncomment the function:
*cells.plot_migratingCell(idx_cell=idx_cell, opacity = 0.2, plot_every = 40)*
Here, opacity determines the translucency, and every 40 frames are plotted.

* To plot the line joining the uropod label and centroid over the migration visualisation of the previous bullet point, uncomment the function:
*cells.plot_uropod_centroid_line(idx_cell = idx_cell, plot_every = 1)*

* To plot the PC series of a single cell, uncomment the function:
*cells.plot_series_PCs(idx_cell=idx_cell, plot_every=5)*



## Many Cell Functions
Running functions using many cell videos (for example, plotting attributes across all cells)

* To plot the histogram comparing uropod-centroid (UC) axis and ellipsoid axis (Supplementary Fig. 4b), uncomment the function:
*cells.alignments(min_length = 0.0025)*

* To plot the histograms of speed_uropod and speed_centroid (Supplementary Fig. 4d), uncomment the function:
*cells.speeds_histogram()*

* To plot the cumulative retraction speeds (i.e. speed_uropod), as in Fig. 3b, uncomment the function:
*cells.plot_cumulatives()*

* To plot the figure showing the emergence of bimodal run-and-stop behaviour (Fig. 3d), uncomment the function:
*cells.bimodality_emergence()*

* To plot cell surfaces sampled over the PCs, uncomment the function:
*cells.plot_component_frames(bin_size=7, pca=True, plot_original = False, max_l = 3)*
Here, plot_original = True / False is for Supplementary Fig. 2a (original surfaces) and Fig. 2b (using truncated, i.e. smoothed, representations), respectively. The bin size gives the number of frames shown, and max_l gives the degree of truncation for the spherical harmonic descriptor.

* To sample the min, mean and max along each PC in the spherical harmonic descriptor space (Fig. 2c), uncomment the function:
*cells.PC_sampling()*

* To plot cell surfaces at their locations in the 3D PCA shape space (Fig. 2d), or 'morphospace', uncomment the function:
*cells.plot_PC_space(plot_original = False, max_l = 3)*
Here, plot_original = True / False is for Supplementary Fig. 1a (original surfaces) and Fig. 2b (using truncated, i.e. smoothed, representations). max_l gives the degree of truncation for the spherical harmonic descriptor.


* To plot all time series and histograms of attributes across all cells (example attributes given), uncomment the function:
*cells.plot_attributes(attributes = ['volume', 'pca0', 'pca1', 'pca2'])*

* To plot pairwise correlations between attributes (e.g. the examples given), uncomment the function:
*cells.correlation(attributes = ['pca0', 'pca1', 'pca2', 'speed_uropod'])*
Note: pairwise correlations between the PCs will show up blank.


* To scatter some attributes (e.g. the examples given) such that hovering over a point gives the cell ID and frame number, uncomment the function:
*cells.scatter_annotate('speed_uropod', 'speed_centroid')*

* To plot how dimensionality varies along PC 1 (Supplementary Fig. 2d), the transition from spherical to polarised, uncomment the function:
*cells.expl_var_bar_plot()*

* To plot the difference in spherical harmonic spectra between subpopulations across PC 1 (Supplementary Fig. 2e), uncomment the function:
*#cells.low_high_PC1_vecs()*
