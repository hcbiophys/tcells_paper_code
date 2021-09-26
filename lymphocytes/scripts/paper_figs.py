import numpy as np
import matplotlib.pyplot as plt
import sys
from mayavi import mlab
from scipy.special import sph_harm
import pyvista as pv
pv.set_plot_theme("document")
from lymphocytes.data.dataloader_good_segs_1 import stack_attributes_1
from lymphocytes.data.dataloader_good_segs_2 import stack_attributes_2
from lymphocytes.cells.cells_class import Cells
import lymphocytes.utils.general as utils_general

def visualise_spherical_harmonics():
    """
    Visualise the real parts of some spherical harmonics on the sphere
    """
    r = 0.3

    phi, theta = np.mgrid[0:np.pi:101j, 0:2*np.pi:101j]

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))

    for l in range(1, 5):
        for m in range(l):
            s = sph_harm(m, l, theta, phi).real

            mlab.mesh(x - l + 0.3*l, y - m + 0.3*m, z, scalars=s, colormap='jet')

    mlab.view(90, 0)
    mlab.show()


def cell_mapping_to_sphere(idx_cell = 2):

    cells = Cells(stack_attributes_1+stack_attributes_2, cells_model = [idx_cell], max_l = 15)
    lymph = cells.cells[idx_cell][0]

    # reconstruction
    plotter = pv.Plotter()
    lymph.plotRecon_singleDeg(plotter=plotter)
    plotter.show()
    sys.exit()



    r = 0.3
    phi, theta = np.mgrid[0:np.pi:101j, 0:2*np.pi:101j]

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    xs, ys, zs = np.zeros(phi.shape, dtype=complex), np.zeros(phi.shape, dtype=complex), np.zeros(phi.shape, dtype=complex)

    for coord_idx, coord_array in zip([0, 1, 2], [xs, ys, zs]):
        for l in np.arange(1, 15 + 1):
            for m in np.arange(0, l+1):
                clm = lymph._get_clm(coord_idx, l, m)
                coord_array += clm*sph_harm(m, l, phi, theta)

    xs, ys, zs = xs.real, ys.real, zs.real
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
    mlab.mesh(x, y, z, scalars=xs, colormap='jet')
    mlab.mesh(x+0.7, y, z, scalars=ys, colormap='jet')
    mlab.mesh(x+1.4, y, z, scalars=zs, colormap='jet')

    mlab.view(90, 90)
    mlab.show()

def justify_extra_descriptor_var():
    cells = Cells(stack_attributes_1+stack_attributes_2, cells_model = [4], max_l = 15) # I think I have to ignore stack_attributes_2 as these are duplicates?
    lymph1 = cells.cells[4][10]
    lymph2 = cells.cells[4][50]
    fig = plt.figure(figsize = (1.1, 0.9))
    for idx_lymph, lymph in enumerate([lymph1, lymph2]):
        ax = fig.add_subplot(2, 1, idx_lymph+1)
        lymph.RI_vector = lymph.RI_vector[:5]
        ax.bar(range(lymph.RI_vector.shape[0]), lymph.RI_vector, color = ['red'] + ['blue']*4)
        ax.set_ylim([0, 3.7])
    for ax in fig.axes:
        ax.tick_params(axis='both', which='major', labelsize=6, pad = 1)
        ax.set_xticks([])

    #plt.show()
    plt.subplots_adjust(hspace = 0, wspace = 0)
    plt.savefig('/Users/harry/Desktop/d0.png', dpi = 500)


if __name__ == '__main__':
    #visualise_spherical_harmonics()
    #cell_mapping_to_sphere()
    justify_extra_descriptor_var()
