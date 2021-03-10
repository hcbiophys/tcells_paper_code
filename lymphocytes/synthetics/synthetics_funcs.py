import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import matplotlib.tri as mtri
import lymphocytes.utils.plotting as utils_plotting


def plotRecon_singleDeg(ax, thetas, phis, xs, ys, zs):

    tris = mtri.Triangulation(phis, thetas)
    print(len(xs))
    collec = ax.plot_trisurf(xs, ys, zs, triangles = tris.triangles, linewidth = 0, antialiased = False)

    ax.grid(False)
    utils_plotting.label_axes_3D(ax)
    utils_plotting.equal_axes_3D(ax)
    utils_plotting.no_pane_3D(ax)

def find_closest(lymph_serieses, min_RI_vector, max_RI_vector):
    lymphs = [lymph for lymph_series in lymph_serieses.lymph_serieses for lymph in lymph_series]
    RI_vectors = [lymph.RI_vector for lymph in lymphs]
    min_distances = [np.linalg.norm(RI_vector-min_RI_vector) for RI_vector in RI_vectors]
    closest_to_min = lymphs[min_distances.index(min(min_distances))]
    max_distances = [np.linalg.norm(RI_vector-max_RI_vector) for RI_vector in RI_vectors]
    closest_to_max = lymphs[max_distances.index(min(max_distances))]

    return closest_to_min, closest_to_max

def linearly_interpolate(closest_to_min, closest_to_max):
    interpolated = []
    for t in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        interpolated.append((1-t)*closest_to_min.vector + t*closest_to_max.vector)
    return interpolated



def surface_from_vector(vector, max_l):
    split = np.split(vector, 3)
    split = [i[..., np.newaxis] for i in split]
    coeff_array = np.concatenate(split, axis = 1)


    thetas = np.linspace(0, np.pi, 50)
    phis = np.linspace(0, 2*np.pi, 50)
    thetas, phis = np.meshgrid(thetas, phis)
    thetas, phis = thetas.flatten(), phis.flatten()
    xs, ys, zs = [], [], []
    for idx_coord, coord_list in enumerate([xs, ys, zs]):
        for t, p in zip(thetas, phis):
            func_value = 0
            for l in np.arange(0, max_l + 1):
                for m in np.arange(0, l+1):
                    if m == 0:
                        a = coeff_array[l*l, idx_coord]
                        b = 0
                    else:
                        a = coeff_array[l*l + 2*m - 1, idx_coord]
                        b = coeff_array[l*l + 2*m, idx_coord]
                    clm = complex(a, b)
                    func_value += clm*sph_harm(m, l, p, t)
            coord_list.append(func_value.real)


    return thetas, phis, xs, ys, zs


def symmetric_PC_reconstructions(lymph_serieses):
    #mean = np.array([0, 5.11, 0.84,  0.73, 0.53, 0.45, 0.37, 0.33])
    fig = plt.figure()

    all = [[4.8193, 0.0572, 0.3514, 0.3106, 0.3156],
    [5.1691, 1.0033, 0.8127, 0.5706, 0.477],

    [4.7706, 1.3603, 0.0474, 0.4375, 0.2384],
    [5.1792, 0.7324, 0.8759, 0.5443, 0.4931],

    [4.4745, 0.7931, 0.8765, 1.2038, 0.6001],
    [5.2408, 0.8503, 0.7035, 0.3849, 0.4179]]

    for idx_PC in [0, 1, 2]:
        min_RI_vector, max_RI_vector = all[2*idx_PC], all[2*idx_PC+1]
        closest_to_min, closest_to_max = find_closest(lymph_serieses, min_RI_vector, max_RI_vector)

        interpolated = linearly_interpolate(closest_to_min, closest_to_max)

        for idx_vector, vector in enumerate(interpolated):
            thetas, phis, xs, ys, zs = surface_from_vector(vector, max_l = len(min_RI_vector))

            ax = fig.add_subplot(3, 6, 6*idx_PC + idx_vector+1, projection = '3d')
            plotRecon_singleDeg(ax, thetas, phis, xs, ys, zs)
