import numpy as np

def equal_axes_3D(*axes):


    x_mids = []
    y_mids = []
    z_mids = []
    ax_ranges = []

    for ax in axes:
        x_min, x_max = ax.get_xlim()
        ax_ranges.append(x_max - x_min)
        x_mids.append((x_min + x_max)/2)
        y_min, y_max = ax.get_ylim()
        ax_ranges.append(y_max - y_min)
        y_mids.append((y_min + y_max)/2)
        z_min, z_max = ax.get_zlim()
        ax_ranges.append(z_max - z_min)
        z_mids.append((z_min + z_max)/2)

    max_range = np.array(ax_ranges).max()

    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(x_mids[ax_idx]-(max_range/2), x_mids[ax_idx]+(max_range/2))
        ax.set_ylim(y_mids[ax_idx]-(max_range/2), y_mids[ax_idx]+(max_range/2))
        ax.set_zlim(z_mids[ax_idx]-(max_range/2), z_mids[ax_idx]+(max_range/2))

def equal_axes_notSquare_3D(*axes):

    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    z_mins = []
    z_maxs = []

    for ax in axes:
        x_min, x_max = ax.get_xlim()
        x_mins.append(x_min)
        x_maxs.append(x_max)
        y_min, y_max = ax.get_ylim()
        y_mins.append(y_min)
        y_maxs.append(y_max)
        z_min, z_max = ax.get_zlim()
        z_mins.append(z_min)
        z_maxs.append(z_max)

    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(min(x_mins), max(x_maxs))
        ax.set_ylim(min(y_mins), max(y_maxs))
        ax.set_zlim(min(z_mins), max(z_maxs))

def equal_axes_notSquare_2D(*axes):

    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []

    for ax in axes:
        x_min, x_max = ax.get_xlim()
        x_mins.append(x_min)
        x_maxs.append(x_max)
        y_min, y_max = ax.get_ylim()
        y_mins.append(y_min)
        y_maxs.append(y_max)

    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(min(x_mins), max(x_maxs))
        ax.set_ylim(min(y_mins), max(y_maxs))


def remove_ticks(*axes):
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])



def no_pane(*axes):
    for ax in axes:
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
