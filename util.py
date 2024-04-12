import numpy as np
import matplotlib as mlib
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-10

def workstation_travel_time(ws1, ws2):
    return abs(ws1.x - ws2.x) + abs(ws1.y - ws2.y)

def plot_colored_grid(data, colors=['white', 'green'], bounds=[0, 0.5, 1], grid=True, labels=False, frame=True):
    """Plot 2d matrix with grid with well-defined colors for specific boundary values.

    :param data: 2d matrix
    :param colors: colors
    :param bounds: bounds between which the respective color will be plotted
    :param grid: whether grid should be plotted
    :param labels: whether labels should be plotted
    :param frame: whether frame should be plotted
    """

    # create discrete colormap
    cmap = mlib.colors.ListedColormap(colors)
    norm = mlib.colors.BoundaryNorm(bounds, cmap.N)

    # enable or disable frame
    plt.figure(frameon=frame)

    # show grid
    if grid:
        plt.grid(axis='both', color='k', linewidth=.5)
        plt.xticks(np.arange(0.5, data.shape[1], 1))  # correct grid sizes
        plt.yticks(np.arange(0.5, data.shape[0], 1))

    # disable labels
    if not labels:
        plt.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    # plot data matrix
    plt.imshow(data, cmap=cmap, norm=norm)

    # display main axis
    plt.show()