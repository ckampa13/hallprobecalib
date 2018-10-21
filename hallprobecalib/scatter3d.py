import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

def Scatter3d(x,y,z, cs, alpha=1, colorsMap='hot', psize=0.5):
    # x, y, z should be x,y,z of measured grid, cs should be whatever magnitude to plot...ie BField, FFT...use a colormap that is good with showing differences (hot,viridis,plasma,etc)
    ps = psize*np.ones(len(x))
    cm = plt.get_cmap(colorsMap)
    cNorm = Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), alpha=alpha)
    scalarMap.set_array(cs)
    cbar = fig.colorbar(scalarMap)
    ax.set_xlabel("X (microns)")
    ax.set_ylabel("Y (microns)")
    ax.set_zlabel("Z (microns)")
    cbar.ax.set_ylabel("B (T)")
    plt.show()
