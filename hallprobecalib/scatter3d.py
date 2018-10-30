import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

def Scatter3d(x,y,z, cs, cslabel='B (T)', alpha=1, colorsMap='hot', psize=0.5, fig=None, ax=None, outline=True, outlinecolor='gray',outlinewidth=0.75,txt=False):
    # x, y, z should be x,y,z of measured grid, cs should be whatever magnitude to plot...ie BField, FFT...use a colormap that is good with showing differences (hot,viridis,plasma,etc)
    ps = psize*np.ones(len(x))
    cm = plt.get_cmap(colorsMap)
    cNorm = Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    if ax == None:
        fig = plt.figure()
        ax = Axes3D(fig)
    ax.scatter(x, y, z, s=ps, c=scalarMap.to_rgba(cs), alpha=alpha)
    if outline:
        GridOutline(x,y,z,ax,color=outlinecolor,linewidth=outlinewidth,txt=txt)
    scalarMap.set_array(cs)
    cbar = fig.colorbar(scalarMap)
    ax.set_xlabel("X (microns)")
    ax.set_ylabel("Y (microns)")
    ax.set_zlabel("Z (microns)")
    cbar.ax.set_ylabel(cslabel)
    plt.show()
    return fig,ax

def GridOutline(x,y,z,ax,color='gray',linewidth=0.75,txt=False):
    # note: ax must be of type Axes3d
    x1 = x.min()
    x2 = x.max()
    y1 = y.min()
    y2 = y.max()
    z1 = z.min()
    z2 = z.max()

    xs1 = [x1,x1,x1,x1,x1]
    ys1 = [y1,y1,y2,y2,y1]
    zs1 = [z1,z2,z2,z1,z1]

    xs2 = [x2,x2,x2,x2,x2]
    ys2 = [y1,y1,y2,y2,y1]
    zs2 = [z1,z2,z2,z1,z1]

    xs3 = [x1,x2,x2,x1,x1]
    ys3 = [y1,y1,y1,y1,y1]
    zs3 = [z1,z1,z2,z2,z1]

    xs4 = [x1,x2,x2,x1,x1]
    ys4 = [y2,y2,y2,y2,y2]
    zs4 = [z1,z1,z2,z2,z1]

    ax.plot(xs1,ys1,zs1,color=color,linewidth=linewidth)
    ax.plot(xs2,ys2,zs2,color=color,linewidth=linewidth)
    ax.plot(xs3,ys3,zs3,color=color,linewidth=linewidth)
    ax.plot(xs4,ys4,zs4,color=color,linewidth=linewidth)

    if txt:
        h = 0.1
        # parallel to x axis
        ax.text((x1+x2)/2.,y1,z1+h,f"y={y1}, z={z1}",horizontalalignment='center')
        ax.text((x1+x2)/2.,y1,z2+h,f"y={y1}, z={z2}",horizontalalignment='center')
        ax.text((x1+x2)/2.,y2,z1+h,f"y={y2}, z={z1}",horizontalalignment='center')
        ax.text((x1+x2)/2.,y2,z2+h,f"y={y2}, z={z2}",horizontalalignment='center')
        # parallel to y axis
        ax.text(x1,(y1+y2)/2.,z1+h,f"x={x1}, z={z1}",horizontalalignment='center')
        ax.text(x1,(y1+y2)/2.,z2+h,f"x={x1}, z={z2}",horizontalalignment='center')
        ax.text(x2,(y1+y2)/2.,z1+h,f"x={x2}, z={z1}",horizontalalignment='center')
        ax.text(x2,(y1+y2)/2.,z2+h,f"x={x2}, z={z2}",horizontalalignment='center')
        # parallel to z axis
        ax.text(x1+h,y1,(z1+z2)/2.,f"x={x1}, y={y1}",horizontalalignment='center')
        ax.text(x1+h,y2,(z1+z2)/2.,f"x={x1}, y={y2}",horizontalalignment='center')
        ax.text(x2+h,y1,(z1+z2)/2.,f"x={x2}, y={y1}",horizontalalignment='center')
        ax.text(x2+h,y2,(z1+z2)/2.,f"x={x2}, y={y2}",horizontalalignment='center')

    return ax
