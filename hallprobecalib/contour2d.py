from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import seaborn as sns

def Contour2d(df, f="GRAD_B_MAG", coordSlice="X", posSlice=0., cutDir='<', quantile=0.25, fill=True, cmap="viridis", originCenter=True, alpha=1, linewidth=1., fig=None, ax=None):
    # passing in the pandas dataframe, variable for z, and positional slice
    # creates a contour plot

    if "X" in coordSlice:
        x = df.Y_ZAB.unique()  # .unique() converts from pandas series to np.array
        y = df.Z_ZAB.unique()
        sv = df.X_ZAB           # s for 'sliced variable'
        xlabel = 'Y (micrometers)'
        ylabel = 'Z (micrometers)'
    elif "Y" in coordSlice:
        x = df.X_ZAB.unique()
        y = df.Z_ZAB.unique()
        sv = df.Y_ZAB
        xlabel = 'X (micrometers)'
        ylabel = 'Z (micrometers)'
    else:
        x = df.X_ZAB.unique()
        y = df.Y_ZAB.unique()
        sv = df.Z_ZAB
        xlabel = 'X (micrometers)'
        ylabel = 'Y (micrometers)'

    # shift position values to have origin at center of slice
    sliceTitle = posSlice  # still want to print the slice number the user put in
    if originCenter==True:
        x -= np.median(x)
        y -= np.median(y)
        posSlice += np.median(sv)   # must put slice into dataframe coordinates for getting proper slice
        xlabel += ' (centered)'
        ylabel += ' (centered)'

    pslice = find_nearest(sv,posSlice)
    df_slice = (sv == pslice)

    ff = df[df_slice][f].abs()
    # # EXAMPLE OF HANDLING INCOMPLETE SLICE FROM GRADIENT CALCULATION
    # gridsize = len(x)*len(y)
    # if gridsize != len(df):
    #     np.append(f,np.zeros())
    q_cut = ff.quantile(quantile)
    if cutDir == '<':
        ff = np.array([value if value<q_cut else q_cut for value in ff])
    else:
        ff = np.array([value if value>q_cut else q_cut for value in ff])
    ff = np.reshape(ff,(len(x),len(y)))
    xx,yy = np.meshgrid(x,y,indexing='ij')

    if ax == None:
        # fig,ax = plt.subplots(figsize=(12, 9))
        fig = plt.figure()
        ax = Axes3D(fig)
    # note: somewhere in CS or clabel sometimes gives error
    # "RuntimeWarning: invalid value encountered in true_divide"
    if fill:
        step = q_cut / 5.
        m = np.max(ff)
        levels = np.arange(0.0, m, step) + step
        surf = ax.plot_surface(xx,yy,ff,cmap=plt.get_cmap(cmap),linewidth=0.,antialiased=False)
        fig.colorbar(surf,shrink=0.5,aspect=5)
        # CS = ax.contourf(xx, yy, ff)
        # CS = ax.contourf(xx, yy, ff,levels,cmap=cmap,alpha=0.5)
        # plt.colorbar(CS)
    else:
        CS = ax.contour(xx, yy, ff)
    # ax.clabel(CS,inline=1,fontsize=10,fmt='%1.1E')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{f}, Slice {coordSlice} = {sliceTitle}, Contour Plot")
    plt.show()
    return fig, ax


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
