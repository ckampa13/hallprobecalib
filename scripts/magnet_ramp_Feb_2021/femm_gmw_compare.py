import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
# local imports
from configs import plotdir, femmfile_70, GMW_currents, GMW_Bs
from plotting import config_plots
config_plots()


def load_data(femmfile, GMW_currents, GMW_Bs):
    # read FEMM file
    df = pd.read_csv(femmfile, skiprows=8, names=['I','B'])
    # construct GMW DataFrame
    df_GMW = pd.DataFrame({'I':GMW_currents, 'B': GMW_Bs})
    # create filtered FEMM DataFrame at specific GMW points
    df_ = df[np.isin(df.I,GMW_currents)].copy()
    # scale currents
    df_GMW.eval('I = I * 2', inplace=True)
    df.eval('I = I * 2', inplace=True)
    df_.eval('I = I * 2', inplace=True)
    return df_GMW, df, df_

def comparison_plot(df_GMW, df_FEMM, df_FEMM_part, savename=plotdir+'final_results/femm_gmw_comparison_ratio'):
    # set up figure with two axes
    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.3, 0.8, 0.6))
    ax2 = fig.add_axes((0.1, 0.1, 0.8, 0.2))
    # plot full FEMM calculation as line and GMW data as points
    ax1.plot(df_FEMM.I, df_FEMM.B, linewidth=2, color='red', label='FEMM Calculation', zorder=99)   
    ax1.plot(df_GMW.I, df_GMW.B, f'.', markersize=10, color='blue', label='GMW Calibration Curve', zorder=100)
    # add vertical line in both axes for physical limit of magnet
    ax1.plot([280, 280], [-0.5, 1.8], color='black', linestyle='-.', label='Physical Limit of Magnet', zorder=98)
    ax2.plot([280, 280], [0, 3], color='black', linestyle='-.', zorder=98)
    # calculate ratio, handling divide by 0 error
    a = df_FEMM_part.B
    b = df_GMW.B.values
    r = np.divide(a, b, out=np.ones_like(a), where=b!=0)
    # plot ratios
    ax2.plot(df_GMW.I.values, r, f'.', markersize=10, color='blue', zorder=99)
    # formatting
    # set ylimit ax1
    ax1.set_ylim([-0.1, 1.7])
    # set ylimit ax2
    ylim = 1.3*np.nanmax(np.abs(r-1))
    ax2.set_ylim([1-ylim, 1+ylim])
    # remove ticklabels for ax1 xaxis
    ax1.set_xticklabels([])
    # axis labels
    ax2.set_xlabel('Magnet Current [A]')
    ax2.set_ylabel(r'($|B|_{\mathrm{FEMM}}/|B|_{\mathrm{GMW}}$)')
    ax1.set_ylabel(r'$|B|$ [T]')
    # force consistent x axis range for ax1 and ax2
    I_min = np.min([df_GMW.I.min(), df_FEMM.I.min()])
    I_max = np.max([df_GMW.I.max(), df_FEMM.I.max()])
    ax1.set_xlim([I_min-10, I_max+10])
    ax2.set_xlim([I_min-10, I_max+10])
    # turn on legend
    ax1.legend()
    # add title
    fig.suptitle('FEMM Calculation vs. GMW Calibration Curve: 70 mm Gap')
    # inward ticks
    ax1.tick_params(axis='y', which='both', direction='in')
    ax2.tick_params(direction='in')
    # save figure
    fig.savefig(savename+'.pdf')
    fig.savefig(savename+'.png')

    return fig, ax1, ax2


if __name__=='__main__':
    print('Running script: femm_gmw_compare.py')
    time0 = datetime.now()
    df_GMW, df, df_ = load_data(femmfile_70, GMW_currents, GMW_Bs)
    fig, ax1, ax2 = comparison_plot(df_GMW, df, df_, plotdir+'final_results/femm_gmw_comparison_ratio')
    timef = datetime.now()
    #plt.show()
    print(f'Runtime: {timef-time0} [H:MM:SS])\n')
