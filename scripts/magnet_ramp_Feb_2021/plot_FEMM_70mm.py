import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotting import config_plots
config_plots()
plt.rcParams["text.usetex"] = True

# ddir = '/home/ckampa/Desktop/digitized_plots/data/'
# fbase = ddir + 'GMW_field_vs_current_250mm-polecap'
# femm_file = 'gap70_changing_current_B_00_results.txt'
femm_file = '/home/ckampa/Desktop/misc_scripts/Lua/output/gap70_changing_current_B_00_results.txt'
df = pd.read_csv(femm_file, skiprows=8, names=['I','B'])

currents = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
GMW_Bs = np.array([0.00000,0.25612,0.50921,0.76165,0.99680,1.14767,1.25025,1.35284,1.42225,1.47993,1.51854])
# currents = np.array([0, 20, 40, 60, 80, 100, 120, 140])
# GMW_Bs = np.array([0.00000,0.25612,0.50921,0.76165,0.99680,1.14767,1.25025,1.35284])
df_GMW = pd.DataFrame({'I':currents, 'B': GMW_Bs})
df_ = df[np.isin(df.I,currents)].copy()
df_GMW.eval('I = I * 2', inplace=True)
df.eval('I = I * 2', inplace=True)
df_.eval('I = I * 2', inplace=True)


def make_scatter_plot(df_GMW=df_GMW, df_FEMM=df, df_FEMM_part=df_):
    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.3, 0.8, 0.6))
    ax2 = fig.add_axes((0.1, 0.1, 0.8, 0.2))
    ax1.plot(df_FEMM.I, df_FEMM.B, linewidth=2, color='red', label='FEMM Calculation')
    ax1.plot(df_GMW.I, df_GMW.B, f'+', markersize=10, color='blue', label='GMW Calibration Curve')
    ax2.scatter(df_GMW.I.values, df_FEMM_part.B.values-df_GMW.B.values, color='red', s=15)
    ax1.set_xticklabels([])
    ax2.set_xlabel('Magnet Current [A]')
    ax2.set_ylabel(r'($|B|_{\mathrm{FEMM}}-|B|_{\mathrm{GMW}}$) [T]')
    ax1.set_ylabel(r'$|B|$ [T]')
    I_min = np.min([df_GMW.I.min(), df_FEMM.I.min()])
    I_max = np.max([df_GMW.I.max(), df_FEMM.I.max()])
    ax1.set_xlim([-10, I_max+10])
    ax2.set_xlim([-10, I_max+10])
    # ax1.set_xlim([-10, 150])
    # ax2.set_xlim([-10, 150])
    ax1.legend()
    fig.suptitle('FEMM Calculation vs. GMW Calibration Curve: 70 mm Gap')

    # inward labels
    ax1.tick_params(axis='y', which='both', direction='in')
    ax2.tick_params(direction='in')

    fig.savefig('plots/femm_gmw_comparison.pdf')
    fig.savefig('plots/femm_gmw_comparison.png')

    return fig, ax1, ax2

def make_scatter_plot_ratio(df_GMW=df_GMW, df_FEMM=df, df_FEMM_part=df_):
    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.3, 0.8, 0.6))
    ax2 = fig.add_axes((0.1, 0.1, 0.8, 0.2))
    ax1.plot(df_FEMM.I, df_FEMM.B, linewidth=2, color='red', label='FEMM Calculation', zorder=99)
    ax1.plot(df_GMW.I, df_GMW.B, f'.', markersize=10, color='blue', label='GMW Calibration Curve', zorder=100)
    ax1.plot([280, 280], [-0.5, 1.8], color='black', linestyle='-.', label='Physical Limit of Magnet', zorder=98)
    r = df_FEMM_part.B.values/df_GMW.B.values
    ylim = 1.3*np.nanmax(np.abs(r-1))
    # ax2.scatter(df_GMW.I.values, r, color='blue', s=15)
    ax2.plot(df_GMW.I.values, r, f'.', markersize=10, color='blue')
    ax2.set_ylim([1-ylim, 1+ylim])
    ax1.set_xticklabels([])
    ax2.set_xlabel('Magnet Current [A]')
    ax2.set_ylabel(r'($|B|_{\mathrm{FEMM}}/|B|_{\mathrm{GMW}}$)')
    ax1.set_ylabel(r'$|B|$ [T]')
    I_min = np.min([df_GMW.I.min(), df_FEMM.I.min()])
    I_max = np.max([df_GMW.I.max(), df_FEMM.I.max()])
    ax1.set_xlim([I_min-10, I_max+10])
    ax1.set_ylim([-0.1, 1.7])
    ax2.set_xlim([I_min-10, I_max+10])
    # ax1.set_xlim([-10, 150])
    # ax2.set_xlim([-10, 150])
    ax1.legend()
    fig.suptitle('FEMM Calculation vs. GMW Calibration Curve: 70 mm Gap')

    # inward labels
    ax1.tick_params(axis='y', which='both', direction='in')
    ax2.tick_params(direction='in')

    fig.savefig('plots/femm_gmw_comparison_ratio.pdf')
    fig.savefig('plots/femm_gmw_comparison_ratio.png')

    return fig, ax1, ax2


if __name__=='__main__':
    _ = make_scatter_plot(df_GMW, df, df_)
    _ = make_scatter_plot_ratio(df_GMW, df, df_)
    plt.show()
