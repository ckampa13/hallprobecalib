import numpy as np
import pandas as pd
import pickle as pkl
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
# local imports
from configs import (
    pkl_interp_fcn_Hall,
    pkl_interp_fcn_NMR,
    pklfit_temp_hall,
    pklfit_temp_nmr,
    pklinfo,
    pklinfo_hall_regress,
    pklinfo_nmr_regress,
    pklproc,
    pklproc_hyst,
    pklproc_ramp,
    plotdir,
    probe,
    T0,
)
# from factory_funcs import get_NMR_B_at_T0_func, get_Hall_B_at_T0_func
# from femm_fits import load_data
# from model_funcs import ndeg_poly1d
from preprocess_data import get_probe_IDs
from plotting import config_plots, get_label, datetime_plt
config_plots()


def plot2d(df,  x, y, xl=None, yl=None, s=2, c='blue', legendlab=None,
           title=None, query=None, scix=False, sciy=False, zorder=None,
           pfile=None):
    if not query is None:
        df_ = df.query(query).copy()
    else:
        df_ = df.copy()
    if xl is None:
        xl = x
    if yl is None:
        yl = y

    fig, ax = plt.subplots()
    ax.scatter(df_[x], df_[y], s=s, c=c, label=legendlab, zorder=zorder)
    # formatting
    # scientific notation
    if scix:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    if sciy:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # ticks in
    ax.tick_params(which='both', direction='in', top=True, right=True)
    # autolocate minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    # labels
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    fig.suptitle(title)

    if not legendlab is None:
        ax.legend()

    if not pfile is None:
        fig.savefig(pfile+'.pdf')
        fig.savefig(pfile+'.png')
    return fig, ax


if __name__ == '__main__':
    print('Running script: misc_plots.py')
    time0 = datetime.now()
    # setup plot directory
    pdir = plotdir+'final_results/misc/'
    # load data
    results_nmr = pkl.load(open(pklfit_temp_nmr, 'rb'))
    results_hall = pkl.load(open(pklfit_temp_hall, 'rb'))
    df_info = pd.read_pickle(pklinfo)
    proc_ramp = pd.read_pickle(pklproc_ramp)
    probes = get_probe_IDs(proc_ramp)
    # generic 2d scatter plot functions
    # yoke vs current
    _ = plot2d(df=proc_ramp, x='Magnet Current [A]', y='Yoke (center magnet)',
               xl=None, yl='Yoke (center magnet) [deg C]', s=2,
               legendlab='Data', title=None, query=None, scix=False,
               sciy=False, zorder=99, pfile=None)
    fig, ax = _
    ax.plot([0, 280], [T0, T0], 'r--', label='HVAC Setting', zorder=97)
    # ax.plot([192.064, 192.064], [12, 25], '-.', c='gray',
    #         label='Chiller Failed', zorder=98)
    ax.fill_between([192.064, 285.], [12, 12], [25, 25], color='gray',
                    alpha=0.5, label='Chiller Failed', zorder=98)
    ax.legend().set_zorder(100)
    fig.savefig(pdir+'Yoke_vs_I.pdf')
    fig.savefig(pdir+'Yoke_vs_I.png')
    # roof vs. current
    _ = plot2d(df=proc_ramp, x='Magnet Current [A]', y='Roof',
               xl=None, yl='Roof Temperature [deg C]', s=2,
               legendlab='Data', title=None, query=None, scix=False,
               sciy=False, zorder=99, pfile=None)
    fig, ax = _
    ax.plot([0, 280], [T0, T0], 'r--', label='HVAC Setting', zorder=97)
    ax.fill_between([192.064, 285.], [T0-0.5, T0-0.5], [T0+0.5, T0+0.5],
                    color='gray', alpha=0.5, label='Chiller Failed', zorder=98)
    ax.legend().set_zorder(100)
    fig.savefig(pdir+'Roof_vs_I.pdf')
    fig.savefig(pdir+'Roof_vs_I.png')
    # chamber wall vs. current
    # 13-20
    _ = plot2d(df=proc_ramp, x='Magnet Current [A]', y='Chamber Wall CH13',
               xl=None, yl='Chamber Wall Temperature [deg C]', s=1,
               legendlab='CH13', title=None, query=None, scix=False,
               sciy=False, zorder=99, pfile=None)
    fig, ax = _
    for i in range(14, 21):
        ax.scatter(proc_ramp['Magnet Current [A]'],
                   proc_ramp[f'Chamber Wall CH{i}'], s=1, label=f'CH{i}',
                   zorder=i+100-14)
    ax.plot([0, 280], [T0, T0], 'r--', label='HVAC Setting', zorder=97)
    l = 14.; h = 17.5
    ax.fill_between([192.064, 285.], [l, l], [h, h], color='gray',
                    alpha=0.5, label='Chiller Failed', zorder=98)
    ax.legend().set_zorder(115)
    fig.savefig(pdir+'Chamber_Walls_vs_I.pdf')
    fig.savefig(pdir+'Chamber_Walls_vs_I.png')
    # yoke vs current + hall probe temp
    _ = plot2d(df=proc_ramp, x='Magnet Current [A]', y='Yoke (center magnet)',
               xl=None, yl='Temperature [deg C]', s=2, c='brown',
               legendlab='Yoke', title=None, query=None, scix=False,
               sciy=False, zorder=99, pfile=None)
    fig, ax = _
    # for i,p in enumerate(probes):
    for i,pc in enumerate(zip(probes, ['red', 'blue', 'purple', 'green'])):
        p, c = pc
        m = np.mean(proc_ramp[f'{p}_Cal_Bmag'])
        print(f'{p}: |B|_mean = {m} [T]')
        ax.scatter(proc_ramp['Magnet Current [A]'],
                   proc_ramp[f'{p}_Cal_T'], s=2, c=c,
                   label=f'Hall probe ({p})', zorder=i+100)
    ax.plot([0, 280], [T0, T0], 'r--', label='HVAC Setting', zorder=97)
    ax.fill_between([192.064, 285.], [12, 12], [25, 25], color='gray',
                    alpha=0.5, label='Chiller Failed', zorder=98)
    ax.legend().set_zorder(110)
    fig.savefig(pdir+'Yoke-Halls_vs_I.pdf')
    fig.savefig(pdir+'Yoke-Halls_vs_I.png')
    # probes B vs. I
    plt.rcParams.update({'font.size': 22})
    probes_ = [p for p in probes if p != probe]
    _ = plot2d(df=proc_ramp, x='Magnet Current [A]', y=f'{probe}_Cal_Bmag',
               xl=None, yl=r'$|B|$ [T]', s=2, c='red',
               legendlab=f'Hall probe ({probe})', title=None, query=None, scix=False,
               sciy=False, zorder=107, pfile=None)
    fig, ax = _
    for i,pc in enumerate(zip(probes_, ['blue', 'purple', 'green'])):
        p, c = pc
        # m = np.mean(proc_ramp[f'{p}_Cal_Bmag'])
        # print(f'{p}: |B|_mean = {m} [T]')
        ax.scatter(proc_ramp['Magnet Current [A]'],
                   proc_ramp[f'{p}_Cal_Bmag'], s=2, c=c,
                   label=f'Hall probe ({p})', zorder=i+100)
    ax.set_ylim([-0.1, 1.5])
    ax.legend(fontsize=20).set_zorder(110)
    fig.savefig(pdir+'B_vs_I_all_Hall.pdf')
    fig.savefig(pdir+'B_vs_I_all_Hall.png')
    # reset configs
    config_plots()
    # print runtime
    timef = datetime.now()
    print(f'Runtime: {timef-time0} [H:MM:SS])\n')
    plt.show()
