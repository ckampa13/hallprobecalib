import numpy as np
import pandas as pd
import pickle as pkl
from datetime import datetime
import matplotlib as mpl
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
    pklraw,
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

    fig.tight_layout()

    if not pfile is None:
        fig.savefig(pfile+'.pdf')
        fig.savefig(pfile+'.png')
    return fig, ax

def errorbar2d(df, x, y, yerr, xl=None, yl=None, ms=2, c='blue', cbar=None,
               cbarlab=None, cb_discrete=False, legendlab=None, title=None,
               query=None, scix=False, sciy=False, zorder=None, fig=None,
               ax=None, pfile=None):
    if not query is None:
        df_ = df.query(query).copy()
    else:
        df_ = df.copy()
    if xl is None:
        xl = x
    if yl is None:
        yl = y
    if zorder is None:
        zorder = 0
    if fig is None:
        fig, ax = plt.subplots()
    ax.errorbar(df_[x], df_[y], yerr=df_[yerr], c=c,
                fmt='o', ls='none', ms=ms, zorder=zorder, capsize=4,
                label=legendlab)
    if not cbar is None:
        if cb_discrete:
            cmap = mpl.cm.viridis
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap',
                                                                cmaplist,
                                                                cmap.N)
            bounds = np.arange(np.min(df_[cbar]), np.max(df_[cbar])+2)-0.5
            bounds_label = np.arange(np.min(df_[cbar]), np.max(df_[cbar])+1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            sc = ax.scatter(df_[x], df_[y], s=ms**2, c=df_[cbar], cmap=cmap,
                            norm=norm, zorder=zorder+1, alpha=1.)
            cb = fig.colorbar(sc, ticks=bounds_label)
        else:
            sc = ax.scatter(df_[x], df_[y], s=ms**2, c=df_[cbar],
                            zorder=zorder+1, alpha=1.)
            cb = fig.colorbar(sc)
        if cbarlab is None:
            cbarlab = cbar
        cb.set_label(cbarlab)

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

    fig.tight_layout()

    if not pfile is None:
        fig.savefig(pfile+'.pdf')
        fig.savefig(pfile+'.png')
    return fig, ax

def hist(df,  x, xl=None, nbins=100, lw=2, c='blue', title=None, query=None,
         scix=False, sciy=False, logy=False, zorder=None, pfile=None):
    if not query is None:
        df_ = df.query(query).copy()
    else:
        df_ = df.copy()
    if xl is None:
        xl = x
    yl = 'Count'
    # if yl is None:
    #     yl = y
    blim = np.max(np.abs(df_[x]))*1.1
    bins = np.linspace(-blim, blim, nbins+1)

    fig, ax = plt.subplots()
    ax.hist(df_[x], bins=bins, histtype='step', linewidth=lw, color=c,
            zorder=zorder, label=get_label(df_[x], bins),
            density=False)
    # formatting
    # scientific notation
    if scix:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    if sciy:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # logy
    if logy:
        ax.set_yscale('log')
    # ticks in
    ax.tick_params(which='both', direction='in', top=True, right=True)
    # autolocate minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    # labels
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    fig.suptitle(title)

    # if not legendlab is None:
    ax.legend()

    fig.tight_layout()

    if not pfile is None:
        fig.savefig(pfile+'.pdf')
        fig.savefig(pfile+'.png')
    return fig, ax

def get_param_val_unc(result, param):
    try:
        p = result.params[param]
        return p.value, p.stderr
    except:
        return np.nan, np.nan

def make_param_arrays(results, param):
    p_tuples = [get_param_val_unc(r, param)
                for k, r in sorted(results.items())]
    p_vals = np.array([i[0] for i in p_tuples])
    p_errs = np.array([i[1] for i in p_tuples])
    return p_vals, p_errs

def load_df_info_with_params(pklinfo, pklfit_nmr, pklfit_hall):
    df = pd.read_pickle(pklinfo)
    df['run_num'] = df.index # helpful for colorbar on plots
    results_nmr = pkl.load(open(pklfit_nmr, 'rb'))
    results_hall = pkl.load(open(pklfit_hall, 'rb'))
    A_N, A_err_N = make_param_arrays(results_nmr, 'A')
    B_N, B_err_N = make_param_arrays(results_nmr, 'B')
    A_H, A_err_H = make_param_arrays(results_hall, 'A')
    B_H, B_err_H = make_param_arrays(results_hall, 'B')
    C_H, C_err_H = make_param_arrays(results_hall, 'C')
    df['A_NMR'] = A_N
    df['A_NMR_err'] = A_err_N
    df['B_NMR'] = B_N
    df['B_NMR_err'] = B_err_N
    df['A_Hall'] = A_H
    df['A_Hall_err'] = A_err_H
    df['B_Hall'] = B_H
    df['B_Hall_err'] = B_err_H
    df['B_Hall'] = B_H
    df['B_Hall_err'] = B_err_H
    return df


if __name__ == '__main__':
    print('Running script: misc_plots.py')
    time0 = datetime.now()
    # setup plot directory
    pdir = plotdir+'final_results/misc/'
    # load data
    results_nmr = pkl.load(open(pklfit_temp_nmr, 'rb'))
    results_hall = pkl.load(open(pklfit_temp_hall, 'rb'))
    # df_info = pd.read_pickle(pklinfo)
    proc_ramp = pd.read_pickle(pklproc_ramp)
    probes = get_probe_IDs(proc_ramp)
    # load df info with fit parameters
    df_info = load_df_info_with_params(pklinfo, pklfit_temp_nmr,
                                       pklfit_temp_hall)
    # hysteresis
    df_ = df_info.query('(ramp) & (run_num != 11)')
    sub_factor = {f'{row.I:0.0f}':row.A_NMR for row in df_.itertuples()}
    df_info['A_NMR_shift'] = np.array([row.A_NMR - sub_factor[f'{row.I:0.0f}']
                                       for row in df_info.itertuples()])
    # stability data
    df_proc = pd.read_pickle(pklproc).query('`Magnet Current [A]` > 125.')
    # subtract mean
    sub_arrays = []
    subt_arrays = []
    for rn in df_proc.run_num.unique():
        # no temp. correction
        df_ = df_proc.query(f'run_num == {rn}').copy()
        mnmr = df_['NMR [T]'].mean()
        sub_ = df_['NMR [T]'].values - mnmr
        sub_arrays.append(sub_)
        # with temp correction
        res = results_nmr[rn]
        B_ = res.eval(x=df_['Yoke (center magnet)'].values)
        subt_ = df_['NMR [T]'].values - B_
        subt_arrays.append(subt_)
    df_proc['NMR_shift'] = np.concatenate(sub_arrays)
    df_proc['NMR_shift_temp'] = np.concatenate(subt_arrays)

    # test output
    print('info (with params):')
    print(df_info)
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
               legendlab=f'Hall probe ({probe})', title=None, query=None,
               scix=False, sciy=False, zorder=107, pfile=None)
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
    # fit parameter summaries (temperature regression)
    # NMR
    pfile = pdir+'param_summary_{0}_{1}'
    # A
    _ = errorbar2d(df=df_info, x='I', y='A_NMR',
                   yerr='A_NMR_err', xl='Magnet Current [A]', yl=r'$A$ [T]',
                   ms=2, c='black', legendlab=None,
                   title='NMR Temperature Regression Summary\n'+
                   rf'$y = A + B (x - {T0})$', query='I>100',
                   cbar='run_num', cbarlab='Run Index', cb_discrete=True,
                   fig=None, ax=None,
                   scix=False, sciy=False, zorder=None,
                   pfile=pfile.format('A', 'NMR'))
    fig, ax = _
    # B
    _ = errorbar2d(df=df_info, x='I', y='B_NMR',
                   yerr='B_NMR_err', xl='Magnet Current [A]',
                   yl=r'$B$ [T/$^\circ$C]', ms=4, c='black',
                   legendlab='Fit to NMR',
                   title='NMR Temperature Regression Summary\n'+
                   rf'$y = A + B (x - {T0})$', query='I>100',
                   cbar='run_num', cbarlab='Run Index', cb_discrete=True,
                   fig=None, ax=None,
                   scix=False, sciy=True, zorder=100,
                   pfile=pfile.format('B', 'NMR'))
    fig, ax = _
    # add chiller and no chiller
    ss = 100
    df_ = df_info.query('(chiller) & (NMR)')
    ax.scatter(df_['I'], df_['B_NMR'], marker='s', s=ss, alpha=0.2,
               c='green', label='Water Chiller', zorder=98)
    df_ = df_info.query('(not chiller) & (NMR)')
    ax.scatter(df_['I'], df_['B_NMR'], marker='s', s=ss, alpha=0.2,
               c='red', label='No Water Chiller', zorder=99)
    ax.legend().set_zorder(101)
    fig.savefig(pdir+'hysteresis_NMR_param_B_chiller-marked.pdf')
    fig.savefig(pdir+'hysteresis_NMR_param_B_chiller-marked.png')
    # Hysteresis
    # A
    # all
    _ = errorbar2d(df=df_info, x='I', y='A_NMR_shift',
                   yerr='A_NMR_err', xl='Magnet Current [A]',
                   yl=r'$\Delta A$ [T] (from measurement in ramp)',
                   ms=4, c='black', legendlab='Fit to NMR',
                   title='Hysteresis from NMR Temperature Regression\n'+
                   rf'$y = A + B (x - {T0})$', query='(I>100) & (hyst)',
                   cbar='run_num', cbarlab='Run Index', cb_discrete=True,
                   fig=None, ax=None,
                   scix=False, sciy=True, zorder=100,
                   pfile=pdir+'hysteresis_NMR_param_A')
    fig, ax = _
    # add chiller and no chiller
    ss = 100
    df_ = df_info.query('(chiller) & (hyst) & (NMR)')
    ax.scatter(df_['I'], df_['A_NMR_shift'], marker='s', s=ss, alpha=0.2,
               c='green', label='Water Chiller', zorder=98)
    df_ = df_info.query('(not chiller) & (hyst) & (NMR)')
    ax.scatter(df_['I'], df_['A_NMR_shift'], marker='s', s=ss, alpha=0.2,
               c='red', label='No Water Chiller', zorder=99)
    ax.legend().set_zorder(101)
    fig.savefig(pdir+'hysteresis_NMR_param_A_chiller-marked.pdf')
    fig.savefig(pdir+'hysteresis_NMR_param_A_chiller-marked.png')
    # chiller
    _ = errorbar2d(df=df_info, x='I', y='A_NMR_shift',
                   yerr='A_NMR_err', xl='Magnet Current [A]',
                   yl=r'$\Delta A$ [T] (from measurement in ramp)',
                   ms=2, c='black', legendlab=None,
                   title='Hysteresis from NMR Temperature Regression\n'+
                   rf'$y = A + B (x - {T0})$',
                   query='(I>100) & (hyst) & (chiller)',
                   cbar='run_num', cbarlab='Run Index', cb_discrete=True,
                   fig=None, ax=None,
                   scix=False, sciy=True, zorder=None,
                   pfile=pdir+'hysteresis_NMR_param_A_chiller')
    fig, ax = _
    # no chiller
    _ = errorbar2d(df=df_info, x='I', y='A_NMR_shift',
                   yerr='A_NMR_err', xl='Magnet Current [A]',
                   yl=r'$\Delta A$ [T] (from measurement in ramp)',
                   ms=2, c='black', legendlab=None,
                   title='Hysteresis from NMR Temperature Regression\n'+
                   rf'$y = A + B (x - {T0})$',
                   query='(I>100) & (hyst) & (not chiller)',
                   cbar='run_num', cbarlab='Run Index', cb_discrete=True,
                   fig=None, ax=None,
                   scix=False, sciy=True, zorder=None,
                   pfile=pdir+'hysteresis_NMR_param_A_nochiller')
    fig, ax = _

    # histograms
    # stability (mean subtraction)
    _ = hist(df_proc, x='NMR_shift',
             xl=r'$\Delta |B|_\mathrm{NMR}$ [T] (Data - Run Mean)',
             nbins=100, lw=2, c='blue',
             title='System Stability: After Temperature Stabilization',
             query=None, scix=True, sciy=False, logy=False, zorder=None,
             pfile=pdir+'NMR_stability')
    fig, ax = _
    # stability (temperature correction)
    _ = hist(df_proc, x='NMR_shift_temp',
             xl=r'$\Delta |B|_\mathrm{NMR}$ [T] (Data - Temp. Regression Fit)',
             nbins=100, lw=2, c='blue',
             title='System Stability: After Temperature Regression',
             query=None, scix=True, sciy=False, logy=False, zorder=None,
             pfile=pdir+'NMR_stability_tempregress')
    fig, ax = _
    '''
    m = np.mean(df_proc['NMR_shift_temp'])
    s = np.std(df_proc['NMR_shift_temp'])
    N = len(df_proc)
    xs = np.linspace(-1e-5, 1e-5, 100)
    from scipy.stats import norm
    ys = norm.pdf(xs, loc=m, scale=s)
    print(xs, ys)
    ax.plot(xs, ys, 'r-')
    '''

    # reset configs
    config_plots()
    # print runtime
    timef = datetime.now()
    print(f'Runtime: {timef-time0} [H:MM:SS])\n')
    # plt.show()
