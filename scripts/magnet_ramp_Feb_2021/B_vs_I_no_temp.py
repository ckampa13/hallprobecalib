import numpy as np
import pandas as pd
import pickle as pkl
import lmfit as lm
from scipy.interpolate import interp1d
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
# local imports
from configs import (
    femmfile_75_Hall,
    femmfile_75_NMR,
    Hall_currents,
    NMR_currents,
    pkl_interp_fcn_Hall,
    pkl_interp_fcn_NMR,
    pklfit_temp_hall,
    pklfit_temp_hall_nmr,
    pklfit_temp_nmr,
    pklinfo,
    pklinfo_hall_regress,
    pklinfo_nmr_regress,
    pklproc,
    pklproc_hyst,
    pklproc_ramp,
    plotdir,
    probe,
)
from factory_funcs import get_NMR_B_at_T0_func, get_Hall_B_at_T0_func
from femm_fits import load_data
from model_funcs import ndeg_poly1d
from plotting import config_plots, get_label
config_plots()


def get_results_lists(pklfit_temp_nmr, pklfit_temp_hall, pklfit_temp_hall_nmr):
    results_nmr = pkl.load(open(pklfit_temp_nmr, 'rb'))
    results_hall = pkl.load(open(pklfit_temp_hall, 'rb'))
    results_hall_nmr = pkl.load(open(pklfit_temp_hall_nmr, 'rb'))
    return results_nmr, results_hall, results_hall_nmr

def create_all_funcs(pklfit_temp_nmr, pklfit_temp_hall, pklfit_temp_hall_nmr):
    temp = get_results_lists(pklfit_temp_nmr, pklfit_temp_hall,
                             pklfit_temp_hall_nmr)
    results_nmr, results_hall, results_hall_nmr = temp
    # use factory funcs
    get_NMR = get_NMR_B_at_T0_func(results_nmr)
    get_Hall = get_NMR_B_at_T0_func(results_hall)
    get_Hall_NMR = get_Hall_B_at_T0_func(results_hall_nmr, results_nmr)
    # return funcs
    return get_NMR, get_Hall, get_Hall_NMR

def regressed_Bs(get_NMR, get_Hall, get_Hall_NMR, df_info,
                 pklinfo_hall_regress, pklinfo_nmr_regress, T0=15):
    # split dataframe
    df_NMR = df_info.query('ramp & NMR').copy()
    df_Hall = df_info.query('ramp').copy()
    # save mask for later
    mask_NMR = (df_Hall['ramp'] & df_Hall['NMR']).values
    # generate data and errors
    # nmr
    NMR_tuples = [get_NMR(T0, i) for i in df_NMR.index]
    NMRs = np.array([i[0] for i in NMR_tuples])
    NMR_errs = np.array([i[1] for i in NMR_tuples])
    # hall no nmr
    Hall_tuples = [get_Hall(T0, i) for i in df_Hall.index]
    Halls = np.array([i[0] for i in Hall_tuples])
    Hall_errs = np.array([i[1] for i in Hall_tuples])
    # hall with nmr
    Hall_NMR_tuples = [get_Hall_NMR(T0, i) for i in df_Hall.index]
    Hall_NMRs = np.array([i[0] for i in Hall_NMR_tuples])
    Hall_NMR_errs = np.array([i[1] for i in Hall_NMR_tuples])
    # ratios
    Rs = Hall_NMRs[mask_NMR] / NMRs
    sigma_Rs = Rs * ((Hall_NMR_errs[mask_NMR]/Hall_NMRs[mask_NMR])**2 +
                     (NMR_errs/NMRs)**2)**(1/2)
    # save to respective info dataframes
    df_NMR['B_reg'] = NMRs
    df_NMR['sigma_B_reg'] = NMR_errs
    # B_Hall / B_NMR
    df_NMR['B_ratio'] = Rs
    df_NMR['sigma_B_ratio'] = sigma_Rs
    # Hall dataframe
    df_Hall['B_reg'] = Hall_NMRs
    df_Hall['sigma_B_reg'] = Hall_NMR_errs
    df_Hall['B_reg_no_NMR'] = Halls
    df_Hall['sigma_B_reg_no_NMR'] = Hall_errs
    # pickle dataframes
    df_NMR.to_pickle(pklinfo_nmr_regress)
    df_Hall.to_pickle(pklinfo_hall_regress)
    return df_NMR, df_Hall

def plot_B_vs_I(df_NMR, df_Hall, plotfile=None):
    fig, ax = plt.subplots()
    ax.errorbar(df_NMR.I, df_NMR.B_reg, yerr=df_NMR.sigma_B_reg, c='blue',
                fmt='o', ls='none', ms=4, zorder=102, capsize=2,
                label='NMR')
    ax.errorbar(df_Hall.I, df_Hall.B_reg, yerr=df_Hall.sigma_B_reg, c='red',
                fmt='o', ls='none', ms=4, zorder=101, capsize=2,
                label='Hall')
    ax.errorbar(df_Hall.I, df_Hall.B_reg_no_NMR,
                yerr=df_Hall.sigma_B_reg_no_NMR, c='green',
                fmt='o', ls='none', ms=4, zorder=100, capsize=2,
                label='Hall (no NMR)')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('Magnet Current [A]')
    ax.set_ylabel(r'$|B|$ [T]')
    ax.set_title(r'$B$ vs. $I$ Regressed Data')
    ax.legend().set_zorder(103)
    if not plotfile is None:
        fig.savefig(plotfile+'.pdf')
        fig.savefig(plotfile+'.pdf')
    return fig, ax

def plot_Hall_compare(df_Hall, I_cut=120, plotfile=None):
    # calculate difference and uncertainties
    df_ = df_Hall.query(f'I>{I_cut}').copy()
    dB = (df_['B_reg'] - df_['B_reg_no_NMR']).values
    sigma_dB = (df_['sigma_B_reg']**2 + df_['sigma_B_reg_no_NMR']**2)**(1/2)
    # plot
    fig, ax = plt.subplots()
    ax.errorbar(df_.I, dB, yerr=sigma_dB, c='red', fmt='o', ls=None, ms=4,
               zorder=101, capsize=2)
    # 1e-4 line
    ax.plot([I_cut-2, 283], [-1e-4, -1e-4], color='black', linestyle='-.',
            zorder=98)
    ax.plot([I_cut-2, 283], [1e-4, 1e-4], color='black', linestyle='-.',
            zorder=98)
    # ax.scatter(df_Hall.I, df_Hall.B_reg-df_Hall.B_reg_no_NMR, c='red', s=5,
    #            zorder=101)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_xlabel('Magnet Current [A]')
    #ax.set_ylabel(r'$|B|_\mathrm{Hall with NMR} -$'+
    #              r'$ |B|_\mathrm{Hall no NMR}$ [T]')
    ax.set_ylabel(r'$|B|_\mathrm{Hall+NMR} -$'+
                  r'$ |B|_\mathrm{Hall}$ [T]')
    ax.set_title('Hall Probe Regression Comparison')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylim([-6e-4, 6e-4])
    #ax.legend().set_zorder(103)
    if not plotfile is None:
        fig.savefig(plotfile+'.pdf')
        fig.savefig(plotfile+'.png')
    return fig, ax

def plot_R_vs_I(df_NMR, femm_nmr, femm_nmr_meas, femm_hall, femm_hall_meas,
                I_cut=120, plotfile=None):
    # ratio from FEMM
    # cut on current
    fn_ = femm_nmr.query(f'I >= {I_cut}').copy()
    fnm_ = femm_nmr_meas.query(f'I >= {I_cut}').copy()
    fh_ = femm_hall.query(f'I >= {I_cut}').copy()
    fhm_ = femm_hall_meas.query(f'I >= {I_cut}').copy()
    I_ = fn_.I.values
    Im_ = fnm_.I.values
    R_ = fh_.B / fn_.B
    Rm_ = fhm_.B / fnm_.B
    # plot
    fig, ax = plt.subplots()
    # regressed data
    ax.errorbar(df_NMR.I, df_NMR.B_ratio, yerr=df_NMR.sigma_B_ratio,
                c='blue', fmt='o', ls='none', ms=4, zorder=101, capsize=2,
                label='Regressed Data')
    # femm
    ax.plot(I_, R_, color='black', linestyle='--', linewidth=2, zorder=99,
            label='FEMM Calculation')
    ax.scatter(Im_, Rm_, c='green', s=15, marker='*', zorder=100,
        label='FEMM @ Set Points')
    # formatting
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('Magnet Current [A]')
    ax.set_ylabel(r'$|B|_\mathrm{Hall} / |B|_\mathrm{NMR}$')
    ax.set_title(r'$B_\mathrm{ratio}$ vs. $I$ Regressed Data')
    ax.legend().set_zorder(102)
    if not plotfile is None:
        fig.savefig(plotfile+'.pdf')
        fig.savefig(plotfile+'.png')
    return fig, ax

def plot_data_vs_FEMM(df_NMR_, df_Hall_, femm_nmr_meas, femm_hall_meas,
                      I_cut_NMR=120, I_cut_Hall=-1000,
                      pfile_NMR=None, pfile_Hall=None):
    # plots FEMM / Data
    # query info dfs
    df_NMR = df_NMR_.query(f'I >= {I_cut_NMR}').copy()
    df_Hall = df_Hall_.query(f'I >= {I_cut_Hall}').copy()
    # cut on current
    fnm_ = femm_nmr_meas.query(f'I >= {I_cut_NMR}').copy()
    fhm_ = femm_hall_meas.query(f'I >= {I_cut_Hall}').copy()
    Inm_ = fnm_.I.values
    Ihm_ = fhm_.I.values
    Bnm_ = fnm_.B.values
    Bhm_ = fhm_.B.values
    # loop through dataframes
    # nmr
    delta_nmrs = []
    for row in df_NMR.itertuples():
        ix = np.argmin(np.abs(row.I-Inm_))
        #delta_nmrs.append(row.B_reg - Bnm_[ix])
        delta_nmrs.append(Bnm_[ix] / row.B_reg)
    delta_nmrs = np.array(delta_nmrs)
    # hall
    delta_halls = []
    for row in df_Hall.itertuples():
        ix = np.argmin(np.abs(row.I-Ihm_))
        # delta_halls.append(row.B_reg - Bhm_[ix])
        delta_halls.append(Bhm_[ix] / row.B_reg)
    delta_halls = np.array(delta_halls)
    # uncertainties
    sigma_n = delta_nmrs * df_NMR.sigma_B_reg / df_NMR.B_reg
    sigma_h = delta_halls * df_Hall.sigma_B_reg / df_Hall.B_reg
    # plot
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    # nmr
    ax1.errorbar(df_NMR.I, delta_nmrs, yerr=sigma_n, c='blue',
                 fmt='o', ls=None, ms=4, zorder=101, capsize=2)
    ax2.errorbar(df_Hall.I, delta_halls, yerr=sigma_h, c='red',
                 fmt='o', ls=None, ms=4, zorder=101, capsize=2)
    # formatting
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax1.set_xlabel('Magnet Current [A]')
    ax2.set_xlabel('Magnet Current [A]')
    ax1.set_ylabel(r'$|B|_\mathrm{FEMM}/|B|_\mathrm{Data} $')
    ax2.set_ylabel(r'$|B|_\mathrm{FEMM}/|B|_\mathrm{Data} $')
    ax1.set_title('Comparison: FEMM / NMR')
    ax2.set_title('Comparison: FEMM / Hall probe')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    if not pfile_NMR is None:
        fig1.savefig(pfile_NMR+'.pdf')
        fig1.savefig(pfile_NMR+'.png')
    if not pfile_Hall is None:
        fig2.savefig(pfile_Hall+'.pdf')
        fig2.savefig(pfile_Hall+'.png')
    return fig1, ax1, fig2, ax2

def fit_B_vs_I(ndeg, df_meas, name='NMR', ycol='B_reg', yerr='sigma_B_reg',
               method='POLYFIT', I_min=-1000, fitcolor='red', datacolor='blue',
               fig=None, axs=None, plotfile=None, pklfile=None):
    # copy dataframes and limit current
    df_ = df_meas.copy()
    df_ = df_.query(f'I >= {I_min}')
    # remove run 11! FIXME!
    m = df_.index == 11
    df_ = df_[~m].copy()
    Is_fine = np.linspace(df_.I.min(), df_.I.max(), 200)
    # set up noise for least-squares fit
    ystd = df_[yerr].values
    # TESTING ONLY
    #ystd = None
    # run modeling (polyfit or interpolation)
    # check method
    if method == 'POLYFIT':
        print(f"Running {ndeg} Degree POLYFIT for {name}")
        # setup lmfit model
        model = lm.Model(ndeg_poly1d, independent_vars=['x'])
        params = lm.Parameters()
        for i in range(ndeg+1):
            params.add(f'C_{i}', value=0, vary=True)
        # fit
        result = model.fit(df_[ycol].values, x=df_.I.values,
                           params=params, weights=1/ystd, scale_covar=False)
        # calculate B for full dataset
        B_full = ndeg_poly1d(Is_fine, **result.params)
        # calculate residual (data - fit)
        res = df_[ycol].values - result.best_fit
        # other formatting
        fit_name = 'Polynomial Fit'
        ylab = 'Fit'
        datalab = ylab
        # label for fit
        label = '\n'
        label += (rf'$\underline{{\mathrm{{Degree\ {ndeg}\ Polynomial}}}}$'+
                 '\n')
        label_coeffs = []
        for i in range(ndeg+1):
            pv = result.params[f'C_{i}'].value
            label_coeffs.append(rf'$C_{{{i}}} = {pv:0.3E}$'+'\n')
        label += (''.join(label_coeffs)+'\n'+
              rf'$\chi^2_\mathrm{{red.}} = {result.redchi:0.2f}$'+'\n')

    elif method == 'INTERP_LIN':
        print(f"Running INTERP_LIN for {name}")
        # set up interpolation
        interp_func = interp1d(df_.I.values, df_[ycol].values,
                               kind='linear', fill_value='extrapolate')
        # calculate B for meas and full dfs
        B_full = interp_func(Is_fine)
        B_meas = interp_func(df_.I.values)
        # residuals
        res = df_[ycol].values - B_meas
        # other formatting
        fit_name = 'Linear Interpolation'
        ylab = 'Interpolation'
        datalab = 'Interp.'
        # label for fit
        label = f'Linear Interpolation'
        # return "result"
        result = interp_func

    elif method == 'INTERP_QUAD':
        print(f"Running INTERP_QUAD for {name}")
        # set up interpolation
        interp_func = interp1d(df_.I.values, df_[ycol].values,
                               kind='quadratic', fill_value='extrapolate')
        # calculate B for meas and full dfs
        B_full = interp_func(Is_fine)
        B_meas = interp_func(df_.I.values)
        # residuals
        res = df_[ycol].values - B_meas
        # other formatting
        fit_name = 'Quadratic Interpolation'
        ylab = 'Interpolation'
        datalab = 'Interp.'
        # label for fit
        label = f'Quadratic Interpolation'
        # return "result"
        result = interp_func

    elif method == 'INTERP_CUBIC':
        print(f"Running INTERP_CUBIC for {name}")
        # set up interpolation
        interp_func = interp1d(df_.I.values, df_[ycol].values, kind='cubic',
                               fill_value='extrapolate')
        # calculate B for meas and full dfs
        B_full = interp_func(Is_fine)
        B_meas = interp_func(df_.I.values)
        # residuals
        res = df_[ycol].values - B_meas
        # other formatting
        fit_name = 'Cubic Interpolation'
        ylab = 'Interpolation'
        datalab = 'Interp.'
        # label for fit
        label = f'Cubic Interpolation'
        # return "result"
        result = interp_func

    else:
        raise NotImplementedError
    # saving fit result function
    if not pklfile is None:
        pkl.dump(result, open(pklfile, 'wb'))
    # plot
    # set up figure with two axes
    if fig is None:
        fig = plt.figure()
        ax1 = fig.add_axes((0.1, 0.31, 0.8, 0.6))
        ax2 = fig.add_axes((0.1, 0.08, 0.8, 0.2))#, sharex=ax1)
    else:
        ax1, ax2 = axs
    # plot data and fit
    # data
    label_data = f'Regressed\nData ({datalab})'
    ax1.errorbar(df_.I.values, df_[ycol].values, yerr=ystd, c=datacolor,
                 fmt='x', ls='none', ms=6, zorder=100, capsize=3,
                 label=label_data)
    # fit
    ax1.plot(Is_fine, B_full, linewidth=1, color=fitcolor,
             zorder=99, label=label)
    # calculate ylimit for ax2
    yl = 1.2*(np.max(np.abs(res)) + np.max(ystd))
    # plot residual
    # zero-line
    xmin = np.min(df_.I.values)
    xmax = np.max(df_.I.values)
    ax2.plot([xmin, xmax], [0, 0], '--', color='black', linewidth=1,
             zorder=98)
    # residual
    ax2.errorbar(df_.I.values, res, yerr=ystd, fmt='x', ls='none', ms=6,
                 c=datacolor, capsize=3, zorder=100)
    # formatting
    # set ylimits
    ax2.set_ylim([-yl, yl])
    # remove ticklabels for ax1 xaxis
    ax1.set_xticklabels([])
    # axis labels
    ax2.set_xlabel('Magnet Current [A]')
    ax2.set_ylabel(f'(Data - {ylab}) [T]')
    ax1.set_ylabel(r'$|B|$')
    # force consistent x axis range for ax1 and ax2
    tmin = np.min(df_.I.values) - 10
    tmax = np.max(df_.I.values) + 10
    ax1.set_xlim([tmin, tmax])
    ax2.set_xlim([tmin, tmax])
    # turn on legend
    ax1.legend(fontsize=13).set_zorder(101)
    # add title
    fig.suptitle(f'Regressed Data -- B vs. I Modeling:'
                 f' {fit_name} for {name} Probe')
    # minor ticks
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    # inward ticks and ticks on right and top
    ax1.tick_params(which='both', direction='in', top=True, right=True,
                    bottom=True)
    ax2.tick_params(which='both', direction='in', top=True, right=True)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # save file
    if not plotfile is None:
        fig.savefig(plotfile+'.pdf')
        fig.savefig(plotfile+'.png')

    return result, fig, ax1, ax2

def plot_proc_and_fit(proc_ramp, result_B_vs_I, ycol='NMR [T]', name='NMR',
                      ylab='Interpolation', fitname='Cubic Interpolation',
                      pfile_scat=None, pfile_hist=None, tempcorrect=True):
    # calculate interpolated data
    df_ = proc_ramp.sort_values(by=['Magnet Current [A]']).copy()
    Is = df_['Magnet Current [A]'].values
    Is_fine = np.linspace(np.min(Is), np.max(Is), 200)
    Bs = df_[ycol].values
    Ts = df_['Yoke (center magnet)'].values
    T0 = 15. # deg C
    # check if regress or interp
    if type(result_B_vs_I) == lm.model.ModelResult:
        # regressed
        # TESTING TEMP ADDITIONS
        if tempcorrect:
            a = -1.5e-4#-1e-4
            Bs_fit = (result_B_vs_I.eval(Is) +
                      (a * (Is-120)/(280-120)) * (Ts-T0))
            Is_fine = Is
            Bs_fit_fine = Bs_fit
            titlesuff = ' (with Temperature Corrections)'
        # standard check -- no corrections
        else:
            Bs_fit = result_B_vs_I.eval(x=Is)
            Bs_fit_fine = result_B_vs_I.eval(xs=Is_fine)
            titlesuff = ''
    else:
        # interpolated
        # TESTING TEMP ADDITIONS
        if tempcorrect:
            a = -1.5e-4#-1e-4
            # Bs_fit = result_B_vs_I(Is) + (a * (Is-120)/(280-120)) * (Ts-T0)
            Bs_fit = result_B_vs_I(Is) + (a * (Is/280)**2) * (Ts-T0)
            #
            '''
            a0 = -2e-5#-1e-3 # chiller works
            a1 = -8e-5 # chiller out
            I_cut = 190.
            #Bs_fit = result_B_vs_I(Is) * (1 + a*(Ts-T0))
            #m0 = Is < I_cut
            #m1 = Is >= I_cut
            Bs_0 = result_B_vs_I(Is[m0])*(1+a0*(Ts[m0]-T0))
            Bs_1 = result_B_vs_I(Is[m1])*(1+a1*(Ts[m1]-T0))
            Bs_fit = np.concatenate([Bs_0, Bs_1])
            '''
            #Bs_fit_fine = result_B_vs_I(Is_fine) * (1 + a*(Ts-T0))
            # TEST
            Is_fine = Is
            Bs_fit_fine = Bs_fit
            titlesuff = ' (with Temperature Corrections)'
        # standard check - no corrections
        else:
            Bs_fit = result_B_vs_I(Is)
            Bs_fit_fine = result_B_vs_I(Is_fine)
            titlesuff = ''

    # residuals
    Bs_res = Bs - Bs_fit
    # errors
    if ycol=='NMR [T]':
        ystd = 5e-6 * np.ones_like(Bs)
    else:
        ystd = 3e-5 * np.ones_like(Bs)

    #if fig is None:
    fig = plt.figure() # residuals plot
    ax1 = fig.add_axes((0.1, 0.31, 0.8, 0.6))
    ax2 = fig.add_axes((0.1, 0.08, 0.8, 0.2))#, sharex=ax1)
    fig_2, ax_2 = plt.subplots() # residuals histogram

    # residuals plot
    label_data = f'Pre-processed Data'
    #label_data = f'Data ({datalab})'
    datacolor='black'
    fitcolor='green'
    # errorbar
    # ax1.errorbar(Is, Bs, yerr=ystd, c=datacolor,
    #              fmt='x', ls='none', ms=6, zorder=100, capsize=3,
    #              label=label_data)
    # scatter
    ax1.scatter(Is, Bs, c=datacolor, s=1, zorder=100, label=label_data)
    # fit
    ax1.plot(Is_fine, Bs_fit_fine, linewidth=1, color=fitcolor,
             zorder=99, label=fitname)

    # calculate ylimit for ax2
    # yl = 1.2*(np.max(np.abs(res)) + ystd[0])
    yl = 1.2*(np.max(np.abs(Bs_res)) + np.max(ystd))
    # plot residual
    # zero-line
    xmin = np.min(Is)
    xmax = np.max(Is)
    ax2.plot([xmin, xmax], [0, 0], '--', color='black', linewidth=1,
             zorder=98)
    # residual
    # errorbar
    # ax2.errorbar(Is, Bs_res, yerr=ystd, fmt='x', ls='none', ms=6,
    #              c=datacolor, capsize=3, zorder=100)
    # scatter
    ax2.scatter(Is, Bs_res, c=datacolor, s=1, zorder=100)
    # formatting
    # set ylimits
    #ax1.set_ylim([-0.25, 1.5])
    ax2.set_ylim([-yl, yl])
    # remove ticklabels for ax1 xaxis
    ax1.set_xticklabels([])
    # axis labels
    ax2.set_xlabel('Magnet Current [A]')
    ax2.set_ylabel(f'(Data - {ylab}) [T]')
    ax1.set_ylabel(r'$|B|$')
    # force consistent x axis range for ax1 and ax2
    tmin = np.min(Is) - 10
    tmax = np.max(Is) + 10
    ax1.set_xlim([tmin, tmax])
    ax2.set_xlim([tmin, tmax])
    # turn on legend
    ax1.legend(fontsize=13).set_zorder(101)
    # add title
    fig.suptitle(r'$B$ vs. $I$ on Pre-processed Data:'+
                 f'\n{fitname} for {name} Probe{titlesuff}')
    # minor ticks
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    # inward ticks and ticks on right and top
    ax1.tick_params(which='both', direction='in', top=True, right=True,
                    bottom=True)
    ax2.tick_params(which='both', direction='in', top=True, right=True)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # save scatter plot to file
    if not pfile_scat is None:
        fig.savefig(pfile_scat+'.pdf')
        fig.savefig(pfile_scat+'.png')
    # HISTOGRAM
    rmax = 1.2 * np.max(np.abs(Bs_res))
    nbins=200#100
    bins = np.linspace(-rmax, rmax, nbins)
    ax_2.hist(Bs_res, bins=bins, histtype='step', linewidth=2,
              label=get_label(Bs_res, bins))
    # formatting
    # minor ticks
    ax_2.xaxis.set_minor_locator(AutoMinorLocator())
    ax_2.yaxis.set_minor_locator(AutoMinorLocator())
    # inward ticks
    ax_2.tick_params(which='both', direction='in',)# top=True, right=True)
    # labels
    # ax_2.set_xlabel(f'(Data ({ycol}) - {ylab}) [T]')
    ax_2.set_xlabel(f'(Data - {ylab}) [T]')
    ax_2.set_ylabel('Count')
    fig_2.suptitle(f'Fit Residuals on Pre-processed Data:'+
                   f' {name} Probe{titlesuff}')
    ax_2.legend()
    if not pfile_hist is None:
        fig_2.savefig(pfile_hist+'.pdf')
        fig_2.savefig(pfile_hist+'.png')

    return Is, Bs, Bs_fit, Bs_res, fig, ax1, ax2, fig_2, ax_2

# SCRATCH CALCULATION OF ERROR FOR B_RATIO
#np.cov(Halls[7:],NMRs[7:])
# Rs = Halls[7:]/NMRs[7:]
# cov = np.cov(Halls[7:],NMRs[7:])
# sigma_R = Rs * (cov[0,0]/Halls[7:]**2 + cov[1,1]/NMRs[7:]**2 - 2 * cov[0,1] /
#                 (NMRs[7:] * Halls[7:]))**(1/2)
# Rs, sigma_R, cov, Hall_errs, NMR_errs
## ATTEMPT 2
# sigma_R = Rs * (Hall_errs[7:]/Halls[7:]**2 + NMR_errs[7:]/NMRs[7:]**2)**(1/2)
# Rs, sigma_R


if __name__=='__main__':
    print('Running script: B_vs_I_no_temp.py')
    time0 = datetime.now()
    # more specific plot directory
    pdir = plotdir+'final_results/B_vs_I/'
    # load info df
    df_info = pd.read_pickle(pklinfo)
    # get NMR and Hall functions (using factory funcs)
    temp = create_all_funcs(pklfit_temp_nmr, pklfit_temp_hall,
                            pklfit_temp_hall_nmr)
    get_NMR, get_Hall, get_Hall_NMR = temp
    # generate regressed B values
    #_ = regressed_Bs(get_NMR, get_Hall, get_Hall_NMR, df_info,
    #                 pklinfo_hall_regress, pklinfo_nmr_regress, T0=15)
    #df_NMR, df_Hall = _
    # load from pickle
    df_NMR = pd.read_pickle(pklinfo_nmr_regress)
    df_Hall = pd.read_pickle(pklinfo_hall_regress)

    # load FEMM data for ratio plot
    femm_hall, femm_hall_meas = load_data(femmfile_75_Hall, Hall_currents)
    femm_nmr, femm_nmr_meas = load_data(femmfile_75_NMR, NMR_currents)
    I_min_NMR = femm_nmr_meas.query('I>120').I.min()

    # plots
    fig, ax = plot_B_vs_I(df_NMR, df_Hall, pdir+'basic_B_vs_I')
    fig, ax = plot_Hall_compare(df_Hall, I_cut=120.,
                                plotfile=pdir+'compare_Hall_regression')
    fig, ax = plot_R_vs_I(df_NMR, femm_nmr, femm_nmr_meas, femm_hall,
                          femm_hall_meas, I_cut=I_min_NMR,
                          plotfile=pdir+'Bratio_vs_I')
    temp = plot_data_vs_FEMM(df_NMR, df_Hall, femm_nmr_meas, femm_hall_meas,
                             I_cut_NMR=120, I_cut_Hall=30,
                             pfile_NMR=pdir+'FEMM_div_NMR',
                             pfile_Hall=pdir+'FEMM_div_Hall')
    fig1, ax1, fig2, ax2 = temp
    # FITS
    #I_min_Hall = df_Hall.query('I>120').I.min()
    I_min_Hall = -1000
    I_min_NMR = df_NMR.query('I>120').I.min()
    # polyfit degrees
    ndeg = 6#8#6#10
    # savefile
    pfile = pdir+'B_vs_I_{0}_{1}'
    # INTERP ONLY PLOTS
    #for I in ['LIN', 'QUAD', 'CUBIC']:
    # TESTING
    for I in ['CUBIC']:
        i = I.lower()
        if I == 'CUBIC':
            pNMR = pkl_interp_fcn_NMR
            pHall = pkl_interp_fcn_Hall
        else:
            pNMR = None
            pHall = None
        # nmr
        temp = fit_B_vs_I(ndeg, df_NMR, name='NMR', ycol='B_reg',
                          yerr='sigma_B_reg', I_min=I_min_NMR, fitcolor='green',
                          datacolor='black', method=f'INTERP_{I}',
                          plotfile=pfile.format('NMR', f'interp_{i}'),
                          pklfile=pNMR)
        result_nmr_interp, fig, ax1, ax2 = temp
        # hall probe
        temp = fit_B_vs_I(ndeg, df_Hall, name='Hall', ycol='B_reg',
                          yerr='sigma_B_reg', I_min=I_min_Hall , fitcolor='green',
                          datacolor='black', method=f'INTERP_{I}',
                          plotfile=pfile.format('Hall', f'interp_{i}'),
                          pklfile=pHall)
        result_hall_interp, fig, ax1, ax2 = temp
    # REGRESSION ONLY PLOTS
    '''
    # for I in ['LIN', 'QUAD', 'CUBIC']:
    #     i = I.lower()
    # nmr
    temp = fit_B_vs_I(ndeg, df_NMR, name='NMR', ycol='B_reg',
                      yerr='sigma_B_reg', I_min=I_min_NMR, fitcolor='red',
                      datacolor='blue', method='POLYFIT',
                      plotfile=pfile.format('NMR', f'polyfit_deg{ndeg}'))
    result_nmr, fig, ax1, ax2 = temp
    # hall probe
    temp = fit_B_vs_I(ndeg, df_Hall, name='Hall', ycol='B_reg',
                      yerr='sigma_B_reg', I_min=I_min_Hall , fitcolor='red',
                      datacolor='blue', method=f'POLYFIT',
                      plotfile=pfile.format('Hall', f'polyfit_deg{ndeg}'))
    result_hall, fig, ax1, ax2 = temp
    '''
    # fit result on pre-processed data
    # plot file template
    pfile2 = pdir+'preproc_{0}_{1}_{2}'
    # load processed ramp data
    proc_ramp = pd.read_pickle(pklproc_ramp)
    nmrs = df_info[df_info.NMR].index
    nmrs = nmrs[nmrs != 11]
    proc_ramp_nmr = proc_ramp[np.isin(proc_ramp.run_num, nmrs)]
    halls = (proc_ramp.run_num != 11)
    proc_ramp_hall = proc_ramp[halls]
    # NMR, cubic interp
    # no temp
    _ = plot_proc_and_fit(proc_ramp_nmr, result_nmr_interp, ycol='NMR [T]',
                          name='NMR', ylab='Interpolation',
                          fitname='Cubic Interpolation',
                          pfile_scat=pfile2.format('NMR', 'B_vs_I',
                                                   'interp_cubic'),
                          pfile_hist=pfile2.format('NMR', 'Hist',
                                                   'interp_cubic'),
                          tempcorrect=False)
    Is, Bs, Bs_fit, Bs_res, fig, ax1, ax2, fig_2, ax_2 = _
    # with temp
    _ = plot_proc_and_fit(proc_ramp_nmr, result_nmr_interp, ycol='NMR [T]',
                          name='NMR', ylab='Interpolation',
                          fitname='Cubic Interpolation',
                          pfile_scat=pfile2.format('NMR_tempcorr', 'B_vs_I',
                                                   'interp_cubic'),
                          pfile_hist=pfile2.format('NMR_tempcorr', 'Hist',
                                                   'interp_cubic'),
                          tempcorrect=True)
    Is, Bs, Bs_fit, Bs_res, fig, ax1, ax2, fig_2, ax_2 = _
    # Hall, cubic interp
    # no temp
    _ = plot_proc_and_fit(proc_ramp_hall, result_hall_interp,
                          ycol=f'{probe}_Cal_Bmag', name='Hall',
                          ylab='Interpolation', fitname='Cubic Interpolation',
                          pfile_scat=pfile2.format('Hall', 'B_vs_I',
                                                   'interp_cubic'),
                          pfile_hist=pfile2.format('Hall', 'Hist',
                                                   'interp_cubic'),
                          tempcorrect=False)
    Is, Bs, Bs_fit, Bs_res, fig, ax1, ax2, fig_2, ax_2 = _
    # with temp
    _ = plot_proc_and_fit(proc_ramp_hall, result_hall_interp,
                          ycol=f'{probe}_Cal_Bmag', name='Hall',
                          ylab='Interpolation', fitname='Cubic Interpolation',
                          pfile_scat=pfile2.format('Hall_tempcorr', 'B_vs_I',
                                                   'interp_cubic'),
                          pfile_hist=pfile2.format('Hall_tempcorr', 'Hist',
                                                   'interp_cubic'),
                          tempcorrect=True)
    Is, Bs, Bs_fit, Bs_res, fig, ax1, ax2, fig_2, ax_2 = _

    # test output
    # print('NMR df:')
    # print(df_NMR)
    # print('Hall df:')
    # print(df_Hall)
    # fit reports?

    timef = datetime.now()
    print(f'Runtime: {timef-time0} [H:MM:SS])\n')
    plt.show()
