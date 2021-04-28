import numpy as np
import pandas as pd
import lmfit as lm
import pickle as pkl
from sys import stdout
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import dates as mdates
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
# local imports
from model_funcs import mod_lin, mod_lin_alt
from configs import (
    pklfit_temp_hall,
    pklfit_temp_hall_nmr,
    pklfit_temp_nmr,
    pklinfo,
    pklproc,
    plotdir,
    probe,
)
from plotting import config_plots
config_plots()


def load_preprocessed_pkls(pklinfo, pklproc):
    # load results from preprocess_data.py
    df_info = pd.read_pickle(pklinfo)
    df = pd.read_pickle(pklproc)
    return df_info, df

def linear_temperature_regression(run_num, df, plotfile, xcol, ycol, ystd,
                                  ystd_sf, force_decreasing):
    # query preprocessed data to get run
    df_ = df.query(f'run_num == {run_num}').copy()
    # create an array for weights if float supplied
    if type(ystd) != np.ndarray:
        ystd = ystd*np.ones(len(df_))
    # setup lmfit model
    # y = A + B * x
    model = lm.Model(mod_lin, independent_vars=['x'])
    params = lm.Parameters()
    params.add('A', value=0, vary=True)
    params.add('B', value=0, vary=True)
    # fit
    result = model.fit(df_[ycol].values, x=df_[xcol].values,
                       params=params, weights=1/ystd, scale_covar=False)
    # check if not decreasing and rerun
    if force_decreasing and (result.params['B'].value > 0.):
        params['B'].vary = False
        result = model.fit(df_[ycol].values, x=df_[xcol].values,
                           params=params, weights=1/ystd, scale_covar=False)
        # label for fit
        label= (r'$\underline{y = A + B x}$'+'\n\n'+
                rf'$A = {result.params["A"].value:0.4f}$'+'\n'+
                rf'$B = {result.params["B"].value:0.1f}$'+' (fixed)\n'+
                rf'$\chi^2_\mathrm{{red.}} = {result.redchi:0.2f}$'+'\n\n')
    else:
        # label for fit
        label= (r'$\underline{y = A + B x}$'+'\n\n'+
                rf'$A = {result.params["A"].value:0.4f}$'+'\n'+
                rf'$B = {result.params["B"].value:0.3E}$'+'\n'+
                rf'$\chi^2_\mathrm{{red.}} = {result.redchi:0.2f}$'+'\n\n')
    # plot

    # set up figure with two axes
    config_plots()
    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.31, 0.7, 0.6))
    ax2 = fig.add_axes((0.1, 0.08, 0.7, 0.2))
    # colorbar axis
    cb_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # plot data and fit
    # data
    # with errorbars
    if ystd_sf == 1:
        label_data = 'Data'
    else:
        label_data = r'Data (error $\times$'+f'{ystd_sf})'
    ax1.errorbar(df_[xcol].values, df_[ycol].values, yerr=ystd_sf*ystd,
                 fmt='o', ls='none', ms=2, zorder=100, label=label_data)
    # scatter with color
    #sc = ax1.scatter(df_[xcol].values, df_[ycol].values, c=df_.index, s=1,
    #                 zorder=101)
    sc = ax1.scatter(df_[xcol].values, df_[ycol].values, c=df_.run_hours, s=1,
                     zorder=101)
    # fit
    ax1.plot(df_[xcol].values, result.best_fit, linewidth=2, color='red',
             zorder=99, label=label)
    # calculate residual (data - fit)
    res = df_[ycol].values - result.best_fit
    # calculate ylimit for ax2
    yl = 1.1*(np.max(np.abs(res)) + ystd_sf*ystd[0])
    # plot residual
    # zero-line
    xmin = np.min(df_[xcol].values)
    xmax = np.max(df_[xcol].values)
    ax2.plot([xmin, xmax], [0, 0], 'k--', linewidth=2, zorder=98)
    # residual
    ax2.errorbar(df_[xcol].values, res, yerr=ystd_sf*ystd, fmt='o', ls='none',
                 ms=2, zorder=99)
    #ax2.scatter(df_[xcol].values, res, c=df_.index, s=1, zorder=101)
    ax2.scatter(df_[xcol].values, res, c=df_.run_hours, s=1, zorder=101)
    # colorbar for ax1
    cb = fig.colorbar(sc, cax=cb_ax)
    cb.set_label('Time [Hours]')
    ## WITH DATETIME
    #cb.set_t
    # print(f'vmin = {sc.colorbar.vmin}')
    # print(f'vmax = {sc.colorbar.vmax}')
    # # change colobar ticks labels and locators
    # cb.set_ticks([sc.colorbar.vmin + t*(sc.colorbar.vmax-sc.colorbar.vmin)
    #              for t in cb.ax.get_yticks()])
    # cbtls = [mdates.datetime.datetime.fromtimestamp((sc.colorbar.vmin +
    #          t*(sc.colorbar.vmax-sc.colorbar.vmin))*1e-9).strftime('%c')
    #          for t in cb.ax.get_yticks()]
    # cb.set_ticklabels(cbtls)
    #cb.ax.set_yticklabels(df_.index.strftime('%m-%d %H:%M'))
    # formatting
    # kludge for NMR or Hall
    if ycol == 'NMR [T]':
        ylabel1 = r'$|B|_\mathrm{NMR}$ [T]'
        title_prefix = 'NMR'
    else:
        ylabel1 = r'$|B|_\mathrm{Hall}$ [T]'
        title_prefix = 'Hall Probe'
    # set ylimit ax2
    ax2.set_ylim([-yl, yl])
    # remove ticklabels for ax1 xaxis
    ax1.set_xticklabels([])
    # axis labels
    ax2.set_xlabel(r'Yoke (center magnet) Temp. [$^{\circ}$C]')
    ax2.set_ylabel('(Data - Fit) [T]')
    ax1.set_ylabel(ylabel1)
    # force consistent x axis range for ax1 and ax2
    tmin = np.min(df_[xcol].values)
    tmax = np.max(df_[xcol].values)
    range_t = tmax-tmin
    ax1.set_xlim([tmin-0.1*range_t, tmax+0.1*range_t])
    ax2.set_xlim([tmin-0.1*range_t, tmax+0.1*range_t])
    # turn on legend
    ax1.legend().set_zorder(102)
    # add title
    fig.suptitle(f'{title_prefix} vs. Temperature Regression: Linear Model\n'+
                 f'Run Index {run_num}')
    # minor ticks
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    # inward ticks and ticks on right and top
    ax1.tick_params(which='both', direction='in', top=True, right=True,
                    bottom=True)
    ax2.tick_params(which='both', direction='in', top=True, right=True)
    # save figure
    #fig.tight_layout()
    fig.savefig(plotfile+'.pdf')
    fig.savefig(plotfile+'.png')
    return result, fig, ax1, ax2

# NMR
def run_nmr_regression_all(df, df_info, pklfit_temp_nmr):
    # message
    print('Running linear NMR vs. temperature regression')
    # generate filenames
    pfiles = [(plotdir+f'final_results/nmr_temp_regress/nmr_run-{i}_lin_temp'+
               f'_regress_fit') for i in df_info.index]
    # get CPU information
    num_cpu = multiprocessing.cpu_count()
    nproc = min([num_cpu, len(df_info)])
    # parallel for loop
    temp = (Parallel(n_jobs=nproc)
            (delayed(linear_temperature_regression)
             (i, df, pf, xcol='Yoke (center magnet)', ycol='NMR [T]',
             ystd=1e-6, ystd_sf=1, force_decreasing=False) for i,pf in
             tqdm(zip(df_info.index, pfiles), file=stdout,desc='Run #',
                  total=len(df_info))
             )
            )
    # split output
    results = {i:j[0] for i,j in zip(df_info.index, temp)}
    # pickle results
    pkl.dump(results, open(pklfit_temp_nmr, "wb" ))
    return results

# Hall probes
# No NMR
def run_hall_regression_all(probe, df, df_info, pklfit_temp_hall):
    # message
    print('Running linear Hall vs. temperature regression (No NMR)')
    # generate filenames
    pfiles = [(plotdir+f'final_results/hall_temp_regress/hall_run-{i}_lin'+
               f'_temp_regress_fit') for i in df_info.index]
    # get CPU information
    num_cpu = multiprocessing.cpu_count()
    nproc = min([num_cpu, len(df_info)])
    # parallel for loop
    temp = (Parallel(n_jobs=nproc)
            (delayed(linear_temperature_regression)
             (i, df, pf, xcol='Yoke (center magnet)',
              ycol=f'{probe}_Cal_Bmag', ystd=3e-5, ystd_sf=1,
              force_decreasing=False) for i,pf in
             tqdm(zip(df_info.index, pfiles), file=stdout, desc='Run #',
                  total=len(df_info))
             )
            )
    # split output
    results = {i:j[0] for i,j in zip(df_info.index, temp)}
    # pickle results
    pkl.dump(results, open(pklfit_temp_hall, "wb" ))
    return results

# Use NMR results
def hall_regression_from_nmr(run_num, df, results_nmr, plotfile, xcol, ycol,
                             ystd, ystd_sf):
    # grab correct NMR result
    nmr_params = results_nmr[run_num].params
    # query preprocessed data to get run
    df_ = df.query(f'run_num == {run_num}').copy()
    # create an array for weights if float supplied
    if type(ystd) != np.ndarray:
        ystd = ystd*np.ones(len(df_))
    # setup lmfit model
    # y = C + (A + B * x), A and B fixed
    model = lm.Model(mod_lin_alt, independent_vars=['x'])
    params = lm.Parameters()
    params.add('A', value=nmr_params['A'].value, vary=False)
    params.add('B', value=nmr_params['B'].value, vary=False)
    params.add('C', value=0, vary=True)
    # fit
    result = model.fit(df_[ycol].values, x=df_[xcol].values,
                       params=params, weights=1/ystd, scale_covar=False)
    # plot
    # label for fit
    label= (r'$\underline{y = C + (A + B x)}$'+'\n\n'+
            rf'$A = {result.params["A"].value:0.4f}$'+' (fixed)\n'+
            rf'$B = {result.params["B"].value:0.3E}$'+' (fixed)\n'+
            rf'$C = {result.params["C"].value:0.4f}$'+' \n'+
            rf'$\chi^2_\mathrm{{red.}} = {result.redchi:0.2f}$'+'\n\n')
    # set up figure with two axes
    config_plots()
    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.31, 0.7, 0.6))
    ax2 = fig.add_axes((0.1, 0.08, 0.7, 0.2))
    # colorbar axis
    cb_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # plot data and fit
    # data
    # with errorbars
    if ystd_sf == 1:
        label_data = 'Data'
    else:
        label_data = r'Data (error $\times$'+f'{ystd_sf})'
    ax1.errorbar(df_[xcol].values, df_[ycol].values, yerr=ystd_sf*ystd,
                 fmt='o', ls='none', ms=2, zorder=100, label=label_data)
    # scatter with color
    # sc = ax1.scatter(df_[xcol].values, df_[ycol].values, c=df_.index, s=1,
    #                  zorder=101)
    sc = ax1.scatter(df_[xcol].values, df_[ycol].values, c=df_.run_hours, s=1,
                     zorder=101)
    # fit
    ax1.plot(df_[xcol].values, result.best_fit, linewidth=2, color='red',
             zorder=99, label=label)
    # calculate residual (data - fit)
    res = df_[ycol].values - result.best_fit
    # calculate ylimit for ax2
    yl = 1.1*(np.max(np.abs(res)) + ystd_sf*ystd[0])
    # plot residual
    # zero-line
    xmin = np.min(df_[xcol].values)
    xmax = np.max(df_[xcol].values)
    ax2.plot([xmin, xmax], [0, 0], 'k--', linewidth=2, zorder=98)
    # residual
    ax2.errorbar(df_[xcol].values, res, yerr=ystd_sf*ystd, fmt='o', ls='none',
                 ms=2, zorder=99)
    ax2.scatter(df_[xcol].values, res, c=df_.index, s=1, zorder=101)
    # colorbar for ax1
    cb = fig.colorbar(sc, cax=cb_ax)
    cb.set_label('Time [Hours]')
    #cb.ax.set_yticklabels(df_.index.strftime('%m-%d %H:%M'))
    # formatting
    ylabel1 = r'$|B|_\mathrm{Hall}$ [T]'
    title_prefix = 'Hall Probe'
    # set ylimit ax2
    ax2.set_ylim([-yl, yl])
    # remove ticklabels for ax1 xaxis
    ax1.set_xticklabels([])
    # axis labels
    ax2.set_xlabel(r'Yoke (center magnet) Temp. [$^{\circ}$C]')
    ax2.set_ylabel('(Data - Fit) [T]')
    ax1.set_ylabel(ylabel1)
    # force consistent x axis range for ax1 and ax2
    tmin = np.min(df_[xcol].values)
    tmax = np.max(df_[xcol].values)
    range_t = tmax-tmin
    ax1.set_xlim([tmin-0.1*range_t, tmax+0.1*range_t])
    ax2.set_xlim([tmin-0.1*range_t, tmax+0.1*range_t])
    # turn on legend
    ax1.legend().set_zorder(102)
    # add title
    fig.suptitle(f'{title_prefix} vs. Temperature Regression: Linear Model'+
                 f' (NMR) + Offset (Hall)\nRun Index {run_num}')
    # minor ticks
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    # inward ticks and ticks on right and top
    ax1.tick_params(which='both', direction='in', top=True, right=True,
                    bottom=True)
    ax2.tick_params(which='both', direction='in', top=True, right=True)
    # save figure
    #fig.tight_layout()
    fig.savefig(plotfile+'_with-NMR.pdf')
    fig.savefig(plotfile+'_with-NMR.png')
    return result, fig, ax1, ax2

def run_hall_nmr_regression_single(run_num, probe, plotfile, df, df_info,
                                   results_nmr):
    # check if NMR is in range and regressed
    if df_info.iloc[run_num].NMR:
        temp = hall_regression_from_nmr(run_num, df, results_nmr, plotfile,
                                        xcol='Yoke (center magnet)',
                                        ycol=f'{probe}_Cal_Bmag',
                                        ystd=3e-5, ystd_sf=1)
    # otherwise run the same regression as for NMR
    else:
        temp = linear_temperature_regression(run_num, df, plotfile,
                                             xcol='Yoke (center magnet)',
                                             ycol=f'{probe}_Cal_Bmag',
                                             ystd=3e-5, ystd_sf=1,
                                             force_decreasing=False)
                                             #force_decreasing=True)
    # split output
    result, fig, ax1, ax2 = temp
    return result, fig, ax1, ax2

def run_hall_nmr_regression_all(probe, df, df_info, results_nmr,
                                pklfit_temp_hall_nmr):
    # message
    print('Running linear Hall vs. temperature regression (With NMR)')
    # generate filenames
    pfiles = [(plotdir+f'final_results/hall_from_nmr_temp_regress/'+
               f'hall_with_nmr_run-{i}_lin_temp_regress_fit')
              for i in df_info.index]
    # get CPU information
    num_cpu = multiprocessing.cpu_count()
    nproc = min([num_cpu, len(df_info)])
    # parallel for loop
    temp = (Parallel(n_jobs=nproc)
            (delayed(run_hall_nmr_regression_single)
             (i, probe, pf, df, df_info, results_nmr) for i,pf in
             tqdm(zip(df_info.index, pfiles), file=stdout, desc='Run #',
                  total=len(df_info))
             )
            )
    # split output
    results = {i:j[0] for i,j in zip(df_info.index, temp)}
    # pickle results
    pkl.dump(results, open(pklfit_temp_hall_nmr, "wb" ))
    return results


if __name__=='__main__':
    print('Running script: process_data_temp_regress.py')
    time0 = datetime.now()
    # load preprocessed data
    df_info, df = load_preprocessed_pkls(pklinfo, pklproc)

    # NMR linear temperature regression
    results_nmr = run_nmr_regression_all(df, df_info, pklfit_temp_nmr)
    # load pickle
    #results_nmr = pkl.load(open(pklfit_temp_nmr, 'rb'))

    # Hall linear temperature regression
    # standard regression
    results_hall = run_hall_regression_all(probe, df, df_info,
                                           pklfit_temp_hall)
    # load pickle
    #results_hall = pkl.load(open(pklfit_temp_hall, 'rb'))

    # regression with NMR when available
    temp = run_hall_nmr_regression_all(probe, df, df_info, results_nmr,
                                       pklfit_temp_hall_nmr)
    results_hall_nmr = temp
    # load pickle
    #results_hall_nmr = pkl.load(open(pklfit_temp_hall_nmr, 'rb'))

    timef = datetime.now()
    print(f'Runtime: {timef-time0} [H:MM:SS])\n')
