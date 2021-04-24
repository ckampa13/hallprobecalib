import numpy as np
import pandas as pd
import lmfit as lm
import pickle as pkl
from sys import stdout
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
# local imports
from model_funcs import mod_lin
from configs import probe, plotdir, pklinfo, pklproc, pklinfo_regress, pklfit_temp_regress
from plotting import config_plots
config_plots()


def load_preprocessed_pkls(pklinfo, pklproc):
    # load results from preprocess_data.py
    df_info = pd.read_pickle(pklinfo)
    df = pd.read_pickle(pklproc)
    return df_info, df

def linear_temperature_regression(run_num, df, plotfile, xcol='Yoke (center magnet)', ycol='NMR [T]', ystd=5e-6, ystd_sf=1):
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
    # plot
    # label for fit
    label=r'$\underline{y = A + B x}$'+'\n\n'\
          +rf'$A = {result.params["A"].value:0.2f}$'+'\n'\
          +rf'$B = {result.params["B"].value:0.3E}$'+'\n'\
          +rf'$\chi^2_\mathrm{{red.}} = {result.redchi:0.2f}$'+'\n\n'
    # set up figure with two axes
    config_plots()
    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.31, 0.7, 0.6))
    ax2 = fig.add_axes((0.1, 0.08, 0.7, 0.2))
    cb_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # plot data and fit
    # data
    ax1.errorbar(df_[xcol].values, df_[ycol].values, yerr=ystd_sf*ystd, fmt='o', ls='none', ms=2, label=r'Data (error $\times$'+f'{ystd_sf})', zorder=100)
    #sc = ax1.scatter(df_[xcol].values, df_[ycol].values, c=df_['run_hours'], s=1, zorder=101)
    sc = ax1.scatter(df_[xcol].values, df_[ycol].values, c=df_.index, s=1, zorder=101)
    # fit
    ax1.plot(df_[xcol].values, result.best_fit, linewidth=2, color='red', label=label, zorder=99)
    # calculate residual (data - fit)
    res = df_[ycol].values - result.best_fit
    # calculate ylimit for ax2
    yl = 1.1*(np.max(np.abs(res)) + ystd_sf*ystd[0])
    # plot residual
    # zero-line
    ax2.plot([np.min(df_[xcol].values), np.max(df_[xcol].values)], [0, 0], 'k--', linewidth=2)
    # residual
    ax2.errorbar(df_[xcol].values, res, yerr=ystd_sf*ystd, fmt='o', ls='none', ms=2, zorder=99)
    ax2.scatter(df_[xcol].values, res, c=df_.index, s=1, zorder=101)
    cb = fig.colorbar(sc, cax=cb_ax)
    cb.ax.set_yticklabels(df_.index.strftime('%m-%d %H:%M'))
    # formatting
    # set ylimit ax2
    ax2.set_ylim([-yl, yl])
    # remove ticklabels for ax1 xaxis
    ax1.set_xticklabels([])
    # axis labels
    ax2.set_xlabel(r'Yoke (center magnet) Temp. [$^{\circ}$C]')
    ax2.set_ylabel('(Data - Fit) [T]')
    ax1.set_ylabel(r'$|B|_\mathrm{NMR}$ [T]')
    # force consistent x axis range for ax1 and ax2
    tmin = np.min(df_[xcol].values)
    tmax = np.max(df_[xcol].values)
    range_t = tmax-tmin
    ax1.set_xlim([tmin-0.1*range_t, tmax+0.1*range_t])
    ax2.set_xlim([tmin-0.1*range_t, tmax+0.1*range_t])
    # turn on legend
    ax1.legend().set_zorder(102)
    # add title
    fig.suptitle(f'NMR vs. Temperature Regression: Linear Model\nRun Index {run_num}')
    # minor ticks
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    # inward ticks and ticks on right and top
    ax1.tick_params(which='both', direction='in', top=True, right=True, bottom=True)
    ax2.tick_params(which='both', direction='in', top=True, right=True)
    # save figure
    fig.savefig(plotfile+'.pdf')
    fig.savefig(plotfile+'.png')
    return result, fig, ax1, ax2

def run_regression_all(df, df_info, pklinfo_regress, pklfit_temp_regress):
    # message
    print('Running linear NMR vs. temperature regression')
    # generate filenames
    pfiles = [plotdir+f'final_results/nmr_temp_regress/run-{i}_lin_temp_regress_fit' for i in df_info.index]
    # get CPU information
    num_cpu = multiprocessing.cpu_count()
    nproc = min([num_cpu, len(df_info)])
    # parallel for loop
    processed_tuples = Parallel(n_jobs=nproc)(delayed(linear_temperature_regression)(i, df, pf, xcol='Yoke (center magnet)', ycol='NMR [T]', ystd=1e-6, ystd_sf=1) for i,pf in tqdm(zip(df_info.index, pfiles), file=stdout, desc='Run #', total=len(df_info)))
    # split output
    results = {i:j[0] for i,j in zip(df_info.index, processed_tuples)}
    # pickle results
    pkl.dump(results, open(pklfit_temp_regress, "wb" ))
    return results


if __name__=='__main__':
    print('Running script: process_data_temp_regress.py')
    time0 = datetime.now()
    # load preprocessed data
    df_info, df = load_preprocessed_pkls(pklinfo, pklproc)
    # test 1 NMR temperature regress
    #run_num = 0
    #pf = plotdir+f'final_results/nmr_temp_regress/run-{run_num}_lin_temp_regress_fit'
    #linear_temperature_regression(run_num, df, pf, xcol='Yoke (center magnet)', ycol='NMR [T]', ystd=5e-6, ystd_sf=1)
    # NMR linear temperature regression
    results = run_regression_all(df, df_info, pklinfo_regress, pklfit_temp_regress)
    timef = datetime.now()
    print(f'Runtime: {timef-time0} [H:MM:SS])\n')
