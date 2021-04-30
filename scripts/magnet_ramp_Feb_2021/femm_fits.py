import numpy as np
import pandas as pd
import lmfit as lm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
# local imports
from model_funcs import ndeg_poly1d
from configs import (
    femmfile_75_estimate,
    femmfile_75_Hall,
    femmfile_75_NMR,
    Hall_currents,
    NMR_currents,
    plotdir
)
from plotting import config_plots
config_plots()


def load_data(femmfile, meas_currents):
    # load FEMM file and multiple current by factor of 2
    if femmfile == femmfile_75_estimate:
        raise NotImplementedError
    # read FEMM file
    df = pd.read_csv(femmfile, skiprows=8, names=['I','B'])
    # scale current
    df.eval('I = I * 2', inplace=True)
    # grab measurement points
    df_meas = df[np.isin(df.I,meas_currents)].copy()
    # remove high end of FEMM df
    #df = df.query(f'I <= {np.max(meas_currents) + 3.}')
    df = df.query('I <= 281.01')
    return df, df_meas

def simple_plot(hall, meas_hall, nmr, meas_nmr):
    # plot the full FEMM datasets and measured points
    fig, ax = plt.subplots()
    # plot full datasets
    ax.plot(nmr.I, nmr.B, color='blue', linewidth=1, zorder=99,
            label='NMR Position (full)')
    ax.plot(hall.I, hall.B, color='orange', linewidth=1, zorder=98,
            label='Hall Position (full)')
    # scatter with measurement data
    ax.scatter(meas_nmr.I, meas_nmr.B, s=15, marker='+', c='black',
               zorder=101, label='NMR Position (measurements)')
    ax.scatter(meas_hall.I, meas_hall.B, s=15, marker='*', c='red',
               zorder=100, label='Hall Position (measurements)')
    # formatting
    ax.set_xlabel('Magnet Current [A]')
    ax.set_ylabel(r'$|B|$ [T]')
    ax.set_title(r'FEMM Models: $75$ mm Gap')
    ax.legend().set_zorder(102)
    return fig, ax

def simple_ratio_plot(hall, meas_hall, nmr, meas_nmr):
    # plot the full FEMM datasets and measured points
    fig, ax = plt.subplots()
    # plot full datasets
    ax.plot(nmr.I, hall.B/nmr.B, color='blue', linewidth=2, zorder=99,
            label='Full')
    # scatter with measurement data
    # must remove low current without NMR
    mh = meas_hall.query('I > 100')
    mn = meas_nmr.query('I > 100')
    ax.scatter(mn.I, mh.B/mn.B, s=50, marker='+', c='black',
               zorder=100, label='Measurements')
    # formatting
    ax.set_xlabel('Magnet Current [A]')
    ax.set_ylabel(r'$|B|_\mathrm{Hall}/|B|_\mathrm{NMR}$')
    ax.set_title(r'FEMM Models: $75$ mm Gap')
    ax.legend().set_zorder(101)
    return fig, ax

def fit_femm_npoly(ndeg, df_meas, df_full, name='NMR', I_min=-1000):
    # query raw data to get run
    # df_i = df_info.iloc[run_num]
    # df_ = df_raw.query(f'"{df_i.t0}" < index < "{df_i.tf}"').copy()
    # df_['run_hours'] = (df_.index - df_.index[0]).total_seconds()/60**2
    # # if run is long enough, only fit first 100 hours
    # if df_['run_hours'].max() > 100.:
    #     df_2 = df_.query('run_hours < 100')
    # else:
    #     df_2 = df_
    # create an array for weights if float supplied
    # if type(ystd) != np.ndarray:
    #     ystd = ystd*np.ones(len(df_2))
    # inject noise in data
    df_ = df_meas.copy()
    df_ = df_.query(f'I >= {I_min}')
    df = df_full.copy()
    df = df.query(f'I >= {I_min}')
    if name=='NMR':
        std = 5e-6
    else:
        std = 3e-5
    ystd = std * np.ones(len(df_))
    noise = np.random.normal(loc=0, scale=std, size=len(df_))
    df_.loc[:, 'B'] = df_.loc[:, 'B'] + noise
    # setup lmfit model
    model = lm.Model(ndeg_poly1d, independent_vars=['x'])
    params = lm.Parameters()
    for i in range(ndeg+1):
        params.add(f'C_{i}', value=0, vary=True)
    # fit
    result = model.fit(df_.B.values, x=df_.I.values,
                       params=params, weights=1/ystd, scale_covar=False)
    # plot
    # label for fit
    #label = rf'$\underline{{\mathrm{{Degree {ndeg} Polynomial}} }}' + '\n\n'
    label = f'Degree {ndeg} Polynomial' + '\n\n'
    label_coeffs = []
    for i in range(ndeg+1):
        pv = result.params[f'C_{i}'].value
        label_coeffs.append(rf'$C_{{{i}}} = {pv:0.3E}$'+'\n')
    #label_coeffs = [rf'$C_{i} = {result.params[f"C_{i}"]:0.3E}$'+'\n' for i in
    #                range(ndeg+1)]
    label += (''.join(label_coeffs)+'\n'+
              rf'$\chi^2_\mathrm{{red.}} = {result.redchi:0.2f}$'+'\n\n')
    #label= (r'$\underline{y = A + B e^{-x / C}}$'+'\n\n'+
            # rf'$A = {result.params["A"].value:0.2f}$'+'\n'+
            # rf'$B = {result.params["B"].value:0.2f}$'+'\n'+
            # rf'$C = {result.params["C"].value:0.2f}$'+'\n'+
            # rf'$\chi^2_\mathrm{{red.}} = {result.redchi:0.2f}$'+'\n\n')
    # set up figure with two axes
    config_plots()
    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.31, 0.8, 0.6))
    ax2 = fig.add_axes((0.1, 0.11, 0.8, 0.2), sharex=ax1)
    # plot data and fit
    # data
    #if ystd_sf == 1:
    #    label_data = 'Data'
    #else:
    #    label_data = r'Data (error $\times$'+f'{ystd_sf})'
    label_data = 'Data'
    ax1.errorbar(df_.I.values, df_.B.values, yerr=ystd,
                 fmt='o', ls='none', ms=4, zorder=100, label=label_data)
    # fit
    ax1.plot(df_.I.values, result.best_fit, linewidth=2, color='red',
             zorder=99, label=label)
    # calculate residual (data - fit)
    res = df_.B.values - result.best_fit
    # full calculation
    res_full = df.B.values - ndeg_poly1d(df.I.values, **result.params)
    # calculate ylimit for ax2
    #yl = 1.1*(np.max(np.abs(res)) + ystd[0])
    yl = 1.1*(np.max(np.abs(res_full)) + ystd[0])
    # plot residual
    # zero-line
    xmin = np.min(df_.I.values)
    xmax = np.max(df_.I.values)
    ax2.plot([xmin, xmax], [0, 0], 'k--', linewidth=2, zorder=98)
    # residual
    ax2.plot(df.I.values, res_full, linewidth=1, color='red',
             zorder=99)
    ax2.errorbar(df_.I.values, res, yerr=ystd, fmt='o', ls='none', ms=4,
                 zorder=100)
    # formatting
    # set ylimit ax2
    ax2.set_ylim([-yl, yl])
    # remove ticklabels for ax1 xaxis
    #ax1.set_xticklabels([])
    # axis labels
    ax2.set_xlabel('Magnet Current [A]')
    ax2.set_ylabel('(Data - Fit) [T]')
    ax1.set_ylabel(r'$|B|$')
    # force consistent x axis range for ax1 and ax2
    tmin = np.min(df_.I.values) - 10
    tmax = np.max(df_.I.values) + 10
    ax1.set_xlim([tmin, tmax])
    ax2.set_xlim([tmin, tmax])
    # turn on legend
    ax1.legend().set_zorder(101)
    # add title
    fig.suptitle(f'FEMM B vs. I Modeling: Polynomial Fit for {name} Probe')
    # minor ticks
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    # inward ticks and ticks on right and top
    ax1.tick_params(which='both', direction='in', top=True, right=True,
                    bottom=True)
    ax2.tick_params(which='both', direction='in', top=True, right=True)
    # tick label format consistent
    #formatter = DateFormatter('%m-%d %H:%M')
    #ax2.xaxis.set_major_formatter(formatter)
    # rotate label for dates
    #ax2.xaxis.set_tick_params(rotation=15)
    # save figure
    #fig.tight_layout()

#######
    #fig.savefig(plotfile+'.pdf')
    #fig.savefig(plotfile+'.png')
    return result, fig, ax1, ax2


if __name__=='__main__':
    print('Running script: femm_fits.py')
    time0 = datetime.now()
    # not implemented
    # load_data(femmfile_75_estimate)
    df_hall, df_hall_meas = load_data(femmfile_75_Hall, Hall_currents)
    df_nmr, df_nmr_meas = load_data(femmfile_75_NMR, NMR_currents)
    #print(len(df_hall), len(df_nmr))
    #print(len(df_hall_meas), len(df_nmr_meas))

    # make some simple plots
    fig, ax = simple_plot(df_hall, df_hall_meas, df_nmr, df_nmr_meas)
    fig, ax = simple_ratio_plot(df_hall, df_hall_meas, df_nmr, df_nmr_meas)
    # fitting
    I_min = df_nmr_meas.query('I>120').I.min()
    #print(I_min)
    ndeg = 8#6#10
    result_nmr, fig, ax1, ax2 = fit_femm_npoly(ndeg, df_nmr_meas, df_nmr,
                                               name='NMR',
                                               I_min=I_min)
                                               #I_min=125)
    result_hall, fig, ax1, ax2 = fit_femm_npoly(ndeg, df_hall_meas, df_hall,
                                                name='Hall',
                                                I_min=I_min)
                                                #I_min=125)

    # test output
    # print('Hall Dataframes')
    # print(df_hall)
    # print(df_hall_meas)
    # print('NMR Dataframes')
    # print(df_nmr)
    # print(df_nmr_meas)

    timef = datetime.now()
    print(f'Runtime: {timef-time0} [H:MM:SS])\n')
    plt.show()
