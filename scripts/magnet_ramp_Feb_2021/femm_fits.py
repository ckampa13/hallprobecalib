import numpy as np
import pandas as pd
import lmfit as lm
from scipy.interpolate import interp1d
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

def fit_B_vs_I_femm(ndeg, df_meas, df_full, name='NMR', method='POLYFIT',
                    I_min=-1000, fitcolor='red', datacolor='blue',
                    fig=None, axs=None, plotfile=None):
    # copy dataframes and limit current
    df_ = df_meas.copy()
    df_ = df_.query(f'I >= {I_min}')
    df = df_full.copy()
    df = df.query(f'I >= {I_min}')
    # get correct noise level
    if name=='NMR':
        std = 5e-6
    else:
        std = 3e-5
    # inject noise in measurement data
    ystd = std * np.ones(len(df_))
    # TESTING ONLY
    #std = 0.
    #ystd = None
    ###
    noise = np.random.normal(loc=0, scale=std, size=len(df_))
    df_.loc[:, 'B'] = df_.loc[:, 'B'] + noise

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
        result = model.fit(df_.B.values, x=df_.I.values,
                           params=params, weights=1/ystd, scale_covar=False)
        # calculate B for full dataset
        B_full = ndeg_poly1d(df.I.values, **result.params)
        # residuals
        # calculate residual (data - fit)
        res = df_.B.values - result.best_fit
        # full calculation
        res_full = df.B.values - B_full
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
        interp_func = interp1d(df_.I.values, df_.B.values, kind='linear',
                               fill_value='extrapolate')
        # calculate B for meas and full dfs
        B_full = interp_func(df.I.values)
        B_meas = interp_func(df_.I.values)
        # residuals
        res = df_.B.values - B_meas
        res_full = df.B.values - B_full
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
        interp_func = interp1d(df_.I.values, df_.B.values, kind='quadratic',
                               fill_value='extrapolate')
        # calculate B for meas and full dfs
        B_full = interp_func(df.I.values)
        B_meas = interp_func(df_.I.values)
        # residuals
        res = df_.B.values - B_meas
        res_full = df.B.values - B_full
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
        interp_func = interp1d(df_.I.values, df_.B.values, kind='cubic',
                               fill_value='extrapolate')
        # calculate B for meas and full dfs
        B_full = interp_func(df.I.values)
        B_meas = interp_func(df_.I.values)
        # residuals
        res = df_.B.values - B_meas
        res_full = df.B.values - B_full
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

    # plot
    # set up figure with two axes
    # config_plots()
    if fig is None:
        fig = plt.figure()
        ax1 = fig.add_axes((0.1, 0.31, 0.8, 0.6))
        ax2 = fig.add_axes((0.1, 0.08, 0.8, 0.2))#, sharex=ax1)
    else:
        ax1, ax2 = axs
    #ax1 = fig.add_axes((0.1, 0.31, 0.7, 0.6))
    #ax2 = fig.add_axes((0.1, 0.08, 0.7, 0.2))
    # plot data and fit
    # data
    label_data = f'Finite Element \n+ Noise ({datalab})'
    #label_data = f'Data ({datalab})'
    ax1.errorbar(df_.I.values, df_.B.values, yerr=ystd, c=datacolor,
                 fmt='o', ls='none', ms=4, zorder=100, capsize=2,
                 label=label_data)
    # fit
    ax1.plot(df.I.values, B_full, linewidth=1, color=fitcolor,
             zorder=99, label=label)

    # calculate ylimit for ax2
    yl = 1.2*(np.max(np.abs(res_full)) + ystd[0])
    # plot residual
    # zero-line
    xmin = np.min(df_.I.values)
    xmax = np.max(df_.I.values)
    ax2.plot([xmin, xmax], [0, 0], '--', color='black', linewidth=1,
             zorder=98)
    # residual
    ax2.plot(df.I.values, res_full, linewidth=1, color=fitcolor,
             zorder=99)
    ax2.errorbar(df_.I.values, res, yerr=ystd, fmt='o', ls='none', ms=4,
                 c=datacolor, capsize=2, zorder=100)
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
    tmin = np.min(df_.I.values) - 10
    tmax = np.max(df_.I.values) + 10
    ax1.set_xlim([tmin, tmax])
    ax2.set_xlim([tmin, tmax])
    # turn on legend
    ax1.legend(fontsize=13).set_zorder(101)
    # add title
    fig.suptitle(f'FEMM B vs. I Modeling: {fit_name} for {name} Probe')
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


if __name__=='__main__':
    print('Running script: femm_fits.py')
    time0 = datetime.now()
    # more specific plot directory
    pdir = plotdir+'final_results/femm/'
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
    #I_min_Hall = -1000
    I_min_Hall = df_hall_meas.query('I>120').I.min()
    I_min_NMR = df_nmr_meas.query('I>120').I.min()
    #I_min = 125
    #I_min = df_nmr_meas.query('I>120').I.min()
    #print(I_min)
    ndeg = 7#6#8#6#10
    # savefile
    pfile = pdir+'{0}_75mm_{1}'
    # COMPARISON PLOTS
    # nmr
    interp_nmr, fig, ax1, ax2 = fit_B_vs_I_femm(ndeg, df_nmr_meas, df_nmr,
                                                name='NMR',
                                                I_min=I_min_NMR,
                                                fitcolor='green',
                                                datacolor='black',
                                                method='INTERP_CUBIC')

    temp = fit_B_vs_I_femm(ndeg, df_nmr_meas, df_nmr, name='NMR',
                           I_min=I_min_NMR,fitcolor='red', datacolor='blue',
                           fig = fig, axs = [ax1,ax2], method='POLYFIT',
                           plotfile=pfile.format('NMR', 'poly_vs_interp'))
    result_nmr, fig, ax1, ax2 = temp

    # hall probe
    interp_hall, fig, ax1, ax2 = fit_B_vs_I_femm(ndeg, df_hall_meas, df_hall,
                                                 name='Hall',
                                                 I_min=I_min_Hall,
                                                 fitcolor='green',
                                                 datacolor='black',
                                                 method='INTERP_CUBIC')

    temp = fit_B_vs_I_femm(ndeg, df_hall_meas, df_hall, name='Hall',
                           I_min=I_min_Hall, fitcolor='red', datacolor='blue',
                           fig = fig, axs = [ax1,ax2], method='POLYFIT',
                           plotfile=pfile.format('Hall', 'poly_vs_interp'))
    result_hall, fig, ax1, ax2 = temp

    # INTERP ONLY PLOTS
    for I in ['LIN', 'QUAD', 'CUBIC']:
        i = I.lower()
        # nmr
        temp = fit_B_vs_I_femm(ndeg, df_nmr_meas, df_nmr, name='NMR',
                               I_min=I_min_NMR, fitcolor='green',
                               datacolor='black', method=f'INTERP_{I}',
                               plotfile=pfile.format('NMR', f'interp_{i}'))
        result_nmr, fig, ax1, ax2 = temp
        # hall probe
        temp = fit_B_vs_I_femm(ndeg, df_hall_meas, df_hall, name='Hall',
                               I_min=I_min_Hall ,fitcolor='green',
                               datacolor='black', method=f'INTERP_{I}',
                               plotfile=pfile.format('Hall', f'interp_{i}'))
        result_nmr, fig, ax1, ax2 = temp

    timef = datetime.now()
    print(f'Runtime: {timef-time0} [H:MM:SS])\n')
    #plt.show()
