import numpy as np
import pandas as pd
import lmfit as lm
import pickle as pkl
from sys import stdout
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import AutoMinorLocator
from datetime import datetime
from dateutil import parser
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
# local imports
from model_funcs import mod_exp
from configs import probe, plotdir, rawfile, pklraw, pklinfo, tex_info, pklproc, pklproc_ramp, pklproc_hyst, pklfit_stable_temp
from run_info import t0s, tfs, ramps, hysts, adcs
from plotting import config_plots, datetime_plt, ticks_in
config_plots()


def get_probe_IDs(df):
    probes = [c[:-6] for c in df.columns if "Raw_X" in c]
    return probes

def load_save_raw(rname, pname):
    # message
    print('Loading raw data into DataFrame')
    # grab headers from first line in file and format nicely (remove leading/trailing spaces, newline)
    with open(rname, 'r') as file:
        headers = [s.strip(" ").rstrip("\n") for s in str.split(next(file), ',')]
    # load with pandas
    df = pd.read_csv(rname, skiprows=1, names=headers)
    # parse dates and set as index
    dates = [parser.parse(row.Time) for row in df.itertuples()]
    df['Datetime'] = pd.to_datetime(dates)
    df.sort_values(by=['Datetime'], inplace=True)
    df = df.set_index('Datetime')
    # calculate time since beginning in useful units
    df['seconds_delta'] = (df.index - df.index[0]).total_seconds()
    df['hours_delta'] = (df.index - df.index[0]).total_seconds()/60**2
    df['days_delta'] = (df.index - df.index[0]).total_seconds()/(24*60**2)
    # save to pickle
    df.to_pickle(pname)
    return df

def gen_save_info_df(df_raw, t0s, tfs, ramps, hysts, adcs, savename):
    # message
    print('Generating run info DataFrame')
    # construct dataframe from run info (saved in run_info.py)
    df_info = pd.DataFrame({'t0':t0s, 'tf':tfs, 'ramp':ramps, 'hyst':hysts, 'adc':adcs})
    # add column for when chiller went out -- hard-coded date
    df_info['chiller'] = df_info.t0 < '2021-03-05 14:00:01' # chiller failed after this
    # loop through each run to find NMR in range and mean current
    nmrs = []
    Bs = []
    Is = []
    hours_data = []
    ns = []
    for i, row in enumerate(df_info.itertuples()):
        # query raw dataframe to get run
        df_ = df_raw.query(f'"{row.t0}" < index < "{row.tf}"')
        # mean of current and NMR
        I = np.mean(df_['Magnet Current [A]'])
        B = np.mean(df_['NMR [T]'])
        # check if NMR was in range
        if B > 0.1:
            nmr = True
        else:
            nmr = False
        # calculate time of run
        dhours = (datetime.strptime(row.tf, '%Y-%m-%d %H:%M:%S')-\
                  datetime.strptime(row.t0, '%Y-%m-%d %H:%M:%S')).total_seconds() / 60 / 60
        n = len(df_)
        #mean_dt = np.mean(np.diff(df_.index) * 1e-9).item()
        #hours_mean = mean_dt * n / 60 / 60
        # append to lists
        nmrs.append(nmr)
        Bs.append(B)
        Is.append(I)
        ns.append(n)
        hours_data.append(dhours)
    # add calculated values to info dataframe
    df_info['NMR'] = nmrs
    df_info['B_NMR'] = Bs
    df_info['I'] = Is
    df_info['hours_data'] = hours_data
    df_info['num_meas'] = ns
    # save to pickle
    df_info.to_pickle(savename)
    return df_info

def timing_checks_latex(df_raw, df_info, writefile):
    # message
    print('Writing LaTeX run info table')
    # write header
    out_str = '\\begin{table}[h]\n'
    out_str += '\\caption{Detailed measurement information. The main ramping measurement was monotonic in current '\
    +'and was preceded and succeeded by hysteresis measurements (n.b. "PS ADC"$=$Power Supply ADC)}\n'\
    +'\\begin{adjustwidth}{-2cm}{}\n'
    out_str += '\\begin{tabular}{|r|r|r|l|l|r|r|c|c|c|c|}\n\\hline\n'
    out_str += 'Index & \\begin{tabular}[c]{@{}l@{}}Current\\\\(mean)\\end{tabular} & PS ADC & Start Time &'\
    +'End Time & \\begin{tabular}[c]{@{}l@{}}Hours\\\\Data\\end{tabular} &'\
    +' \\begin{tabular}[c]{@{}l@{}}\\# \\\\Meas.\\end{tabular} & Ramp? &'\
    +'\\begin{tabular}[c]{@{}l@{}}Hys-\\\\teresis?\\end{tabular} &'\
    +'Chiller? & \\begin{tabular}[c]{@{}l@{}}NMR in\\\\range?\\end{tabular} \\\\ \\hline \\hline\n'
    # loop - write a line in the table for each row in the info df
    for i, row in enumerate(df_info.itertuples()):
        # add all numerical/string values
        out_str += f'${i}$ & ${row.I:0.3f}$ & ${row.adc}$ & {row.t0[5:-3]} & {row.tf[5:-3]} & ${row.hours_data:0.1f}$ & ${row.num_meas}$ &'
        # check boolean columns and add a checkmark
        if row.ramp:
            out_str += ' \\checkmark &'
        else:
            out_str += ' &'
        if row.hyst:
            out_str += ' \\checkmark &'
        else:
            out_str += ' &'
        if row.chiller:
            out_str += ' \\checkmark &'
        else:
            out_str += ' &'
        if row.NMR:
            out_str += ' \\checkmark'
        # end line and add horizontal line
        out_str += ' \\\\ \\hline\n'
    # write footer
    out_str += '\\end{tabular}\n\\end{adjustwidth}\n\\label{tab:data}\n\\end{table}\n'
    # write to file
    with open(writefile, 'w') as f:
        f.write(out_str)
    return out_str

def fit_temperature_stable(run_num, df_info, df_raw, plotfile, ycol='Yoke (center magnet)', ystd=5e-2, ystd_sf=0.01):
    # query raw data to get run
    df_i = df_info.iloc[run_num]
    df_ = df_raw.query(f'"{df_i.t0}" < index < "{df_i.tf}"').copy()
    df_['run_hours'] = (df_.index - df_.index[0]).total_seconds()/60**2
    # if run is long enough, only fit first 100 hours
    if df_['run_hours'].max() > 100.:
        df_2 = df_.query('run_hours < 100')
    else:
        df_2 = df_
    # create an array for weights if float supplied
    if type(ystd) != np.ndarray:
        ystd = ystd*np.ones(len(df_2))
    # setup lmfit model
    # y = A + B * np.exp(- x / C)
    model = lm.Model(mod_exp, independent_vars=['x'])
    params = lm.Parameters()
    params.add('A', value=0, vary=True)
    params.add('B', value=0, vary=True)
    params.add('C', value=1, min=0, vary=True)
    # fit
    result = model.fit(df_2[ycol].values, x=df_2.run_hours.values,
                       params=params, weights=1/ystd)
    # plot
    # label for fit
    label=r'$\underline{y = A + B e^{-x / C}}$'+'\n\n'\
          +rf'$A = {result.params["A"].value:0.2f}$'+'\n'\
          +rf'$B = {result.params["B"].value:0.2f}$'+'\n'\
          +rf'$C = {result.params["C"].value:0.2f}$'+'\n'\
          +rf'$\chi^2_\mathrm{{red.}} = {result.redchi:0.2f}$'+'\n\n'
    # set up figure with two axes
    config_plots()
    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.31, 0.8, 0.6))
    ax2 = fig.add_axes((0.1, 0.11, 0.8, 0.2))
    # plot data and fit
    # data
    ax1.errorbar(df_2.index.values, df_2[ycol].values, yerr=ystd_sf*ystd, fmt='o', ls='none', ms=2, label=r'Data (error $\times$'+f'{ystd_sf})', zorder=100)
    # fit
    ax1.plot(df_2.index.values, result.best_fit, linewidth=2, color='red', label=label, zorder=99)
    # calculate residual (data - fit)
    res = df_2[ycol].values - result.best_fit
    # calculate ylimit for ax2
    yl = 1.1*(np.max(np.abs(res)) + ystd_sf*ystd[0])
    # plot residual
    # zero-line
    ax2.plot([np.min(df_2.index.values), np.max(df_2.index.values)], [0, 0], 'k--', linewidth=2)
    # residual
    ax2.errorbar(df_2.index.values, res, yerr=ystd_sf*ystd, fmt='o', ls='none', ms=2, zorder=99)
    # formatting
    # set ylimit ax2
    ax2.set_ylim([-yl, yl])
    # remove ticklabels for ax1 xaxis
    ax1.set_xticklabels([])
    # axis labels
    ax2.set_xlabel('Datetime [2021 MM-DD hh:mm]')
    ax2.set_ylabel(r'(Data - Fit) [$^{\circ}$C]')
    ax1.set_ylabel(r'Yoke (center magnet) Temp. [$^{\circ}$C]')
    # force consistent x axis range for ax1 and ax2
    #tmin = np.min(df_2.index.values)
    #tmax = np.max(df_2.index.values)
    #ax1.set_xlim([tmin, tmax])
    #ax2.set_xlim([tmin, tmax])
    # turn on legend
    ax1.legend()
    # add title
    fig.suptitle(f'Temperature Stability Fit: Exponential Decay\nRun Index {run_num}')
    # minor ticks
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    # inward ticks and ticks on right and top
    ax1.tick_params(which='both', direction='in', top=True, right=True, bottom=True)
    ax2.tick_params(which='both', direction='in', top=True, right=True)
    # tick label format consistent
    formatter = DateFormatter('%m-%d %H:%M')
    ax2.xaxis.set_major_formatter(formatter)
    # rotate label for dates
    ax2.xaxis.set_tick_params(rotation=15)
    # save figure
    fig.savefig(plotfile+'.pdf')
    fig.savefig(plotfile+'.png')
    # remove first "time constant" of data 
    hmax = df_.run_hours.max()
    tau = result.params["C"].value
    # special case (water chiller failed) -- only remove 1/2 time constant
    if hmax/tau < 1:
        df_ = df_.query(f'run_hours > {tau/2}').copy()
    # else normal case -- remove one time constant
    else:
        df_ = df_.query(f'run_hours > {tau}').copy()
    # add run number to df
    df_['run_num'] = run_num
    return result, df_, fig, ax1, ax2

def filter_outliers(run_num, df_clean, nsig=7, ycol='NMR [T]'):
    # query correct run
    if run_num is None:
        df_ = df_clean.copy()
    else:
        df_ = df_clean.query(f'run_num == {run_num}').copy()
    # calculate mean and stddev
    std_ = df_[ycol].std()
    mean_ = df_[ycol].mean()
    # calculate allowed range
    min_ = mean_ - nsig*std_
    max_ = mean_ + nsig*std_
    if mean_ > 0.1:
        df_ = df_.query(f'{min_} < `{ycol}` < {max_}').copy()
    return df_

def process_raw_single(run_num, df_raw, df_info):
    # generate plot file name
    pf = plotdir+f'final_results/stable_temp/run-{run_num}_temp_fit'
    # fit + plot temperature stability
    result, df_, fig, ax1, ax2 = fit_temperature_stable(run_num, df_info, df_raw, plotfile=pf, ycol='Yoke (center magnet)', ystd=0.014, ystd_sf=1)
    # filter based on NMR 2 times
    df_ = filter_outliers(None, df_, nsig=7, ycol='NMR [T]')
    df_ = filter_outliers(None, df_, nsig=7, ycol='NMR [T]')
    return df_, result

def process_raw(df_raw, df_info, p_full, p_ramp, p_hyst, p_stable):
    # message
    print('Processing raw data (temperature stability, remove NMR outliers)')
    # get probes
    probes = get_probe_IDs(df_raw)
    # get CPU information
    num_cpu = multiprocessing.cpu_count()
    nproc = min([num_cpu, len(df_info)])
    # parallel for loop
    processed_tuples = Parallel(n_jobs=nproc)(delayed(process_raw_single)(i, df_raw, df_info) for i in tqdm(df_info.index, file=stdout, desc='Run #')) 
    # split output
    dfs = [i[0] for i in processed_tuples]
    results = {i:j[1] for i,j in zip(df_info.index, processed_tuples)}
    # concatenate for cleaned dataframe
    df = pd.concat(dfs)
    # calculations for Hall probes
    for p in probes:
        # magnitudes
        df[f'{p}_Cal_Bmag'] = (df[f'{p}_Cal_X']**2 + df[f'{p}_Cal_Y']**2 + df[f'{p}_Cal_Z']**2)**(1/2)
        df[f'{p}_Raw_Bmag'] = (df[f'{p}_Raw_X']**2 + df[f'{p}_Raw_Y']**2 + df[f'{p}_Raw_Z']**2)**(1/2)
        # B field angles
        df[f'{p}_Cal_Theta'] = np.arccos(df[f'{p}_Cal_Z']/df[f'{p}_Cal_Bmag'])
        df[f'{p}_Cal_Phi'] = np.arctan2(df[f'{p}_Cal_Y'],df[f'{p}_Cal_X'])
    # split to ramp and hysteresis
    ramp_ind = df_info.query('ramp == True').index.values
    hyst_ind = df_info.query('hyst == True').index.values
    df_ramp = df[np.isin(df.run_num, ramp_ind)]
    df_hyst = df[np.isin(df.run_num, hyst_ind)]
    # pickle results
    pkl.dump(results, open(pklfit_stable_temp, "wb" ))
    # pickle dataframes
    df.to_pickle(p_full)
    df_ramp.to_pickle(p_ramp)
    df_hyst.to_pickle(p_hyst)
    return df, df_ramp, df_hyst, results


if __name__=='__main__':
    print('Running script: preprocess_data.py')
    time0 = datetime.now()
    
    df_raw = load_save_raw(rawfile, pklraw)
    #df_raw = pd.read_pickle(pklraw)
    probes = get_probe_IDs(df_raw)
    df_info = gen_save_info_df(df_raw, t0s, tfs, ramps, hysts, adcs, pklinfo)
    #df_info = pd.read_pickle(pklinfo)
    tex_info_str = timing_checks_latex(df_raw, df_info, writefile=tex_info)
    # run processing of raw data
    df, df_ramp, df_hyst, results = process_raw(df_raw, df_info, pklproc, pklproc_ramp, pklproc_hyst, pklfit_stable_temp)
    '''
    # Load pickles
    df = pd.read_pickle(pklproc)
    df_ramp = pd.read_pickle(pklproc_ramp)
    df_hyst = pd.read_pickle(pklproc_hyst)
    results = pkl.load(open(pklfit_stable_temp, "rb" ))
    '''
    
    timef = datetime.now()
    
    '''
    # Optional text output
    print('\nOutput:')
    print('Raw DataFrame:')
    print(df_raw)
    #print('\n')
    print('Info DataFrame:')
    print(df_info)
    #print('\n')
    #print(f'From configs.py: probe={probe}')
    #print(f'Scraping from raw df: probes={probes}\n')
    #print(f'LaTeX Run Info:')
    #print(tex_info_str)
    print('\nFit Report for Temperature Stability, Run Index 0')
    print(results[0].fit_report())
    print('Cleaned DataFrame:')
    print(df)
    print('\n')
    '''
    
    print(f'Runtime: {timef-time0} [H:MM:SS])\n')
    
    #plt.show()
