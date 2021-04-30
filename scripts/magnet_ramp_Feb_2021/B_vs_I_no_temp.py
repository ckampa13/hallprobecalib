import numpy as np
import pandas as pd
import pickle as pkl
import lmfit as lm
from scipy.interpolate import interp1d
from datetime import datetime
import matplotlib.pyplot as plt
# local imports
from configs import (
    pklfit_temp_hall,
    pklfit_temp_hall_nmr,
    pklfit_temp_nmr,
    pklinfo,
    pklinfo_hall_regress,
    pklinfo_nmr_regress,
    pklproc,
    plotdir,
    probe,
)
from factory_funcs import get_NMR_B_at_T0_func, get_Hall_B_at_T0_func
from plotting import config_plots
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

def plot_B_vs_I(df_NMR, df_Hall):
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
    ax.set_xlabel('Magnet Current [A]')
    ax.set_ylabel(r'$|B|$ [T]')
    ax.set_title(r'$B$ vs. $I$ Regressed Data')
    ax.legend().set_zorder(103)
    return fig, ax

def plot_R_vs_I(df_NMR):
    fig, ax = plt.subplots()
    ax.errorbar(df_NMR.I, df_NMR.B_ratio, yerr=df_NMR.sigma_B_ratio,
                c='blue', fmt='o', ls='none', ms=4, zorder=101, capsize=2,
                label='Hall')
    ax.set_xlabel('Magnet Current [A]')
    ax.set_ylabel(r'$|B|_\mathrm{Hall} / |B|_\mathrm{NMR}$')
    ax.set_title(r'$B_\mathrm{ratio}$ vs. $I$ Regressed Data')
    #ax.legend().set_zorder(103)
    return fig, ax

# SCRATCH CALCULATION OF ERROR
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

    # plots
    fig, ax = plot_B_vs_I(df_NMR, df_Hall)
    fig, ax = plot_R_vs_I(df_NMR)

    # test output
    print('NMR df:')
    print(df_NMR)
    print('Hall df:')
    print(df_Hall)

    timef = datetime.now()
    print(f'Runtime: {timef-time0} [H:MM:SS])\n')
    plt.show()
