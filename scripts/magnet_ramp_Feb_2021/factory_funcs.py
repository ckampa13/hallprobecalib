import numpy as np
# local imports
from model_funcs import mod_lin, mod_lin_alt


def get_NMR_B_at_T0_func(results):
    # results can be a dictionary from either NMR or Hall
    # temperature regression containing lmfit results
    def lin_plus_unc(T0, run_num):
        if T0 == results[run_num].params['X0'].value:
            print(f'T0 equal, run_num {run_num}')
            y0 = results[run_num].params['X0'].value
            yerr = results[run_num].params['X0'].stderr
        else:
            x = T0 # redefinition
            x0 = results[run_num].params['X0'].value
            # model calculation at T0
            y0 = mod_lin(x, **results[run_num].params)
            # error propagation
            cov = results[run_num].covar
            yerr = ((x-x0)**2 * cov[1,1] + cov[0,0] +
                    2 * (x-x0) * cov[0,1])**(1/2)
        return y0, yerr
    return lin_plus_unc

def get_Hall_B_at_T0_func(results_hall_nmr, results_nmr):
    # results can be a dictionary from either NMR or Hall
    # temperature regression containing lmfit results
    def lin_plus_unc_hall(T0, run_num):
        x = T0 # redefinition
        x0 = results[run_num].params['X0'].value
        # get parameters and covariance matrices from Hall and NMR results
        params = results_hall_nmr[run_num].params
        params_nmr = results_nmr[run_num].params
        cov = results_hall_nmr[run_num].covar
        cov_nmr = results_nmr[run_num].covar
        # determine which regression model was used
        # parse result.model.name
        if results_hall_nmr[run_num].model.name[6:-1] == 'mod_lin':
            # linear model
            # model calculation at T0
            y0 = mod_lin(x, **params)
            # error propagation
            yerr = ((x-x0)**2 * cov[1,1] + cov[0,0] +
                    2 * (x-x0) * cov[0,1])**(1/2)
        # else mod_lin_alt
        else:
            # alternate linear model
            # model calculation at T0
            y0 = mod_lin_alt(x, **params)
            # error propagation
            yerr = ((x-x0)**2 * cov_nmr[1,1] + cov_nmr[0,0] +
                    2 * (x-x0) * cov_nmr[0,1] + cov[0,0])**(1/2)
        return y0, yerr
    return lin_plus_unc_hall

# before T0 offset
'''
def get_NMR_B_at_T0_func(results):
    # results can be a dictionary from either NMR or Hall
    # temperature regression containing lmfit results
    def lin_plus_unc(T0, run_num):
        x = T0 # redefinition
        # model calculation at T0
        y0 = mod_lin(x, **results[run_num].params)
        # error propagation
        cov = results[run_num].covar
        yerr = (x**2 * cov[1,1] + cov[0,0] + 2 * x * cov[0,1])**(1/2)
        return y0, yerr
    return lin_plus_unc

def get_Hall_B_at_T0_func(results_hall_nmr, results_nmr):
    # results can be a dictionary from either NMR or Hall
    # temperature regression containing lmfit results
    def lin_plus_unc_hall(T0, run_num):
        x = T0 # redefinition
        # get parameters and covariance matrices from Hall and NMR results
        params = results_hall_nmr[run_num].params
        params_nmr = results_nmr[run_num].params
        cov = results_hall_nmr[run_num].covar
        cov_nmr = results_nmr[run_num].covar
        # determine which regression model was used
        # parse result.model.name
        if results_hall_nmr[run_num].model.name[6:-1] == 'mod_lin':
            # linear model
            # model calculation at T0
            y0 = mod_lin(x, **params)
            # error propagation
            yerr = (x**2 * cov[1,1] + cov[0,0] + 2 * x * cov[0,1])**(1/2)
        # else mod_lin_alt
        else:
            # alternate linear model
            # model calculation at T0
            y0 = mod_lin_alt(x, **params)
            # error propagation
            yerr = (x**2 * cov_nmr[1,1] + cov_nmr[0,0] + 2 * x * cov_nmr[0,1] + cov[0,0])**(1/2)
        return y0, yerr
    return lin_plus_unc_hall
'''
