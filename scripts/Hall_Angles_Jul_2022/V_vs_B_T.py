import numpy as np
import pandas as pd
from scipy.optimize import fsolve
# from datetime import datetime
# from datetime import timedelta
# from pandas.api.types import is_numeric_dtype
#from scipy.interpolate import interp1d
#import lmfit as lm
# from copy import deepcopy
#from dateutil import parser
# from plotly import graph_objects as go
# from plotly.offline import plot
# import matplotlib.pyplot as plt
# from matplotlib.ticker import AutoMinorLocator
# %matplotlib inline

# local imports
# from plotting import config_plots, datetime_plt
# from load_slow import *
# from Zaber_Magnet_Convert import *
# hallprobecalib package
# from hallprobecalib.hpcplots import scatter3d, scatter2d, histo
# from mu2e.mu2eplots import mu2e_plot3d


# altered from simulation / fitting studies September 2020
# but removed theta & phi dependence
def V_forward(Bs, ts, **params):
    '''
    Parameters: 
    length-N numpy.arrays -- Bs (magnetic field), 
    ts (temperature)
    dictionary -- params where key is of form "c_knlm"
    and value is a float. Also needs "Bmin, Bmax, tmin, tmax" for scaling Chebyshev polynomial inputs
    
    Return: V (Hall voltage), a length-N numpy.array
    '''
#     params_ = deepcopy(dict(params)) # wrap with dict for use in lmfit
    params_ = params
    # normalizations
    B_sub = (params['Bmax'] + params['Bmin']) / 2
    B_div = (params['Bmax'] - params['Bmin']) / 2
    Bs_scaled = (Bs - B_sub) / B_div
    t_sub = (params['tmax'] + params['tmin']) / 2
    t_div = (params['tmax'] - params['tmin']) / 2
    ts_scaled = (ts - t_sub) / t_div
    # sum limits
    kmax = params['kmax']
    nmax = params['nmax']
    # delete non-sum related parameters
    #[params.pop(key) for key in ['Bmin', 'Bmax', 'tmin', 'tmax', 'kmax', 'nmax', 'lmax', 'theta0', 'phi0']]
    
#     [params.pop(key) for key in ['Bmin', 'Bmax', 'tmin', 'tmax', 'kmax', 'nmax']]
    
    # loop B
#     V = 0.
    V = np.zeros_like(Bs_scaled)
    for k in range(kmax+1):
        coeffs_k = np.zeros(kmax+1)
        coeffs_k[k] = 1.
        T_k = np.polynomial.chebyshev.chebval(Bs_scaled, coeffs_k)
        # loop t
        for n in range(nmax+1):
            c_kn = params[f'c_{k}{n}']
            coeffs_n = np.zeros(nmax+1)
            coeffs_n[n] = 1.
            T_n = np.polynomial.chebyshev.chebval(ts_scaled, coeffs_n)
            V += c_kn * T_k * T_n 
    
    return V

def inv_B_v2(B, V, T, params):
    return [(V_forward(B, T, **params) - V)[0]]

def invert_row(df, index, probe_ID, result, B_init=1.0, Bcomp='Mag'):
    row = df.iloc[index]
    #B_ = fsolve(func=inv_B, x0=[0.8], args=(row[f'{probe}_Raw_Bmag'], row[f'{probe}_Cal_T'], result.params))
    #B_ = fsolve(func=inv_B_v2, x0=[0.8], args=(np.array([row[f'{probe}_Raw_Bmag']]), np.array([row[f'{probe}_Cal_T']]), result.params))
    B_ = fsolve(func=inv_B_v2, x0=[B_init], args=(np.array([row[f'{probe_ID}_Raw_{Bcomp}']]), np.array([row[f'{probe_ID}_Cal_T']]), result.params))
    return B_[0]

def calc_B_column(df, probe_ID, result, B_init=1.0, Bcomp='Mag'):
    Bs = []
    for i in range(len(df)):
        B_ = invert_row(df, i, probe_ID, result, B_init, Bcomp)
        Bs.append(B_)
    return np.array(Bs)