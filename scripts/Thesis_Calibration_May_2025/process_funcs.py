import numpy as np
import pandas as pd
from dateutil import parser
from pandas.api.types import is_numeric_dtype
from scipy.interpolate import interp1d

# local imports
from Zaber_Magnet_Convert import ADC_to_mm


def load_data(file, header=None, pklname=None, low_NMR=False):
    if header is None:
        with open(file, 'r') as f:
            firstline = f.readline().split(',')
            header = [e.strip(' ').strip('\n') for e in firstline]
    df = pd.read_csv(file, names=header, skiprows=1)
    # parse dates and set as index
    dates = [parser.parse(row.Time) for row in df.itertuples()]
    df['Datetime'] = pd.to_datetime(dates)
    df.sort_values(by=['Datetime'], inplace=True)
    df = df.set_index('Datetime')
    # calculate time since beginning in useful units
    df['seconds_delta'] = (df.index - df.index[0]).total_seconds()
    df['hours_delta'] = (df.index - df.index[0]).total_seconds()/60**2
    df['days_delta'] = (df.index - df.index[0]).total_seconds()/(24*60**2)
    # remove bad column ""
    cols = list(df.columns)
    if "" in cols:
        cols.remove("")
        df = df[cols]
    # add conversion to magnet coordinates
    if "Zaber_Meas_Encoder_X" in cols:
        for i in ['X', 'Y', 'Z']:
            df[f'magnet_{i}_mm'] = ADC_to_mm(df[f'Zaber_Meas_Encoder_{i}'].values, coord=i, low_NMR=low_NMR)
    # save to pickle
    if not pklname is None:
        df.to_pickle(pklname)
    return df

def get_probe_IDs(df):
    probes = [c[:-6] for c in df.columns if "Raw_X" in c]
    return sorted(probes)

def Hall_probe_basic_calcs(df, probes):
    # magnitudes and angles
    for p in probes:
        for t in ['Raw', 'Cal']:
            df[f'{p}_{t}_Bmag'] = (df[f'{p}_{t}_X']**2 + df[f'{p}_{t}_Y']**2 + df[f'{p}_{t}_Z']**2)**(1/2)
            df[f'{p}_{t}_Theta'] = np.arccos(df[f'{p}_{t}_Z']/df[f'{p}_{t}_Bmag'])
            df[f'{p}_{t}_Phi'] = np.arctan2(df[f'{p}_{t}_Y'],df[f'{p}_{t}_X'])
    # again using theta redefining Cal components
    for p in probes:
        for t in ['Cal']:
            df[f'{p}_{t}_Theta2'] = np.arccos(df[f'{p}_{t}_X']/df[f'{p}_{t}_Bmag'])
            df[f'{p}_{t}_Phi2'] = np.arctan2(df[f'{p}_{t}_Z'],df[f'{p}_{t}_Y'])
        ### flip phi when theta < 0 (-2)
        '''
        #inds = np.where(df.loc[:, f'{p}_Cal_Theta2'] < -2.0)
        inds = np.where((df.loc[:, 'SmarAct_Meas_Angle_1'] >= 349) & (df.loc[:, 'SmarAct_Meas_Angle_1'] <= 358))
        print(df.iloc[inds][f'{p}_Cal_Phi2'].iloc[0])
        df.loc[df.index[inds], f'{p}_Cal_Phi2'] = df.loc[df.index[inds], f'{p}_Cal_Phi2'] - np.pi
        print(df.iloc[inds][f'{p}_Cal_Phi2'].iloc[0])
        inds2 = np.where(df.loc[:, f'{p}_Cal_Phi2'] < -np.pi)
        print(df.iloc[inds2][f'{p}_Cal_Phi2'].iloc[0])
        df.loc[df.index[inds2], f'{p}_Cal_Phi2'] = df.loc[df.index[inds2], f'{p}_Cal_Phi2'] + 2*np.pi
        print(df.iloc[inds2][f'{p}_Cal_Phi2'].iloc[0])
        '''
    # magnet slow controls
    df['Magnet Resistance [Ohm]'] = df['Magnet Voltage [V]'] / df['Magnet Current [A]']
    df['Coil Resistance [Ohm]'] = 2*df['Magnet Resistance [Ohm]']
    df['Magnet Power [W]'] = df['Magnet Voltage [V]'] * df['Magnet Current [A]']
    # center smaract measured values around zero
    a1 = df.loc[:, 'SmarAct_Meas_Angle_1'].copy().values
    a2 = df.loc[:, 'SmarAct_Meas_Angle_2'].copy().values
    a1[a1 > 180.] = a1[a1 > 180.] - 360.
    a2[a2 > 180.] = a2[a2 > 180.] - 360.
    df.loc[:, 'SmarAct_Meas_Angle_1_Centered'] = a1
    df.loc[:, 'SmarAct_Meas_Angle_2_Centered'] = a2
    ### INTERPOLATE NMR VALUES TO SCAN TIMES
    #### STOPPED HERE 2022-11-29 17:41:36 
    return df

# run time matching and interpolation
def match_temp_scan_dfs(df_temp, df):
    # assumes scan df is a subset of temp df.
    t0 = df.index[0]
    tf = df.index[-1]
    df_t = df_temp.query(f'"{t0}" <= Datetime <= "{tf}"').copy()
    return df_t

def interp_temp_col(df_temp, df, col='NMR [T]', kind='linear'):
    interp_func = interp1d(df_temp.seconds_delta, df_temp[col].values, kind=kind, fill_value='extrapolate')
    df.loc[:, col] = interp_func(df.seconds_delta)
    return df