import numpy as np
import pandas as pd
from dateutil import parser
from pandas.api.types import is_numeric_dtype

# local imports
#from Zaber_Magnet_Convert import ADC_to_mm


def load_data(file, pklname=None):
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
    #for i in ['X', 'Y', 'Z']:
    #    df[f'magnet_{i}_mm'] = ADC_to_mm(df[f'Zaber_Meas_Encoder_{i}'].values, coord=i)
    # save to pickle
    if not pklname is None:
        df.to_pickle(pklname)
    return df
