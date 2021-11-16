import numpy as np
import pandas as pd
from dateutil import parser
from pandas.api.types import is_numeric_dtype

# local imports
from Zaber_Magnet_Convert import ADC_to_mm


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
    for i in ['X', 'Y', 'Z']:
        df[f'magnet_{i}_mm'] = ADC_to_mm(df[f'Zaber_Meas_Encoder_{i}'].values, coord=i)
    # save to pickle
    if not pklname is None:
        df.to_pickle(pklname)
    return df

# BAD COORDINATE TRANSFORMATIONS -- CHECK
'''
ddir = '/home/ckampa/Dropbox/LogFiles/'
slowfile = '2021-06-14 151158slow.txt'
df = load_data(ddir+slowfile)

# coordinate transformations
# check
Zaber_ADC_lims = {'X':[422190, 554838], 'Y':[201704, 637825], 'Z': [-1783025, -1442393]}
ADC_at_lims = {i: Zaber_ADC_lims[i][0] for i in ['X', 'Y', 'Z']}
ADC_per_micron_XYZ = np.array([(df[f"Zaber_Meas_Encoder_{i}"].max() - df[f"Zaber_Meas_Encoder_{i}"].min()) for i in ['X', 'Y', 'Z']])\
/np.array([(df[f"Zaber_Meas_Micron_{i}"].max() - df[f"Zaber_Meas_Micron_{i}"].min()) for i in ['X', 'Y', 'Z']])
ADC_per_micron = ADC_per_micron_XYZ[0]
micron_per_ADC = 1/ADC_per_micron
ADC_per_mm = ADC_per_micron * 1e3 # 1e3 micron / mm
mm_per_ADC = 1/ADC_per_mm
# reference ADCs
#ref_ADCS = {i: sf*df[f'Zaber_Pattern_{i}'].min() for i, sf in zip(['X', 'Y', 'Z'], [1, 1, -1])}
ref_ADCS = {i: sf*df[f'Zaber_Meas_Encoder_{i}'].min() for i, sf in zip(['X', 'Y', 'Z'], [1, 1, -1])}
ref_mm = {i: sf*df[f'Zaber_Meas_Micron_{i}'].min() for i, sf in zip(['X', 'Y', 'Z'], [1e-3, 1e-3, -1e-3])}
# calculate difference from limits to reference
mm_at_lims = {}
for i in ref_ADCS.keys():
    delta_ADC = ref_ADCS[i] - Zaber_ADC_lims[i][0]
    delta_mm = delta_ADC * mm_per_ADC
    mm_at_lims[i] = ref_mm[i] - delta_mm
# brass spacers
spacer_R = 19.33/2 # mm
spacer_centers = {'X': [0, 0 , 0], 'Y': [51.+spacer_R, 136.+spacer_R, 226.+spacer_R], 'Z': [50., 230.+spacer_R, 47.]}
# offsets
xwidth = 12.90 # mm, digital caliper
#xwidth = 11.788 # mm, laser + zaber move
#xwidth = 30.1 - 18.2  # mm ESTIMATED!!! PLEASE MEASURE
xoff = xwidth - 2 # mm, accounts for NMR sample depth
zoff = 12.73 + 4/32 * 25.4 # mm, width of Al / 2 + height Al above spacer
offset_AL = {'X': xoff, 'Y': 0., 'Z': zoff}
#offset_AL = {'X': xoff, 'Y': 0., 'Z': 12.73+3/32 * 25.4} # NEED TO MEASURE X

# conversion
def zaber_mm_to_mag_mm(zaber_mm, coord):
    return zaber_mm - (mm_at_lims[coord] - offset_AL[coord] - spacer_centers[coord][0])

def mag_mm_to_zaber_mm(mag_mm, coord):
    return mag_mm + (mm_at_lims[coord] - offset_AL[coord] - spacer_centers[coord][0])

def zaber_mm_to_zaber_ADC(zaber_mm, coord):
    delta_mm = zaber_mm - mm_at_lims[coord]
    delta_ADC = delta_mm * ADC_per_mm
    return ADC_at_lims[coord] + delta_ADC

def zaber_ADC_to_zaber_mm(zaber_ADC, coord):
    delta_ADC = zaber_ADC - ADC_at_lims[coord]
    delta_mm = delta_ADC * mm_per_ADC
    return mm_at_lims[coord] + delta_mm

def mag_mm_to_zaber_ADC(mag_mm, coord):
    _ = mag_mm_to_zaber_mm(mag_mm, coord)
    return zaber_mm_to_zaber_ADC(_, coord)

def zaber_ADC_to_mag_mm(zaber_ADC, coord):
    _ = zaber_ADC_to_zaber_mm(zaber_ADC, coord)
    return zaber_mm_to_mag_mm(_, coord)
'''
