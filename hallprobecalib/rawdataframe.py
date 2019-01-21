#!/usr/bin/python
import numpy as np
import pandas as pd
import pickle as pkl
from hallprobecalib import hpc_ext_path

def RawDataFrame(filename,suffix='',frompickle=False,grad=True,clean=True,makepickle=False):
    if frompickle:
        meta_dict = pd.read_pickle(hpc_ext_path+filename+"_meta.pkl")
        return pd.read_pickle(hpc_ext_path+filename+".pkl"),meta_dict

    # f1 = "../hallprobecalib_extras/datafiles/2018-10-03 125726.txt"
    with open(hpc_ext_path+filename+'.txt','r') as file:
        data = file.readlines() # data is a list where each value is a string containing one line from the file
    data =  [x for x in data if x != '\n'] # strip blank lines
    data = [list(x.strip('\n').split(',')) for x in data] # strip new line character, then split comma separated values
    data = [[x+'0' if x[-1]=='E' else x for x in row] if row[0]=='PS_DMM' else row for row in data] # handle scientific notation case 'E' with no trailing digit, which means x10^0
    datetime = data[0] # string containing 'date SPACE time'
    meta_headers = data[1] # headers for metadata about x,y,z , step size, etc
    meta = data[2] # metadata about data points: x,y,z , step size, etc
    del data[0:3] # delete the header/metadata rows from the list
    data = np.array(data) # convert data into an np array for reshaping
    data = data.reshape(-1,9) # makes a 'list of lists' with 9 columns (number of collections) and -1 makes it pick the correct value...ie independent of dataset size!
    ndata = len(data) # number of datapoints, probably not needed but leaving for now
    meta_dict = {"datetime":datetime,"headers":meta_headers,"data":meta}
    data = [{ls[0] : np.array([float(x) for x in ls[1:]]) for ls in row} for row in data] # for a row in data (9 columns), create a dictionary where each key is a column of the final dataframe, and the value is the remaining part of the list, or the data
    data = pd.DataFrame(data) # turn list of list of dictionaries into a dataframe!
    data['X_ZAB'] = pd.Series([row.Zaber_Meas_Coord[0] for row in data.itertuples()])
    data['Y_ZAB'] = pd.Series([row.Zaber_Meas_Coord[2] for row in data.itertuples()])
    data['Z_ZAB'] = pd.Series([row.Zaber_Meas_Coord[4] for row in data.itertuples()])
    data = data.sort_values(by=['X_ZAB','Y_ZAB','Z_ZAB']) # sort values for gradient calculation
    # center coordinates
    x0 = data.X_ZAB.median()
    y0 = data.Y_ZAB.median()
    z0 = data.Z_ZAB.median()
    data["X"] = pd.Series([row.X_ZAB-x0 for row in data.itertuples()])
    data["Y"] = pd.Series([row.Y_ZAB-y0 for row in data.itertuples()])
    data["Z"] = pd.Series([row.Z_ZAB-z0 for row in data.itertuples()])
    data["PAT_X"] = pd.Series([row.Zaber_Pattern_Coord[0] for row in data.itertuples()])
    data["PAT_Y"] = pd.Series([row.Zaber_Pattern_Coord[1] for row in data.itertuples()])
    data["PAT_Z"] = pd.Series([row.Zaber_Pattern_Coord[2] for row in data.itertuples()])
    data['BX_CAL_1'] = pd.Series([row.Hall_Cal_Field[0] for row in data.itertuples()])
    data['BY_CAL_1'] = pd.Series([row.Hall_Cal_Field[2] for row in data.itertuples()])
    data['BZ_CAL_1'] = pd.Series([row.Hall_Cal_Field[4] for row in data.itertuples()])
    data['BX_CAL_2'] = pd.Series([row.Hall_Cal_Field[1] for row in data.itertuples()])
    data['BY_CAL_2'] = pd.Series([row.Hall_Cal_Field[3] for row in data.itertuples()])
    data['BZ_CAL_2'] = pd.Series([row.Hall_Cal_Field[5] for row in data.itertuples()])
    data['BX_RAW_1'] = pd.Series([row.Hall_Raw_Field[0] for row in data.itertuples()])
    data['BY_RAW_1'] = pd.Series([row.Hall_Raw_Field[2] for row in data.itertuples()])
    data['BZ_RAW_1'] = pd.Series([row.Hall_Raw_Field[4] for row in data.itertuples()])
    data['BX_RAW_2'] = pd.Series([row.Hall_Raw_Field[1] for row in data.itertuples()])
    data['BY_RAW_2'] = pd.Series([row.Hall_Raw_Field[3] for row in data.itertuples()])
    data['BZ_RAW_2'] = pd.Series([row.Hall_Raw_Field[5] for row in data.itertuples()])
    data['B_MAG_CAL_1'] = pd.Series([(row.BX_CAL_1**2+row.BY_CAL_1**2+row.BZ_CAL_1**2)**(1/2) for row in data.itertuples()])
    data['B_MAG_CAL_2'] = pd.Series([(row.BX_CAL_2**2+row.BY_CAL_2**2+row.BZ_CAL_2**2)**(1/2) for row in data.itertuples()])
    data["NMR_B_AVG"] = pd.Series([np.mean(row.NMR_B) for row in data.itertuples()])
    data["FFT_MAX"] = pd.Series([np.max(row.NMR_FFT) for row in data.itertuples()])
    data['TEMP_1'] = pd.Series([row.Hall_Cal_Temp[0] for row in data.itertuples()])
    data['TEMP_2'] = pd.Series([row.Hall_Cal_Temp[1] for row in data.itertuples()])

    if grad:
        data['GRAD_B_X'],data['GRAD_B_Y'],data['GRAD_B_Z'],data['GRAD_B_MAG'] = gradient_calc(df=data,f="NMR_B_AVG")

    if clean:
        data = clean_dataframe(data)

    if makepickle:
        make_pickle(data,meta_dict,filename,suffix=suffix)


    return data, meta_dict # return the dataframe and the metadata dictionary for the input file


def gradient_calc(df, f='NMR_B_AVG', coord='ZAB'):
    # f determines what quantity to determine the gradient; coord handles discrepancy of measured vs pattern coordinates: 'ZAB' for measured, 'PAT' for pattern
    if len(df.X.unique())*len(df.Y.unique())*len(df.Z.unique()) != len(df):
        cut = df.X != df.X.unique()[-1]
    else:
        cut = df.FFT_MAX != -1000
    if 'ZAB' in coord:
        x = df[cut].X_ZAB.unique()
        y = df[cut].Y_ZAB.unique()
        z = df[cut].Z_ZAB.unique()
    elif 'PAT' in coord:
        x = df[cut].PAT_X.unique()
        y = df[cut].PAT_Y.unique()
        z = df[cut].PAT_Z.unique()
    # xx,yy,zz = np.meshgrid(x,y,z,indexing='ij')
    ff = np.array(df[cut] [f])  #1D here
    ff = np.reshape(ff,(len(x),len(y),len(z))) #3D here
    gradx, grady, gradz = np.gradient(ff,x,y,z)
    gradmag = (gradx**2 + grady**2 + gradz**2)**(1/2)
    # must now get back into 1D to stick into dataframe
    gradx = pd.Series(gradx.flatten())
    grady = pd.Series(grady.flatten())
    gradz = pd.Series(gradz.flatten())
    gradmag = pd.Series(gradmag.flatten())

    return gradx,grady,gradz,gradmag


def clean_dataframe(df,delete=["NMR_B","NMR_FFT","Hall_Cal_Field","Hall_Cal_Temp","Hall_Raw_Field","Hall_Raw_Temp","Zaber_Meas_Coord","Zaber_Pattern_Coord"],keep=[]):
    if type(delete)!=list: delete = list(delete)
    if type(keep)!=list: delete = list(keep)
    d = list(set(delete)-set(keep))
    d = [i for i in d if i in df.columns] # don't drop something that doesn't exist! allows us to run function on an already cleaned dataframe!
    return df.drop(d,axis=1)


def make_pickle(df,meta,filename,suffix=''):
    meta_filename = filename+suffix+"_meta.pkl"
    filename = filename+suffix+".pkl"
    pkl.dump(meta, open(hpc_ext_path+meta_filename, "wb"), protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(df, open(hpc_ext_path+filename, "wb"), protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    df_raw, meta_raw = RawDataFrame(hpc_ext_path+"datafiles/2018-10-03 125726.txt")
