#!/usr/bin/python
import numpy as np
from datetime import datetime
from dateutil import parser
import pandas as pd
import pickle as pkl
from hallprobecalib import hpc_ext_path

def HallRawDataFrame(filename,suffix='',frompickle=False,clean=True,makepickle=False):
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
    date_idx = []
    i = 0
    for idx,row in enumerate(data[:25]): # this is probably not good long term...fine as long as we have <= 25 columns in datafile.
        if isdatetime(row[0]):
            date_idx.append(idx)
            i += 1
        if i == 2:
            break
    # FIX ME...kludge for old data file without timing info
    if len(date_idx) == 0:
        ncol = 9
    else:
        ncol = date_idx[1]-date_idx[0]
    ###CHECK NUMBER HERE###
    data = data.reshape(-1,ncol) # makes a 'list of lists' with 9 columns (number of collections) and -1 makes it pick the correct value...ie independent of dataset size!
    # data = data.reshape(-1,9) # makes a 'list of lists' with 9 columns (number of collections) and -1 makes it pick the correct value...ie independent of dataset size!
    ndata = len(data) # number of datapoints, probably not needed but leaving for now
    meta_dict = {"datetime":datetime,"headers":meta_headers,"data":meta}
    if len(date_idx) != 0:
        for row in data:
            # row[0] = ['DATETIME',datetime.strptime(str(row[0][0]), '%m/%d/%Y %I:%M:%S %p')]
            # row[0] = ['DATETIME',row[0][0][:-2].replace(":","").replace("/","").replace(" ","")]
            row[0] = ['DATETIME',parser.parse(row[0][0])]
    data = [{ls[0] : nparraycreate(ls) for ls in row} for row in data] # for a row in data (9 columns), create a dictionary where each key is a column of the final dataframe, and the value is the remaining part of the list, or the data
    # data = [{ls[0] : np.array([float(x) for x in ls[1:]]) for ls in row} for row in data] # for a row in data (9 columns), create a dictionary where each key is a column of the final dataframe, and the value is the remaining part of the list, or the data
    data = pd.DataFrame(data) # turn list of list of dictionaries into a dataframe!

    # power supply multimeter
    try:
        data['PS_DMM_AVG'] = pd.Series([np.mean(row.PS_DMM) for row in data.itertuples()])
    except:
        pass

    # LakeShore probes...not relevant for Mu2e but nice to check out
    try:
        data.rename(columns={"LakeShore_Field X":"LakeShore_Field_X","LakeShore_Field Y":"LakeShore_Field_Y","LakeShore_Field Z":"LakeShore_Field_Z","LakeShore_Field V":"LakeShore_Field_V"}, inplace=True)
        data["LS_BX"] = pd.Series([np.mean(row.LakeShore_Field_X)/10. for row in data.itertuples()])
        data["LS_BY"] = pd.Series([np.mean(row.LakeShore_Field_Y)/10. for row in data.itertuples()])
        data["LS_BZ"] = pd.Series([np.mean(row.LakeShore_Field_Z)/10. for row in data.itertuples()])
        data["LS_B_MAG"] = pd.Series([np.mean(row.LakeShore_Field_V)/10. for row in data.itertuples()])
    except:
        pass

    # nmr probe
    try:
        data["NMR_B_AVG"] = pd.Series([np.mean(row.NMR_B) for row in data.itertuples()])
        data["FFT_MAX"] = pd.Series([npmaxempty(row.NMR_FFT) for row in data.itertuples()])
    except:
        pass

    # if "NMR_FFT" in list(data.columns):
    #     _ = np.array([])
    #     for row in data.itertuples():
    #         try:
    #             dum = np.max(row.NMR_FFT)
    #         except:
    #             print("FFT Import Error!")
    #             print(list(row.NMR_FFT))
    #             dum = np.nan
    #         _ = np.append(_,dum)
    #     data["FFT_MAX"] = pd.Series(_)
    #     # data["FFT_MAX"] = pd.Series([np.nanmax(row.NMR_FFT) for row in data.itertuples()])

    # smar_act stage
    try:
        #ANGLE STUFF
        data['AX0_DEG'] = pd.Series([row.SmarAct_Meas_Coord[0] for row in data.itertuples()])
        data['AX1_DEG'] = pd.Series([row.SmarAct_Meas_Coord[2] for row in data.itertuples()])
        data["AX0_PAT"] = pd.Series([row.SmarAct_Pattern_Coord[0] for row in data.itertuples()])
        data["AX1_PAT"] = pd.Series([row.SmarAct_Pattern_Coord[1] for row in data.itertuples()])
    except:
        pass

    # zaber stage
    try:
        data['X_ZAB'] = pd.Series([row.Zaber_Meas_Coord[0] for row in data.itertuples()])
        data['Y_ZAB'] = pd.Series([row.Zaber_Meas_Coord[2] for row in data.itertuples()])
        data['Z_ZAB'] = pd.Series([row.Zaber_Meas_Coord[4] for row in data.itertuples()])
        data['X_ZAB_PAT'] = pd.Series([row.Zaber_Meas_Coord[1] for row in data.itertuples()])
        data['Y_ZAB_PAT'] = pd.Series([row.Zaber_Meas_Coord[3] for row in data.itertuples()])
        data['Z_ZAB_PAT'] = pd.Series([row.Zaber_Meas_Coord[5] for row in data.itertuples()])
        #data = data.sort_values(by=['X_ZAB','Y_ZAB','Z_ZAB']) # sort values for gradient calculation
        # center coordinates
        x0 = data.X_ZAB.median()
        y0 = data.Y_ZAB.median()
        z0 = data.Z_ZAB.median()
        x0p = data.X_ZAB_PAT.median()
        y0p = data.Y_ZAB_PAT.median()
        z0p = data.Z_ZAB_PAT.median()
        data["X"] = pd.Series([row.X_ZAB-x0 for row in data.itertuples()])
        data["Y"] = pd.Series([row.Y_ZAB-y0 for row in data.itertuples()])
        data["Z"] = pd.Series([row.Z_ZAB-z0 for row in data.itertuples()])
        data["X_PAT"] = pd.Series([row.X_ZAB_PAT-x0p for row in data.itertuples()])
        data["Y_PAT"] = pd.Series([row.Y_ZAB_PAT-y0p for row in data.itertuples()])
        data["Z_PAT"] = pd.Series([row.Z_ZAB_PAT-z0p for row in data.itertuples()])
    except:
        pass

    try:
        data['GRAD_B_X'],data['GRAD_B_Y'],data['GRAD_B_Z'],data['GRAD_B_MAG'] = gradient_calc(df=data,f="NMR_B_AVG")
    except:
        pass

    # Hall probes
    try:
        #BFIELD STUFF
        data['BX_RAW'] = pd.Series([row.Hall_Raw_Field[0::3] for row in data.itertuples()])
        data['BY_RAW'] = pd.Series([row.Hall_Raw_Field[1::3] for row in data.itertuples()])
        data['BZ_RAW'] = pd.Series([row.Hall_Raw_Field[2::3] for row in data.itertuples()])
        data['B_MAG_RAW'] = (data.BX_RAW**2+data.BY_RAW**2+data.BZ_RAW**2)**(1/2)
        data['BX_CAL'] = pd.Series([row.Hall_Cal_Field[0::3] for row in data.itertuples()])
        data['BY_CAL'] = pd.Series([row.Hall_Cal_Field[1::3] for row in data.itertuples()])
        data['BZ_CAL'] = pd.Series([row.Hall_Cal_Field[2::3] for row in data.itertuples()])
        data['B_MAG_CAL'] = (data.BX_CAL**2+data.BY_CAL**2+data.BZ_CAL**2)**(1/2)
        # data['BX_RAW'] = data['BY_RAW'] = data['BZ_RAW'] = data['BX_CAL'] = data['BY_CAL'] = data['BZ_CAL'] = pd.Series()
        # for row in data.itertuples():
            # row['BX_RAW'] = row.Hall_Raw_Field[0::3]
            # row.BY_RAW = row.Hall_Raw_Field[1::3]
            # row.BZ_RAW = row.Hall_Raw_Field[2::3]
            # row.BX_CAL = row.Hall_Cal_Field[0::3]
            # row.BY_CAL = row.Hall_Cal_Field[1::3]
            # row.BZ_CAL = row.Hall_Cal_Field[2::3]

        #TEMP STUFF
        data['TEMP'] = data.Hall_Cal_Temp

        if clean:
            data = clean_dataframe(data)

        # EXPLODE STUFF
        # read in ID name map
        with open(hpc_ext_path+'probeinfo/probe_ids.txt','r') as file:
            names = file.readlines() # data is a list where each value is a string containing one line from the file
        names = [list(x.strip('\n').split(',')) for x in names] # strip new line character, then split comma separated values
        names = {i[0]: i[2] for i in names}
        # set new column with probe ids...set as categorical data to save space and time
        n_probes = len(data.BX_RAW[0])
        probes = meta_headers[len(meta_headers)-n_probes:]
        probes = [i.replace(" ","") if i != " " else 'test' for i in probes]
        data['ID'] = pd.Series([probes for j in range(len(data))])
        probe_cols = ['BX_RAW', 'BY_RAW', 'BZ_RAW', 'B_MAG_RAW', 'BX_CAL', 'BY_CAL', 'BZ_CAL', 'B_MAG_CAL', 'TEMP', 'ID']
        data = explode(data, probe_cols, fill_value=float('nan'))
        # should maybe figure another way to do this, but for now takes np.arrays of length one
        # if "DATETIME" in data.columns:
        try:
            single_cols = ['DATETIME']
            data = explode(data, single_cols)
        except:
            pass
        data['ID'] = data['ID'].astype('category')
        data['ID_NAME'] = data.ID.map(names).astype('category') # map ID string to corresponding probe name

        ##########
        #OLD...keeping for quick reference/troubleshooting at Argonne 11/29/2018
        ##########
        # data['X_ZAB'] = pd.Series([row.Zaber_Meas_Coord[0] for row in data.itertuples()])
        # data['Y_ZAB'] = pd.Series([row.Zaber_Meas_Coord[2] for row in data.itertuples()])
        # data['Z_ZAB'] = pd.Series([row.Zaber_Meas_Coord[4] for row in data.itertuples()])
        # data = data.sort_values(by=['X_ZAB','Y_ZAB','Z_ZAB']) # sort values for gradient calculation
        # # center coordinates
        # x0 = data.X_ZAB.median()
        # y0 = data.Y_ZAB.median()
        # z0 = data.Z_ZAB.median()
        # data["X"] = pd.Series([row.X_ZAB-x0 for row in data.itertuples()])
        # data["Y"] = pd.Series([row.Y_ZAB-y0 for row in data.itertuples()])
        # data["Z"] = pd.Series([row.Z_ZAB-z0 for row in data.itertuples()])
        # data['BX_CAL_1'] = pd.Series([row.Hall_Cal_Field[0] for row in data.itertuples()])
        # data['BY_CAL_1'] = pd.Series([row.Hall_Cal_Field[2] for row in data.itertuples()])
        # data['BZ_CAL_1'] = pd.Series([row.Hall_Cal_Field[4] for row in data.itertuples()])
        # data['BX_CAL_2'] = pd.Series([row.Hall_Cal_Field[1] for row in data.itertuples()])
        # data['BY_CAL_2'] = pd.Series([row.Hall_Cal_Field[3] for row in data.itertuples()])
        # data['BZ_CAL_2'] = pd.Series([row.Hall_Cal_Field[5] for row in data.itertuples()])
        # data['BX_RAW_1'] = pd.Series([row.Hall_Raw_Field[0] for row in data.itertuples()])
        # data['BY_RAW_1'] = pd.Series([row.Hall_Raw_Field[2] for row in data.itertuples()])
        # data['BZ_RAW_1'] = pd.Series([row.Hall_Raw_Field[4] for row in data.itertuples()])
        # data['BX_RAW_2'] = pd.Series([row.Hall_Raw_Field[1] for row in data.itertuples()])
        # data['BY_RAW_2'] = pd.Series([row.Hall_Raw_Field[3] for row in data.itertuples()])
        # data['BZ_RAW_2'] = pd.Series([row.Hall_Raw_Field[5] for row in data.itertuples()])
        # data['B_MAG_CAL_1'] = pd.Series([(row.BX_CAL_1**2+row.BY_CAL_1**2+row.BZ_CAL_1**2)**(1/2) for row in data.itertuples()])
        # data['B_MAG_CAL_2'] = pd.Series([(row.BX_CAL_2**2+row.BY_CAL_2**2+row.BZ_CAL_2**2)**(1/2) for row in data.itertuples()])
    except:
        pass


    # clean out unneeded columns
    if clean:
        data = clean_dataframe(data)

    # make pickle serialized file, if specified in function call
    if makepickle:
        make_pickle(data,meta_dict,filename,suffix=suffix)


    return data, meta_dict # return the dataframe and the metadata dictionary for the input file


def isdatetime(string):
    '''Helper function to handle exceptions in the list comprehension to find number of columns'''
    try:
        parser.parse(string)
        return True
    except:
        return False

def nparraycreate(ls):
    '''Helper function to handle exceptions in the list comprehension converting each line of data
    to a dictionary, for later. Specifically handles str to float error from missing data.
    input: list of form ["header","datum1","datum2",...]'''
    try:
        return np.array([float(x) for x in ls[1:]])
    except:
        #return float('NaN')
        if len(ls) == 2:
            return ls[1]
        else:
            return ls[1:] # only case when a list might still appear in dataframe...fix them!

def npmaxempty(ls):
    '''Helper function to handle exceptions in the list comprehension finding the max value
    (e.g. FFT_MAX). Specifically handles case with empty data (corresponds to no response
    from NMR)
    input: nparray (could be empty)'''
    try:
        return np.max(ls)
    except:
        return np.nan

def explode(df, lst_cols, fill_value=''):
    '''Function to expand dataframe with numpy arrays for column values to have a new row for
    each value. This makes dataframe manipulations much nicer.'''
    # make sure 'lst_cols' is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except 'lst_cols'
    idx_cols = df.columns.difference(lst_cols)
    # length of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # no empty lists
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one empty list...likely from bad measurements
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]

def clean_dataframe(df,delete=["PS_DMM","NMR_B","NMR_FFT","Hall_Cal_Field","Hall_Cal_Temp","Hall_Raw_Field","Hall_Raw_Temp","Zaber_Meas_Coord","Zaber_Pattern_Coord","SmarAct_Meas_Coord","SmarAct_Pattern_Coord","LakeShore_Field Precision","LakeShore_Field Unit",'LakeShore_Field_V', 'LakeShore_Field_X', 'LakeShore_Field_Y','LakeShore_Field_Z'],keep=[]):
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

def gradient_calc(df, f='NMR_B_AVG', coord='ZAB', ds=False):
    # f determines what quantity to determine the gradient; coord handles discrepancy of measured vs pattern coordinates: 'ZAB' for measured, 'PAT' for pattern
    if len(df.X.unique())*len(df.Y.unique())*len(df.Z.unique()) != len(df):
        cut = df.X != df.X.unique()[-1]
    elif not ds:
        cut = df.FFT_MAX != -1000
    else:
        cut = df.Y != -10000000.
    if ('ZAB' in coord) or ds:
        x = df[cut].X.unique()
        y = df[cut].Y.unique()
        z = df[cut].Z.unique()
    elif 'PAT' in coord:
        x = df[cut].X_PAT.unique()
        y = df[cut].Y_PAT.unique()
        z = df[cut].Z_PAT.unique()
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


if __name__ == '__main__':
    df_raw, meta_raw = RawDataFrame(hpc_ext_path+"datafiles/2018-11-29 113332.txt")

