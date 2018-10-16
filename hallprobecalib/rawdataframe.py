#!/usr/bin/python
import numpy as np
import pandas as pd

def RawDataFrame(filename):
    # f1 = "../hallprobecalib_extras/datafiles/2018-10-03 125726.txt"
    with open(filename,'r') as file:
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
    data['X_ZAB'] = pd.Series([data['Zaber_Meas_Coord'][row][0] for row in range(ndata)])
    data['Y_ZAB'] = pd.Series([data['Zaber_Meas_Coord'][row][2] for row in range(ndata)])
    data['Z_ZAB'] = pd.Series([data['Zaber_Meas_Coord'][row][4] for row in range(ndata)])
    data['BX_CAL_1'] = pd.Series([data.Hall_Cal_Field[row][0] for row in range(ndata)])
    data['BY_CAL_1'] = pd.Series([data.Hall_Cal_Field[row][2] for row in range(ndata)])
    data['BZ_CAL_1'] = pd.Series([data.Hall_Cal_Field[row][4] for row in range(ndata)])
    data['BX_CAL_2'] = pd.Series([data.Hall_Cal_Field[row][1] for row in range(ndata)])
    data['BY_CAL_2'] = pd.Series([data.Hall_Cal_Field[row][3] for row in range(ndata)])
    data['BZ_CAL_2'] = pd.Series([data.Hall_Cal_Field[row][5] for row in range(ndata)])
    data['B_1'] = pd.Series([(data.BX_CAL_1[row]**2+data.BY_CAL_1[row]**2+data.BZ_CAL_1[row]**2)**(1/2) for row in range(ndata)])
    data['B_2'] = pd.Series([(data.BX_CAL_2[row]**2+data.BY_CAL_2[row]**2+data.BZ_CAL_2[row]**2)**(1/2) for row in range(ndata)])
    return data, meta_dict # return the dataframe and the metadata dictionary for the input file



if __name__ == '__main__':
    df_raw, meta_raw = RawDataFrame(hpc_ext_path+"datafiles/2018-10-03 125726.txt")
