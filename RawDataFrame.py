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
    data_points = len(data) # number of datapoints, probably not needed but leaving for now
    meta_dict = {"datetime":datetime,"headers":meta_headers,"data":meta}
    data = [{ls[0] : np.array([float(x) for x in ls[1:]]) for ls in row} for row in data] # for a row in data (9 columns), create a dictionary where each key is a column of the final dataframe, and the value is the remaining part of the list, or the data
    data = pd.DataFrame(data) # turn list of list of dictionaries into a dataframe!
    return data, meta_dict # return the dataframe and the metadata dictionary for the input file


df_raw, meta_raw = RawDataFrame("../hallprobecalib_extras/datafiles/2018-10-03 125726.txt")

# if __name__ == '__main__':
#     import timeit
#     f = "../hallprobecalib_extras/datafiles/2018-10-03 125726.txt"
#     print(timeit.timeit("RawDataFrame(f)", setup="from __main__ import RawDataFrame"))
