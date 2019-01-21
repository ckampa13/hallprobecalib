# package imports
from hallprobecalib import hpc_ext_path
from hallprobecalib.hallrawdataframe import HallRawDataFrame

# external imports
import os
import re
from dateutil import parser

directory = hpc_ext_path+'datafiles/'

def pickling(frompickle=True):
    df = []
    meta = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith('raw') and filename.endswith(".txt"):
            try:
                filename = filename[:-4]
                if frompickle==True:
                    df_, meta_ = HallRawDataFrame('datafiles/'+filename,frompickle=True)
                else:
                    df_, meta_ = HallRawDataFrame('datafiles/'+filename,makepickle=True)
                df.append(df_)
                meta.append(meta_)
                print(f"{filename}: {len(df_)} points")
            except:
                print(f"FAILED: {filename}")
    return df,meta

def main():
    df,meta = pickling(frompickle=True)

if __name__ == '__main__':
    main()
