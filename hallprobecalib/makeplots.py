#!/usr/bin/python
from hallprobecalib.hallbhistos1 import HallBHistos1
from hallprobecalib.tempplots1 import TempPlots1

def MakePlots(df):
    HallBHistos1(df)
    TempPlots1(df)

if __name__=="__main__":
    data, meta = RawDataFrame(hpc_ext_path+"datafiles/2018-10-03 125726.txt")
    MakePlots(data)
    input()
