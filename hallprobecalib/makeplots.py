#!/usr/bin/python
from hallprobecalib.hallbhistos1 import HallBHistos1
from hallprobecalib.tempplots1 import TempPlots1
from hallprobecalib.scatter3d import Scatter3d

def MakePlots(df):
    # cut T = 0
    temp1 = df.TEMP_1 != 0
    temp2 = df.TEMP_2 != 0
    HallBHistos1(df[temp1 & temp2])
    TempPlots1(df[temp1 & temp2])
    Scatter3d(df)
    Scatter3d(x=df.X_ZAB,y=df.Y_ZAB,z=df.Z_ZAB, cs=df.FFT_MAX,cslabel='FFT Max (a.u.)', alpha=0.5, colorsMap='hot', psize=3)
    Scatter3d(x=df.X_ZAB,y=df.Y_ZAB,z=df.Z_ZAB, cs=df.NMR_B_AVG,cslabel='B (T)', alpha=0.5, colorsMap='hot', psize=3)

if __name__=="__main__":
    data, meta = RawDataFrame(hpc_ext_path+"datafiles/2018-10-03 125726.txt")
    MakePlots(data)
    input()
