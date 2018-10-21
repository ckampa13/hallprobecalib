#!/usr/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hallprobecalib.rawdataframe import RawDataFrame
from hallprobecalib import hpc_ext_path

def TempPlots1(df):
    t1 = (df['TEMP_1'] != 0)
    t2 = (df['TEMP_2'] != 0)

    ftempboth = plt.figure()
    df[t1]["TEMP_1"].plot(title="Temperature vs. Measurement # (Near Hall Probes)",linewidth=0.75,figsize=(10,7),label="Probe #1")
    df[t2]["TEMP_2"].plot(linewidth=0.75,label="Probe #2")
    plt.xlabel("Measurement #")
    plt.ylabel("Temp (deg C)")
    plt.legend()
    ftempboth.show()

    ftempsub = plt.figure()
    df[t1 & t2]["TEMP_1"].sub(df[t1 & t2]["TEMP_2"]).plot(title="Measured Temperatured Difference vs. Measurement #",linewidth=0.5,figsize=(10,7),label="TEMP_1 - TEMP_2")
    plt.axhline(y=0.,color='GRAY',linewidth=0.75,linestyle='--')
    plt.xlabel("Measurement #")
    plt.ylabel("Temp (deg C)")
    plt.legend()
    ftempsub.show()

if __name__=="__main__":
    data, meta = RawDataFrame(hpc_ext_path+"datafiles/2018-10-03 125726.txt")
    TempPlots1(data)
    input()
