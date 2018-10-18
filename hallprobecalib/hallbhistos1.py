#!/usr/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hallprobecalib.rawdataframe import RawDataFrame
from hallprobecalib import hpc_ext_path

def HallBHistos1(df):
    bins = np.linspace(start=-0.000,stop=0.0005,num=101)
    bins1 = np.linspace(start=-0.0002,stop=0.0003,num=101)
    bins2 = np.linspace(start=-0.0004,stop=0.0002,num=121)

    fmag = plt.figure()
    df['B_1'].hist(bins=bins,alpha=1.0,grid=False,figsize=(10,7),label='Hall Probe #1')
    df['B_2'].hist(bins=bins,alpha=0.7,grid=False,label='Hall Probe #2')
    plt.axvline(x=5.358e-5,color='RED',linestyle='--',label='B @ IB2 (World Magnetic Model)')
    plt.title("BField (Magnitude)")
    plt.xlabel("B (T)")
    plt.ylabel("# Measurements")
    plt.legend()
    fmag.show()

    fprobe1 = plt.figure()
    df['BX_CAL_1'].hist(bins=bins1,alpha=1.0,grid=False,figsize=(10,7),label='Bx')
    df['BY_CAL_1'].hist(bins=bins1,alpha=0.5,grid=False,label='By')
    df['BZ_CAL_1'].hist(bins=bins1,alpha=0.5,grid=False,label='Bz')
    plt.title("BField (Magnitude)")
    plt.xlabel("B (T)")
    plt.ylabel("# Measurements")
    plt.legend()
    fprobe1.show()

    fprobe2 = plt.figure()
    df['BX_CAL_2'].hist(bins=bins2,alpha=1.0,grid=False,figsize=(10,7),label='Bx')
    df['BY_CAL_2'].hist(bins=bins2,alpha=0.5,grid=False,label='By')
    df['BZ_CAL_2'].hist(bins=bins2,alpha=0.5,grid=False,label='Bz')
    plt.title("Hall Probe #2: BField Calibrated")
    plt.xlabel("B (T)")
    plt.ylabel("# Measurements")
    plt.legend()
    fprobe2.show()

if __name__=="__main__":
    data, meta = RawDataFrame(hpc_ext_path+"datafiles/2018-10-03 125726.txt")
    HallBHistos1(data)
    input()

