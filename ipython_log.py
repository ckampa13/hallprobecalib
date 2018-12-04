# IPython log file

get_ipython().run_line_magic('logstart', '')
import matplotlib.backends.backend_pdf
import matplotlib
from hallprobecalib.scatter3d import Scatter3d
from hallprobecalib import hpc_ext_path
from hallprobecalib.hallrawdataframe import HallRawDataFrame
df,meta = HallRawDataFrame('datafiles/2018-11-29 113749',frompickle=True)
df.head()
for i in range(len(df.B_MAG_CAL[0])):
    fig = plt.figure()
    B = np.array([row.B_MAG_CAL[i] for row in df.itertuples()])
    plt.hist(B,bins=25,range=[mean(B)-2*std(B),mean(B)+2*std(B)],label=f"{i}: range: {min(B):.5f}-{max(B):.5f}T\nmean: {mean(B):.5f} T\nstd: {std(B):.6f}T")
    plt.legend()
    plt.title("$B_{mag}$ Calibrated")
    plt.xlabel("$B_{mag}$ (T)")
    #pdf.savefig(fig)
    #plt.close(fig)
    
B = np.array([row.B_MAG_CAL[14] for row in df.itertuples()])
fig = plt.figure()
import numpy as np
import pandas as pd
plt.hist(B,bins=25,range=[mean(B)-2*std(B),mean(B)+2*std(B)],label=f"{i}: range: {min(B):.5f}-{max(B):.5f}T\nmean: {mean(B):.5f} T\nstd: {std(B):.6f}T")
plt.legend()
B = np.clip(B,0.,1.45)
B
B = np.array([row.B_MAG_CAL[14] for row in df.itertuples()])
len(B)
B = np.clip(B,0.,1.45)
len(B)
B = np.array([row.B_MAG_CAL[14] for row in df.itertuples()])
B>1.45
B[B>1.45]
B[B<1.45]
len(B[B<1.45])
B = B[B<1.45]
fig = plt.hist()
fig = plt.figure9)
fig = plt.figure()
for i in range(len(df.B_MAG_CAL[0])):
    afsd
    fig = plt.figure()
    B = np.array([row.B_MAG_CAL[i] for row in df.itertuples()])
    plt.hist(B,bins=25,range=[mean(B)-2*std(B),mean(B)+2*std(B)],label=f"{i}: range: {min(B):.5f}-{max(B):.5f}T\nmean: {mean(B):.5f} T\nstd: {std(B):.6f}T")
    plt.legend()
    plt.title("$B_{mag}$ Calibrated")
    plt.xlabel("$B_{mag}$ (T)")
    #pdf.savefig(fig)
    #plt.close(fig)
    afsd
    
plt.hist(B,bins=25,range=[mean(B)-2*std(B),mean(B)+2*std(B)],label=f"{i}: range: {min(B):.5f}-{max(B):.5f}T\nmean: {mean(B):.5f} T\nstd: {std(B):.6f}T")
len(B)
min(B)
B = B[B>1.445]
fig = plt.figure()
plt.hist(B,bins=25,range=[mean(B)-2*std(B),mean(B)+2*std(B)],label=f"{i}: range: {min(B):.5f}-{max(B):.5f}T\nmean: {mean(B):.5f} T\nstd: {std(B):.6f}T")
plt.legend()
plt.title("$B_{mag}$ Calibrated")
plt.xlabel("$B_{mag}$ (T)")
fig = plt.figure()
plt.hist(B,bins=25,range=[mean(B)-2*std(B),mean(B)+2*std(B)],label=f"14: range: {min(B):.5f}-{max(B):.5f}T\nmean: {mean(B):.5f} T\nstd: {std(B):.6f}T")
plt.legend()
plt.title("$B_{mag}$ Calibrated")
plt.xlabel("$B_{mag}$ (T)")
120*120
exit()
