import numpy as np
import pandas as pd
from hallprobecalib import hpc_ext_path
from hallprobecalib.rawdataframe import RawDataFrame

df,meta = RawDataFrame('datafiles/2018-10-03 125726',frompickle=True)

nss = [25,50,100,200,500]

xs = df.X_ZAB.unique()

for x in xs:
    p = df.X_ZAB == x
    bs = np.array(df[p].NMR_B_AVG)
    bs.sort()
    print(f"X: {x}")
    for ns in nss:
        B_ranges = np.array([])
        for i in range(len(bs)-ns+1):
            B_ranges = np.append(B_ranges,bs[i+ns-1]-bs[i])
        print(f"num_elements: {ns}, min B range (T): {B_ranges.min()}")

