import numpy as np
import pandas as pd
import pickle as pkl
# from hallprobecalib.hallprobe import HallProbe
from hallprobecalib.hallprobe import *
from hallprobecalib import hpc_ext_path

ddir = hpc_ext_path+"datafiles/simulations/"

df = pd.read_pickle(ddir+"validation_df.pkl")

df["Bz"] = df["B"]*np.cos(df["Theta"])
df["Bx"] = df["B"]*np.sin(df["Theta"])*np.cos(df["Phi"])
df["By"] = df["B"]*np.sin(df["Theta"])*np.sin(df["Phi"])
df["V0"] = df["V_obs"]
df["t"] = df["Temp"]

df_full = df.loc[np.isin(df.Theta_deg, df.Theta_deg.unique()[::4]) \
                 & np.isin(df.Phi_deg, df.Phi_deg.unique()[::4])]

"""
params_lists = pkl.load(open(ddir+"params_list_3.pkl","rb"))
params_truths = pkl.load(open(ddir+"params_truth_3.pkl","rb"))
params_fits = pkl.load(open(ddir+"params_fit_3.pkl","rb"))

hp = HallProbe(params_lists, params_truths, params_fits, df,)
hp.to_pickle(ddir+"hp_test_01.pkl")

hp_load = HallProbe.from_pickle(ddir+"hp_test_01.pkl")
"""
