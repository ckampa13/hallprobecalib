import numpy as np
import pandas as pd
import pickle as pkl
from hallprobecalib.hallprobe import HallProbe
from hallprobecalib import hpc_ext_path

ddir = hpc_ext_path+"datafiles/simulations/"

df = pd.read_pickle(ddir+"validation_df.pkl")

params_lists = pkl.load(open(ddir+"params_list_3.pkl","rb"))
params_truths = pkl.load(open(ddir+"params_truth_3.pkl","rb"))
params_fits = pkl.load(open(ddir+"params_fit_3.pkl","rb"))

hp = HallProbe(params_lists, params_truths, params_fits, df,)
hp.to_pickle(ddir+"hp_test_01.pkl")

hp_load = HallProbe.from_pickle(ddir+"hp_test_01.pkl")
