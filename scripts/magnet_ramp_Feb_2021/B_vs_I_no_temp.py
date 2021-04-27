import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib.pyplot as plt
# local imports
from config import probe, plotdir, pklinfo, pklproc, pklinfo_regress, pklfit_temp_regress_nmr
from plotting import config_plots,
from factory_funcs import get_B_at_T0_func

### EXAMPLE HOW TO GET NMR VALS
results_nmr = pkl.load(open(pklfit_temp_regress_nmr, 'rb'))
