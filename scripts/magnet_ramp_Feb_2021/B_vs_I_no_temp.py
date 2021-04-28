import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib.pyplot as plt
# local imports
from config import (
    pklfit_temp_regress_nmr,
    pklinfo,
    pklproc,
    plotdir,
    probe,
)
from factory_funcs import get_NMR_B_at_T0_func
from plotting import config_plots
config_plots()

### EXAMPLE HOW TO GET NMR VALS
results_nmr = pkl.load(open(pklfit_temp_regress_nmr, 'rb'))
