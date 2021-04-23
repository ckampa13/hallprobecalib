import numpy as np

def mod_exp(x, **params):
    return params['A'] + params['B'] * np.exp(-x / params['C'])

def mod_lin(x, **params):
    return params['A'] + params['B'] * x
