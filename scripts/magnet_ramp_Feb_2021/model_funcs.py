import numpy as np

# exponential: temperature stabilization
def mod_exp(x, **params):
    return params['A'] + params['B'] * np.exp(-x / params['C'])

# linear: NMR vs. temperature regression
def mod_lin(x, **params):
    return params['A'] + params['B'] * x

# linear alternate: Hall vs. temperature regression using NMR slope
def mod_lin_alt(x, **params):
    return params['C'] + params['A'] + params['B'] * x
