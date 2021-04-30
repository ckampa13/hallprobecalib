import numpy as np

# preprocessing and processing
# exponential: temperature stabilization
def mod_exp(x, **params):
    return params['A'] + params['B'] * np.exp(-x / params['C'])

# linear: NMR vs. temperature regression
def mod_lin(x, **params):
    return params['A'] + params['B'] * x

# linear alternate: Hall vs. temperature regression using NMR slope
def mod_lin_alt(x, **params):
    return params['C'] + params['A'] + params['B'] * x

# B vs. I
def ndeg_poly(x, **params):
    # parameters must be labeled as "C_#" where # is the degree of that term
    # e.g. "C_2" is the quadratic term
    tot = 0.
    for key, val in params.items():
        deg = int(key[2:])
        tot += params[f'C_{deg}']*x**deg
    return tot

def ndeg_poly1d(x, **params):
    # parameters must be labeled as "C_#" where # is the degree of that term
    # e.g. "C_2" is the quadratic term

    # determine degree of polynomial
    max_deg = np.max([int(k[2:]) for k in params.keys()])
    # create coefficient array with zeros as default
    coeffs = np.zeros(max_deg+1)
    # loop through parameters that exist, in order
    for k, v in sorted(params.items()):
        coeffs[int(k[2:])] = v
    # reverse order (high degree first) for np.poly1d
    coeffs = coeffs[::-1]
    p = np.poly1d(coeffs)
    return p(x)
