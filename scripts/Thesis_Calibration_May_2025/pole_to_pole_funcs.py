import numpy as np

def get_B_corr_func(params, I0=50.):
    poly_func = np.polynomial.Polynomial(params[::-1])
    def B_corr_func(I):
        #I0 = 50.
        m = (I < I0)
        B0 = -poly_func(I0)
        B_Hall_min_NMR = -poly_func(I)
        B_Hall_min_NMR[m] = B0
        return B_Hall_min_NMR
    return B_corr_func