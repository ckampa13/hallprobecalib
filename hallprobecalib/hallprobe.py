#!/usr/bin/env python
import numpy as np
import scipy.special
import pandas as pd
import pickle as pkl

from hallprobecalib import hpc_ext_path


def voltage_decomp_all(df, kmax, nmax, lmax, sigma):
    # df contains calibration data with columns:
    # Bx, By, Bz, t, V0, V1, V2

    # estimate th0 and ph0 for each probe.
    th0 = {}
    ph0 = {}
    for n in [0, 1, 2]:
        th0[n], ph0[n] = est_angles(df, n)

    # call secondary decomp function given th0 and ph0
    pararms = {}
    covs = {}
    for n in [0, 1, 2]:
        params[n], covs[n] = voltage_decomp(df, n, th0[n], ph0[n], kmax, nmax, lmax, sigma)

    return params[0], params[1], params[2]

def est_angles(df, n):
    # n is an int
    return 0., 0.

def voltage_decomp(df, n, th0, ph0, kmax, nmax, lmax, sigma):
    #

    # setup to V, thetas, phis, t
    Vs = df[f"V{n}"].values
    Bs = ((df["Bx"]**2+df["By"]**2+df["Bz"]**2)**(1/2)).values
    thetas = (np.arctan2((df["Bx"]**2+df["By"]**2)**(1/2),df["Bz"]) + th0).values
    phis = (np.arctan2(df["By"],df["Bx"]) + ph0).values
    ts = df["t"].values

    # return Vs, Bs, thetas, phis, ts

    # params indexes
    ks = np.arange(0, kmax+1)
    ns = np.arange(0, nmax+1)
    ls = np.arange(0, lmax+1)

    # construct G
    js = []
    for k in ks:
        for n in ns:
            for l in ls:
                for m in range(0, l+1):
                    js.append({"k":k,"n":n,"l":l,"m":m})
    G = []
    for j in js:
        # G.append(G_j(Bs, ts, thetas, phis, **j).values)
        G.append(G_j(Bs, ts, thetas, phis, **j))
    G = np.array(G).T

    # return G

    #d = Vs # data vector
    GtG_inv = np.linalg.inv(G.T @ G)
    m = GtG_inv @ G.T @ Vs
    # cov
    m_cov = sigma**2 * GtG_inv

    # print(len(js), len(m))

    # make dict
    params = {f"c_{j['k']}{j['n']}{j['l']}{j['m']}":m_ for j, m_ in zip(js, m)}
    # for j, m_ in zip(js, m):
    #     m_dict[f"c_{j[k]}{j[n]}{j[l]}{j[m]}"] =

    return params, m_cov


def G_j(Bs, ts, thetas, phis, k, n, l, m):
    '''
    Parameters:
    length-N numpy.arrays -- Bs (magnetic field),
    ts (temperature), thetas (polar), phis (azimuthal)
    int -- k, n, l, m (parameters needed in that column)

    Return: length-N numpy.array -- G_j (column j of G matrix)
    '''
    # initialize Chebyshev coefficients
    cs_k = np.zeros(k+1)
    cs_k[-1] = 1
    cs_n = np.zeros(n+1)
    cs_n[-1] = 1
    # calculate Chebyshev and Spherical
    Tk = np.polynomial.chebyshev.chebval(Bs, cs_k)
    Tn = np.polynomial.chebyshev.chebval(ts, cs_n)
    Ylm = np.real(scipy.special.sph_harm(m, l, phis, thetas))
    return Tk * Tn * Ylm





class HallProbe(object):
    def __init__(self, id_str, params_0, params_1, params_2):
        self.id_str = id_str
        self.he0 = HallElement(params_0)
        self.he1 = HallElement(params_1)
        self.he2 = HallElement(params_2)

    @classmethod
    def from_pickle(cls, filename):
        return pkl.load(open(filename, "rb"))

    @classmethod
    def from_cal_dat(cls, id_str, df, kmax, nmax, lmax, sigma):
        # df contains calibration data with columns:
        # Bx, By, Bz, t, V0, V1, V2
        params_0, params_1, params_2 = voltage_decomp_all(df, kmax, nmax, lmax, sigam)
        return cls(id_str, params_0, params_1, params_2)

    def V(self, Bx, By, Bz, t):
        # solve forward problem (given parameters)
        return


    def B(self, V0, V1, V2):
        # solve inverse problem
        #ff
        return

    def to_pickle(self, filename):
        pkl.dump(self, open(filename, "wb"))

"""
class HallProbe(object):
    def __init__(self, params_lists, params_truths=3*[None], params_fits=3*[None], theta0s=[np.pi, np.pi/2, np.pi/2], phi0s=[0, np.pi/2, np.pi], calib_data=None, id_str="testprobe"):
        '''
        f
        '''
        self.id = id_str
        self.calib_data = calib_data
        self.he0 = HallElement(params_lists[0], params_truths[0], params_fits[0], theta0s[0], phi0s[0])
        self.he1 = HallElement(params_lists[1], params_truths[1], params_fits[1], theta0s[1], phi0s[1])
        self.he2 = HallElement(params_lists[2], params_truths[2], params_fits[2], theta0s[2], phi0s[2])

    def voltage_decomps(self):


    def B_calculation(self, V0, V1, V2, t):
        # solves second inverse problem (voltage, temp, return Bx, By, Bz)
        #self.

        # lmfit


    @classmethod
    def from_pickle(cls, filename):
        return pkl.load(open(filename, "rb"))

    def to_pickle(self, filename):
        pkl.dump(self, open(filename, "wb"))



class HallElement(object):
    def __init__(self, params_list, params_truth=None, params_fit=None, theta0=0., phi0=0.):
        '''Initializes HallElement instance.

        Parameters:
        params (dict): Parameter dictionary

        Returns:
        HallElement object
        '''
        # check if real data (no truth)
        # if params_truth is None and params_fit is None and calib_data is None:
        #     raise RuntimeError("Cannot initialize Hall element missing "\
        #                       +"params_truth, params_fit, and calib_data!")
        self.params_list = params_list
        self.params_truth = params_truth
        self.params_fit = params_fit
        self.theta0 = theta0
        self.phi0 = phi0

        # if params_fit is None:
        #     if calib_data is not None:
        #         self.params_fit = voltage_decomp(calib_data, params_list)

        # for p in [params_list, params_truth, params_fit, calib_data]

    # def
"""

### FROM INVERSE METHODS FINAL PROJECT ###
def V_forward(Bs, ts, thetas, phis, **params):
    '''
    Parameters:
    length-N numpy.arrays -- Bs (magnetic field),
    ts (temperature), thetas (polar), phis (azimuthal)
    dictionary -- params where key is of form "c_knlm"
    and value is a float

    Return: V (Hall voltage), a length-N numpy.array
    '''
    # first determine ks, ns, and lms in use
    ks = list(set(int(i[2]) for i in params.keys()))
    ns = list(set(int(i[3]) for i in params.keys()))
    kmax = np.max(np.array(ks))
    nmax = np.max(np.array(ns))
    lms = list(set((i[4:]) for i in params.keys()))
    # loop through each lm and split into l and m
    # to store needed spherical harmonics in a dict
    Ylms = {}
    for lm_ in lms:
        l, m = [int(i) for i in list(lm_)]
        # store spherical harmonic for each lm
        Ylms[lm_] = np.real(scipy.special.sph_harm(m, l, phis, thetas))

    # initialize coefficient array for Chebyshev (B)
    # starting with T_0(B) = 0 for all N data points
    cs_B = [np.zeros_like(Bs)]
    # loop through all terms in model
    #### B ####
    for k in ks:
        # initialize coefficients to add to cs_B as zero
        cs = np.zeros_like(Bs)
        #### t ####
        for n in ns:
            # initialize another coeff array for Chebyshev (t)
            cs_t = np.zeros(nmax+1)
            # set the term we care about (n) to 1
            # and keep the rest = 0
            cs_t[n] = 1
            # generate Chebyshev (t)
            Tn_t = np.polynomial.chebyshev.chebval(ts, cs_t)

            # prepare array to store c_knlm * Y_lm
            params_ylms = np.zeros_like(Bs)
            #### theta, phi ####
            for lm_ in lms:
                params_ylms += params[f"c_{k}{n}{lm_}"] * Ylms[lm_]
            # add to cs (iteratively) as: T_n(t) * c_knlm * Y_lm
            cs += Tn_t * params_ylms
        # add completed cs array into cs_B
        cs_B.append(cs)
    cs_B = np.array(cs_B)
    # calculate V from all prepared terms
    V = np.polynomial.chebyshev.chebval(Bs, cs_B)
    return V


def gen_data(Bs, ts, thetas, phis, sigma):
    '''
    Parameters:
    length-N numpy.arrays -- Bs (magnetic field),
    ts (temperature), thetas (polar), phis (azimuthal)
    float -- sigma (noise standard deviation)

    Return: df, pandas.DataFrame
    '''
    # create meshgrid for all input independent variables
    B, Te, Th, Ph = np.meshgrid(Bs, ts, thetas, phis)
    # flatten all to 1D arrays
    B = B.flatten()
    Te = Te.flatten()
    Th = Th.flatten()
    Ph = Ph.flatten()

    # create pandas.DataFrame to store information from each data point
    df = pd.DataFrame({"B":B, "Temp":Te, "Theta": Th, "Phi": Ph})
    # add columns for angles in degrees for convenience
    # df["Theta_deg"] = np.degrees(df.Theta)
    # df["Phi_deg"] = np.degrees(df.Phi)
    df.loc[:,"Theta_deg"] = np.degrees(df.Theta)
    df.loc[:,"Phi_deg"] = np.degrees(df.Phi)
    # use forward function to generate voltage data and store in new column
    # df["V_obs"] = V_forward(df.B, df.Temp, df.Theta, df.Phi, **params)
    df.loc[:,"V_obs"] = V_forward(df.B, df.Temp, df.Theta, df.Phi, **params)
    # inject noise
    # df["V_obs"] = df["V_obs"] + np.random.normal(loc=0.0, scale=sigma, size=len(df))
    df.loc[:,"V_obs"] = df["V_obs"] + np.random.normal(loc=0.0, scale=sigma, size=len(df))
    # set sigma as a column
    # df["sigma_V"] = sigma
    df.loc[:,"sigma_V"] = sigma

    return df

"""
def G_j(Bs, ts, thetas, phis, k, n, l, m):
    '''
    Parameters:
    length-N numpy.arrays -- Bs (magnetic field),
    ts (temperature), thetas (polar), phis (azimuthal)
    int -- k, n, l, m (parameters needed in that column)

    Return: length-N numpy.array -- G_j (column j of G matrix)
    '''
    # initialize Chebyshev coefficients
    cs_k = np.zeros(k+1)
    cs_k[-1] = 1
    cs_n = np.zeros(n+1)
    cs_n[-1] = 1
    # calculate Chebyshev and Spherical
    Tk = np.polynomial.chebyshev.chebval(Bs, cs_k)
    Tn = np.polynomial.chebyshev.chebval(ts, cs_n)
    Ylm = np.real(scipy.special.sph_harm(m, l, phis, thetas))
    return Tk * Tn * Ylm
"""

def hall_cal_least_squares(df, params):
    '''
    Parameters:
    pandas.DataFrame -- df

    Return:
    pandas.DataFrame -- df
    dictionary -- m_dict (parameters with names)
    length-M numpy.array -- m (parameters)
    M x M numpy.ndarray -- m_cov (parameters covariance matrix)
    '''
    # determine which parameters are in parameter set
    ks = list(set(int(i[2]) for i in params.keys()))
    ns = list(set(int(i[3]) for i in params.keys()))
    kmax = np.max(np.array(ks))
    nmax = np.max(np.array(ns))
    lms = list(set((i[4:]) for i in params.keys()))

    # generate G matrix
    # determine which knlm to use for each column of G
    js = []
    for k in ks:
        for n in ns:
            for lm_ in lms:
                js.append({"k":k, "n":n, "l":int(lm_[0]), "m":int(lm_[1])})
    # generate G matrix one column at a time
    # usig G_j function
    G = []
    for j in js:
        G.append(G_j(df.B, df.Temp, df.Theta, df.Phi, **j).values)
    # combine to full G matrix
    G = np.array(G).T

    # data vector
    d = df.V_obs.values
    # least squares solution
    GtG_inv = np.linalg.inv(G.T @ G)
    m = GtG_inv @ G.T @ d
    # calculate m_cov
    m_cov = df.sigma_V.iloc[0]**2 * GtG_inv

    # construct m_dict
    m_dict = {}
    i = 0
    for k in ks:
        for n in ns:
            for lm_ in lms:
                m_dict[f"c_{k}{n}{lm_[0]}{lm_[1]}"] = m[i]
                i += 1

    # generate synthetic data from model vector m after least squares solution
    # df["V_syn"] = V_forward(df.B, df.Temp, df.Theta, df.Phi, **m_dict)
    df = df.assign(V_syn=V_forward(df.B, df.Temp, df.Theta, df.Phi, **m_dict))
    # df.loc[:,"V_syn"] = pd.Series(V_forward(df.B, df.Temp, df.Theta, df.Phi, **m_dict), index=df.index)
    # calculatate residual and relative residual
    # df["res"] = df.V_obs - df.V_syn
    df.loc[:,"res"] = df.V_obs - df.V_syn
    # df["res_rel"] = df.res / df.V_obs
    df.loc[:,"res_rel"] = df.res / df.V_obs

    return df, m_dict, m, m_cov


if __name__ == "__main__":
    # pd.options.mode.chained_assignment = 'raise'

    # parameter keys
    params_knlm = ["1110", "1122", "1210", "1222", "2110", "2122",
                           "2210", "2222", "3110", "3122", "3210", "3222",]
    base = 1e5
    # pick realistic values
    params_vals = [base, base*1e-2, base*1e-2, base*1e-3, base*1e-2, base*1e-3,
                           base*1e-3, base*1e-4, base*1e-3, base*1e-4, base*1e-4, base*1e-5]
    # generate dictionary
    params = {f"c_{p}":v for p,v in zip(params_knlm, params_vals)}
    pkl.dump(params, open(hpc_ext_path+"datafiles/simulations/params_truth.pkl", "wb"))

    # params_list = [f"c_{p}" for p in params_knlm]

    # generate starting conditions
    Nphi, Ntheta, NB, Nt = (360, 181, 4, 3)
    phis = np.linspace(0, 2*np.pi - 2*np.pi/Nphi, Nphi)
    thetas = np.linspace(0, np.pi, Ntheta)
    Bs = np.linspace(0.5, 1.25, NB)
    ts = np.linspace(21, 25, Nt)
    sigma = 0.

    # call forward data generation function and store in dataframe
    df = gen_data(Bs, ts, thetas, phis, sigma)

    # all theta
    df_full = df.loc[np.isin(df.Theta_deg, df.Theta_deg.unique()[::4]) \
                         & np.isin(df.Phi_deg, df.Phi_deg.unique()[::4])]
    df_full, m_dict, m, m_cov = hall_cal_least_squares(df_full, params)
    pkl.dump(m_dict, open(hpc_ext_path+"datafiles/simulations/params_fit_full.pkl", "wb"))

    # df["V_syn_full"] = V_forward(df.B, df.Temp, df.Theta, df.Phi, **m_dict)
    # df["res_full"] = df.V_obs - df.V_syn_full
    # df["res_rel_full"] = df.res_full / df.V_obs
    df.loc[:,"V_syn_full"] = V_forward(df.B, df.Temp, df.Theta, df.Phi, **m_dict)
    df.loc[:,"res_full"] = df.V_obs - df.V_syn_full
    df.loc[:,"res_rel_full"] = df.res_full / df.V_obs

    # limited theta
    max_theta_deg = 15 # deg
    max_theta = np.radians(max_theta_deg)

    df_FNAL = df.loc[np.isin(df.Theta_deg, df.Theta_deg.unique()[::4]) \
                         & np.isin(df.Phi_deg, df.Phi_deg.unique()[::4])]

    df_FNAL = df_FNAL.query(f"Theta <= {max_theta} or Theta >= {np.pi - max_theta}")

    cols = ["B", "Temp", "Theta", "Phi", "Theta_deg", "Phi_deg", "V_obs", "sigma_V"]

    df_FNAL = df_FNAL.loc[:,cols]
    df_FNAL, m_dict, m, m_cov = hall_cal_least_squares(df_FNAL, params)
    pkl.dump(m_dict, open(hpc_ext_path+"datafiles/simulations/params_fit_FNAL.pkl", "wb"))

    # df["V_syn_FNAL"] = V_forward(df.B, df.Temp, df.Theta, df.Phi, **m_dict)
    # df["res_FNAL"] = df.V_obs - df.V_syn_FNAL
    # df["res_rel_FNAL"] = df.res_FNAL / df.V_obs
    df.loc[:,"V_syn_FNAL"] = V_forward(df.B, df.Temp, df.Theta, df.Phi, **m_dict)
    df.loc[:,"res_FNAL"] = df.V_obs - df.V_syn_FNAL
    df.loc[:,"res_rel_FNAL"] = df.res_FNAL / df.V_obs

    # pkl.dump(df, hpc_ext_path+"simulations/validation_df.pkl")
    df.to_pickle(hpc_ext_path+"datafiles/simulations/validation_df.pkl")
