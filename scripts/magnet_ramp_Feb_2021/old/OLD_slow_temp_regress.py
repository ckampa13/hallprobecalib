import numpy as np
import pandas as pd
import lmfit as lm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from plotting import config_plots

config_plots()

# pickle file
ddir = '/home/ckampa/data/hallprobecalib_extras/datafiles/magnet_ramp/'

# FEMM data
df_f0 = pd.read_csv(ddir+'gap75_B_vs_I_r0z0_0-300_results.txt', skiprows=8, names=['I','B'], delimiter=',')
df_fp = pd.read_csv(ddir+'gap75_B_vs_I_r0z37.5_results.txt', skiprows=8, names=['I', 'B'], delimiter=',')
df_f0 = df_f0.query('I <= 140').copy()
df_f0.eval('I = I*2', inplace=True)
df_fp = df_fp.query('I <= 140').copy()
df_fp.eval('I = I*2', inplace=True)

df_f0['B_ratio'] = df_f0.B / df_fp.B

# FEMM df
df_f = pd.read_csv(ddir+'gap75_B_vs_I_r0z0_0-300_results.txt', skiprows=8, names=['I','B'], delimiter=',')
df_f = df_f.query('I <= 140').copy()
df_f.eval('I = I*2', inplace=True)
# df = pd.read_pickle('/home/ckampa/Desktop/slow_controls_current.pkl')
df = pd.read_pickle(ddir+'ramp_2021-02-24_processed.pkl')
probe = '6A0000000D61333A'

# Is_set = np.array([224., 240.]) # testing equipment
# Is_set = np.array([0., 32., 64., 96., 128., 144., 160., 176., 224., 240.])
Is_set = np.array([0., 32., 64., 96., 128., 144., 160., 176., 192., 208., 224., 240., 256., 272., 281.])
dI = 5. # amps

T0 = 14.5 # deg C, middle of Yoke (center magnet) range, given a room set point of 15 deg

def lin(temp, **params):
    return params['b'] + params['m'] * temp


I_means = []
B_fits = []
B_stds = []
NMR_fits = []
NMR_stds = []
# for I in Is_set[:1]:
for I in Is_set:
    df_ = df[(df['Magnet Current [A]'] >= I-dI) & (df['Magnet Current [A]'] <= I+dI)].copy()
    # quick check
    m = abs(df_['NMR [T]']-df_['NMR [T]'].mean()) < 0.01 # 0.1 # filter outliers
    df_ = df_[m].copy()
    I_means.append(df_['Magnet Current [A]'].mean())
    # I_means.append(df_['Magnet Current [A]'].median())
    print(f'mean(I) = {I_means[-1]:1.3E}')
    # fig, ax = plt.subplots()
    # ax.scatter(df_['Yoke (center magnet)'], df_[f'{probe}_Cal_Bmag'])
    # plt.show()
    # NMR
    sd = df_.seconds_delta.diff()
    nmr = np.mean(df_['NMR [T]'])
    if (sd.max() > 1000) & (nmr > 0.1):
        print(len(df_))
        df_NMR = df_.iloc[:np.argmax(sd)]
        print(len(df_NMR))
    else:
        df_NMR = df_
    # df_NMR = df_NMR[20:]
    df_NMR = df_NMR[60:] # remove first hour
    # df_NMR = df_NMR[140:] # remove first hours
    # THIS ONE
    # df_NMR = df_NMR[240:] # remove first hours
    # regress out the temp
    x = df_NMR['Yoke (center magnet)']
    y = df_NMR[f'{probe}_Cal_Bmag']
    ix = x.argsort()
    x = x[ix]
    y = y[ix]
    # NMR
    if nmr > 0.1:
        df_NMR2 = df_NMR[df_NMR['NMR [T]'] > 0.1]
    else:
        df_NMR2 = df_NMR
    # df_NMR2 = df_NMR2[60:] # remove first hour
    # df_NMR2 = df_NMR2[60:] # remove first hour
    x2 = df_NMR2['Yoke (center magnet)']
    y2 = df_NMR2['NMR [T]']
    ix2 = x2.argsort()
    y2 = y2[ix2]
    # print(len(x2), len(y2))
    model = lm.Model(lin, independent_vars=['temp'])
    params = lm.Parameters()
    params.add('b', value=0, vary=True)
    params.add('m', value=0, vary=True)
    # result = model.fit(y, temp=x, params=params)
    result = model.fit(y, temp=x, params=params, weights=1/(3e-5))# 1/(1e-4))
    result2 = model.fit(y2, temp=x2, params=params, weights=1/(1e-6))# 1/(1e-4))
    # print(result.fit_report())
    # print(f'm / b = {result.params["m"].value / result.params["b"].value : 0.3E}')
    # print(f'm / I = {result.params["m"].value / I_means[-1] : 0.3E}')
    print(result2.fit_report())
    print(f'm / b = {result2.params["m"].value / result2.params["b"].value : 0.3E}')
    print(f'm / I = {result2.params["m"].value / I_means[-1] : 0.3E}')
    # result.plot()
    result2.plot(data_kws={'c':'green'})
    # print fit value at T0
    B_ = lin(T0, **result.params)
    print(f'Best fit B value at T0 = {B_:1.5f}')
    p_m = {'m':result.params['m'].value-result.params['m'].stderr, 'b':result.params['b']-result.params['b'].stderr}
    p_p = {'m':result.params['m'].value+result.params['m'].stderr, 'b':result.params['b']+result.params['b'].stderr}
    B_m = lin(T0, **p_m)
    B_p = lin(T0, **p_p)
    print(f'B_min = {B_m:1.5f}, B_max = {B_p:1.5f}')
    print(f'B-B_min = {B_-B_m:1.5f}, B_max-B = {B_p-B_:1.5f}')
    B_err = np.mean([B_-B_m, B_p-B_])
    B_fits.append(B_)
    B_stds.append(B_err)
    # NMR
    B2_ = lin(T0, **result2.params)
    p2_m = {'m':result2.params['m'].value-result2.params['m'].stderr, 'b':result2.params['b']-result2.params['b'].stderr}
    p2_p = {'m':result2.params['m'].value+result2.params['m'].stderr, 'b':result2.params['b']+result2.params['b'].stderr}
    B2_m = lin(T0, **p2_m)
    B2_p = lin(T0, **p2_p)
    NMR_err = np.mean([B2_-B2_m, B2_p-B2_])
    NMR_fits.append(B2_)
    NMR_stds.append(NMR_err)

I_means = np.array(I_means)
B_fits = np.array(B_fits)
B_stds = np.array(B_stds)
NMR_fits = np.array(NMR_fits)
NMR_stds = np.array(NMR_stds)

print(I_means)
print(B_fits)
print(B_stds)
print(NMR_fits)
print(NMR_stds)

# fig, ax = plt.subplots()
# ax.errorbar(I_means, B_fits, B_stds, capsize=3, markersize=5, fmt='o')

# fit on prepped data!
def B_func(I, **params):
    I_l = I[I<params['cutoff']]
    I_h = I[I>=params['cutoff']]
    B_l = params['a'] + params['b']*(I_l - params['cutoff'])
    B_h = params['a'] + params['b1']*(I_h - params['cutoff'])**(params['p1'])
    return np.concatenate([B_l,B_h])

# df_ = df[df['Magnet Current [A]'] < 110].copy()
cutoff = 285 # 280 # 150
m = I_means < cutoff
x = I_means[m]
y = B_fits[m]
yerr = B_stds[m]
ix = x.argsort()
x = x[ix]
y = y[ix]
yerr = yerr[ix]
model2 = lm.Model(B_func, independent_vars=['I'])
params = lm.Parameters()
params.add('cutoff', value=cutoff, vary=False)
params.add('b1', value=0, vary=False)
params.add('p1', value=0, vary=False)
params.add('a', value=0, vary=True)
params.add('b', value=0, vary=True)
# result = model.fit(y, temp=x, params=params)
result2 = model2.fit(y, I=x, params=params, weights=1/yerr)# 1/(1e-4))
print(result2.fit_report())
result2.plot()

# polyfit
deg = 7 # 6 # 5 # 3
p = np.polyfit(x, y, deg)
print(f'Polyfit coeff: {p}')
yfit = np.polyval(p, x)
# interp
f = interp1d(x, y, kind='cubic', fill_value='extrapolate')
# xs = np.linspace(0, cutoff, 100)
xs = np.linspace(0, 285, 2851)
y_f = np.polyval(p, xs)
y_int = f(xs)
B_f = np.polyval(p, x)
B_int = f(x)
fig1_r, ax1_r = plt.subplots()
ax1_r.plot(df_f.I, df_f.B, color='gray', label='FEMM')
ax1_r.errorbar(I_means, B_fits, B_stds, capsize=3, markersize=5, fmt='o', label='Processed Data (Hall)')
ax1_r.errorbar(I_means, NMR_fits, NMR_stds, capsize=3, markersize=5, fmt='o', color='green', label='Processed Data (NMR)')
ax1_r.plot(xs, y_f, 'r--', label='Fit')
ax1_r.plot(xs, y_int, '-.', color='purple', label='Interpolation (cubic)')
ax1_r.set_xlabel('Magnet Current [A]')
ax1_r.set_ylabel(r'$|B|$ [T]')
ax1_r.legend()

fig2_r, ax2_r = plt.subplots()
ax2_r.scatter(xs, y_int - y_f, s=5, label='interp - fit')
ax2_r.set_xlabel('Magnet Current [A]')
ax2_r.set_ylabel(r'$\Delta B$ (interp. - fit) [T]')

fig3_r, ax3_r = plt.subplots()
ax3_r.scatter(I_means, B_fits - B_f, s=5, label=f'Fit (deg {deg} poly)')
ax3_r.scatter(I_means, B_fits - B_int, s=5, label='Interp')
ax3_r.set_xlabel('Magnet Current [A]')
ax3_r.set_ylabel(r'$\Delta B$ (data - fit) [T]')
ax3_r.legend()

redchi2 = np.sum((yfit-y)**2/yerr**2) / (len(y)-deg)
print(f'With polyfit, degree {deg}: redchi2 = {redchi2:.3f}')

# B ratio
fig4_r, ax4_r = plt.subplots()
ax4_r.plot(I_means, B_fits / NMR_fits, 'o-', markersize=5, label='Processed Data')
ax4_r.scatter(df['Magnet Current [A]'], df[f'{probe}_Cal_Bmag'] / df['NMR [T]'], s=1, color='purple', label='Raw Data')
m = df_f0.I > 80.
ax4_r.plot(df_f0[m].I, df_f0[m].B_ratio, color='gray', label='FEMM')
# ax.errorbar(I_means, NMR_fits, NMR_stds, capsize=3, markersize=5, fmt='o', color='green', label='Processed Data (NMR)')
# ax.plot(xs, y_f, 'r--', label='Fit')
# ax.plot(xs, y_int, '-.', color='purple', label='Interpolation (cubic)')
ax4_r.set_xlabel('Magnet Current [A]')
ax4_r.set_ylabel(r'$|B|_\mathrm{Hall} / |B|_\mathrm{NMR}$')
ax4_r.legend()


if __name__=='__main__':
    plt.show()
