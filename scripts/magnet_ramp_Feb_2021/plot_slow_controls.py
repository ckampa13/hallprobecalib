import pandas as pd
import matplotlib.pyplot as plt

from plotting import config_plots
config_plots()

# pickle file
ddir = '/home/ckampa/data/hallprobecalib_extras/datafiles/magnet_ramp/'
# ddir = '/home/ckampa/Desktop/'
# slow_file = ddir+'ramp_2021-02-24_raw.pkl'
slow_file = ddir+'ramp_2021-02-24_processed.pkl'

# process incoming data
sdir = '/home/ckampa/coding/hallprobecalib/scripts/magnet_ramp_Feb_2021/'
script = sdir + 'process_current_21_01_19.py'
with open(script, 'rb') as source_file:
    code = compile(source_file.read(), script, "exec")
exec(code)
# exec(code, globals, locals)

# FEMM data
df_f0 = pd.read_csv(ddir+'gap75_B_vs_I_r0z0_0-300_results.txt', skiprows=8, names=['I','B'], delimiter=',')
df_fp = pd.read_csv(ddir+'gap75_B_vs_I_r0z37.5_results.txt', skiprows=8, names=['I', 'B'], delimiter=',')
df_f0 = df_f0.query('I <= 140').copy()
df_f0.eval('I = I*2', inplace=True)
df_fp = df_fp.query('I <= 140').copy()
df_fp.eval('I = I*2', inplace=True)

df_f0['B_ratio'] = df_f0.B / df_fp.B


# probe
probe = '6A0000000D61333A'
probe0 = 'C90000000D53983A'

df = pd.read_pickle(slow_file)
df['B_ratio'] = df[f'{probe}_Cal_Bmag']/df['NMR [T]']
df['Magnet Resistance'] = df['Magnet Voltage [V]'] / df['Magnet Current [A]']
print(df.tail())

probes = []
for c in df.columns:
    if 'Cal_Bmag' in c:
        probes.append(c[:16])

temps = []
water = []
for c in df.columns:
    # if ("Chamber" in c) or ('Coil' in c) or ('LCW' in c) or ('ICW' in c) or ('Ambient' in c) or ('Floor' in c) or ('Roof' in c) or ('PS' in c) or ('Tripp' in c) or ('HVAC' in c) or ('Yoke' in c) or ('Hall Element' in c):
    if ("Chamber" in c) or (c == 'Floor') or ('Roof' in c) or ('Parameter HVAC' in c) or ('Yoke' in c) or ('Hall Element' in c):
        temps.append(c)
    if ('LCW' in c) or ('ICW' in c) or ('Coil' in c):
        water.append(c)


def make_scatter(df, xs=2*['hours_delta'], ys=['NMR [T]', f'{probe}_Cal_Bmag'], llabs=['NMR', 'Hall Probe (calibrated)'],
                xlabel='Time [hours]', ylabel=r'$|B|$ [T]'):
    fig, ax = plt.subplots()
    for x, y, ll in zip(xs, ys, llabs):
        ax.scatter(df[x], df[y], s=3, label=ll)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax

fig1, ax1 = make_scatter(df, xs=2*['hours_delta'], ys=['NMR [T]', f'{probe}_Cal_Bmag'], llabs=['NMR', 'Hall Probe (calibrated)'],
                         xlabel='Time [hours]', ylabel=r'$|B|$ [T]')
fig2, ax2 = make_scatter(df, xs=2*['hours_delta'], ys=['Yoke (center magnet)', f'{probe}_Cal_T'], llabs=['Yoke', 'Hall Probe'],
                         xlabel='Time [hours]', ylabel=r'Temperature [$^\circ$C]')
fig3, ax3 = make_scatter(df, xs=['hours_delta'], ys=['Magnet Current [A]'], llabs=[None],
                         xlabel='Time [hours]', ylabel=r'Magnet Current [A]')
fig4, ax4 = make_scatter(df, xs=2*['Magnet Current [A]'], ys=['NMR [T]', f'{probe}_Cal_Bmag'], llabs=['NMR', 'Hall probe'],
                         xlabel='Magnet Current [A]', ylabel=r'$|B|$ [T]')
fig5, ax5 = make_scatter(df, xs=2*['Yoke (center magnet)'], ys=['NMR [T]', f'{probe}_Cal_Bmag'], llabs=['NMR', 'Hall probe'],
                         xlabel=r'Yoke Temp. [$^\circ C$]', ylabel=r'$|B|$ [T]')
fig6, ax6 = make_scatter(df, xs=['hours_delta'], ys=[f'B_ratio'], llabs=['Hall/NMR'],
                         xlabel='Time [hours]', ylabel=r'$|B|_{\mathrm{Hall}}/|B|_{\mathrm{NMR}}$')
fig7, ax7 = make_scatter(df, xs=[f'Magnet Current [A]'], ys=[f'B_ratio'], llabs=['Hall/NMR'],
                         xlabel='Magnet Current [A]', ylabel=r'$|B|_{\mathrm{Hall}}/|B|_{\mathrm{NMR}}$')
fig8, ax8 = make_scatter(df, xs=len(probes)*['hours_delta'], ys=[f'{p}_Cal_Bmag' for p in probes], llabs=[f'{p}' for p in probes],
                         xlabel='Time [hours]', ylabel=r'$|B|$ [T]')
fig9, ax9 = make_scatter(df, xs=3*['hours_delta'], ys=[f'{probe}_Cal_{i}' for i in ['X','Y','Z']], llabs=[fr'$B_{i}$' for i in ['X','Y','Z']],
                         xlabel='Time [hours]', ylabel=r'$B$ [T] (components)')
fig10, ax10 = make_scatter(df, xs=len(temps)*['hours_delta'], ys=temps, llabs=temps,
                         xlabel='Time [hours]', ylabel=r'Temperature [$^\circ$C]')
fig11, ax11 = make_scatter(df, xs=len(water)*['hours_delta'], ys=water, llabs=water,
                         xlabel='Time [hours]', ylabel=r'Temperature [$^\circ$C]')
fig12, ax12 = make_scatter(df, xs=3*['hours_delta'], ys=[f'{probe0}_Cal_{i}' for i in ['X','Y','Z']], llabs=[fr'$B_{i}$' for i in ['X','Y','Z']],
                         xlabel='Time [hours]', ylabel=r'$B$ [T] (components)')
fig13, ax13 = make_scatter(df, xs=['Magnet Current [A]'], ys=['B_ratio'], llabs=['Hall/NMR'],
                         xlabel='Magnet Current [A]', ylabel=r'$|B|$ ratio (center / pole)')
fig14, ax14 = make_scatter(df, xs=['hours_delta'], ys=['Magnet Resistance'], llabs=[None],
                         xlabel='Time [hours]', ylabel=r'Magnet Equiv. Resistance [Ohms]')
m = df_f0.I > 80.
ax13.plot(df_f0[m].I, df_f0[m].B_ratio, color='gray', label='FEMM')
ax13.plot(df_f0[m].I, df_f0[m].B_ratio*(1.+1.5e-4), color='orange', label='FEMM (scaled)')
ax13.plot(df_f0[m].I+35, df_f0[m].B_ratio*(1.+1.5e-4), color='purple', label='FEMM (shifted I, scaled ratio)')
ax13.legend()

if __name__=='__main__':
    plt.show()
