import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from plotly import graph_objects as go
from scipy.interpolate import interp1d


#### processing
def get_probe_IDs(df):
    probes = [c[:-6] for c in df.columns if "Raw_X" in c]
    return sorted(probes)

def Hall_probe_basic_calcs(df, probes):
    # magnitudes and angles
    for p in probes:
        for t in ['Raw', 'Cal']:
            df[f'{p}_{t}_Bmag'] = (df[f'{p}_{t}_X']**2 + df[f'{p}_{t}_Y']**2 + df[f'{p}_{t}_Z']**2)**(1/2)
            df[f'{p}_{t}_Theta'] = np.arccos(df[f'{p}_{t}_Z']/df[f'{p}_{t}_Bmag'])
            df[f'{p}_{t}_Phi'] = np.arctan2(df[f'{p}_{t}_Y'],df[f'{p}_{t}_X'])
    # again using theta redefining Cal components
    for p in probes:
        for t in ['Cal']:
            df[f'{p}_{t}_Theta2'] = np.arccos(df[f'{p}_{t}_X']/df[f'{p}_{t}_Bmag'])
            df[f'{p}_{t}_Phi2'] = np.arctan2(df[f'{p}_{t}_Z'],df[f'{p}_{t}_Y'])
        ### flip phi when theta < 0 (-2)
        '''
        #inds = np.where(df.loc[:, f'{p}_Cal_Theta2'] < -2.0)
        inds = np.where((df.loc[:, 'SmarAct_Meas_Angle_1'] >= 349) & (df.loc[:, 'SmarAct_Meas_Angle_1'] <= 358))
        print(df.iloc[inds][f'{p}_Cal_Phi2'].iloc[0])
        df.loc[df.index[inds], f'{p}_Cal_Phi2'] = df.loc[df.index[inds], f'{p}_Cal_Phi2'] - np.pi
        print(df.iloc[inds][f'{p}_Cal_Phi2'].iloc[0])
        inds2 = np.where(df.loc[:, f'{p}_Cal_Phi2'] < -np.pi)
        print(df.iloc[inds2][f'{p}_Cal_Phi2'].iloc[0])
        df.loc[df.index[inds2], f'{p}_Cal_Phi2'] = df.loc[df.index[inds2], f'{p}_Cal_Phi2'] + 2*np.pi
        print(df.iloc[inds2][f'{p}_Cal_Phi2'].iloc[0])
        '''
    # magnet slow controls
    df['Magnet Resistance [Ohm]'] = df['Magnet Voltage [V]'] / df['Magnet Current [A]']
    df['Coil Resistance [Ohm]'] = 2*df['Magnet Resistance [Ohm]']
    df['Magnet Power [W]'] = df['Magnet Voltage [V]'] * df['Magnet Current [A]']
    # center smaract measured values around zero
    a1 = df.loc[:, 'SmarAct_Meas_Angle_1'].copy().values
    a2 = df.loc[:, 'SmarAct_Meas_Angle_2'].copy().values
    a1[a1 > 180.] = a1[a1 > 180.] - 360.
    a2[a2 > 180.] = a2[a2 > 180.] - 360.
    df.loc[:, 'SmarAct_Meas_Angle_1_Centered'] = a1
    df.loc[:, 'SmarAct_Meas_Angle_2_Centered'] = a2
    ### INTERPOLATE NMR VALUES TO SCAN TIMES
    #### STOPPED HERE 2022-11-29 17:41:36 
    return df

def match_temp_scan_dfs(df_temp, df):
    # assumes scan df is a subset of temp df.
    t0 = df.index[0]
    tf = df.index[-1]
    df_t = df_temp.query(f'"{t0}" <= Datetime <= "{tf}"').copy()
    return df_t

def interp_temp_col(df_temp, df, col='NMR [T]'):
    interp_func = interp1d(df_temp.seconds_delta, df_temp[col].values, kind='linear', fill_value='extrapolate')
    df.loc[:, col] = interp_func(df.seconds_delta)
    return df

#### plots
def make_run_plot_with_temp(df, df_temp, probe):
    df_ = df.copy()
    df_t_ = df_temp.copy()
    
    fig = plt.figure(figsize=(14, 20))
    ax = plt.axes([0.1, 0.6, 0.8, 0.3])
    ax2 = ax.twinx()
    ax3 = plt.axes([0.1, 0.35, 0.8, 0.25])
    ax4 = ax3.twinx()
    ax5 = plt.axes([0.1, 0.1, 0.8, 0.25])
    # turn off ticks
    ax.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    # mean lines
    l1, = ax2.plot([df_t_.index.min(), df_t_.index.max()], 2*[df_t_['NMR [T]'].mean()], 'k--', label='mean(B)')
    #ax.plot([df_t_.index.min(), df_t_.index.max()], 2*[df_t_['NMR [T]'].mean()], 'k--', label='mean(B)')
    l2 = ax2.fill_between([df_t_.index.min(), df_t_.index.max()], df_t_['NMR [T]'].mean() - 1e-4, df_t_['NMR [T]'].mean() + 1e-4,
                            color='green', alpha=0.1, label=r'mean(B)$\pm$1e-4')
    # main data
    a3 = df_[f'{probe}_Cal_Bmag'].plot(ax=ax, color='blue', linestyle='None', marker='.', label='Hall Probe |B| [T]')
    l3 = a3.get_legend_handles_labels()[0][-1]
    # df_[f'{probe}_Raw_Bmag'].plot(ax=ax, color='blue')
    a4 = df_t_['NMR [T]'].plot(ax=ax2, color='red', linestyle='None', marker='.', label='NMR |B| [T]')
    l4 = a4.get_legend_handles_labels()[0][-1]
    #df_['SmarAct_Meas_Angle_2_Centered'].plot(ax=ax3, color='black', linestyle='None', marker='.')
    #df_['SmarAct_Meas_Angle_1_Centered'].plot(ax=ax4, color='green', linestyle='None', marker='.')
    aa1 = df_['SmarAct_Meas_Angle_2'].plot(ax=ax3, color='black', linestyle='None', marker='.', label='Angle 2')
    ll1 = aa1.get_legend_handles_labels()[0][-1]
    aa2 = df_['SmarAct_Meas_Angle_1'].plot(ax=ax4, color='green', linestyle='None', marker='.', label='Angle 1')
    ll2 = aa2.get_legend_handles_labels()[0][-1]
    tt1 = df_[f'{probe}_Cal_T'].plot(ax=ax5, color='magenta', linestyle='None', marker='.', label=f'{probe}_Cal_T')
    lll1 = tt1.get_legend_handles_labels()[0][-1]
    #tt2 = df_t_[f'Parameter HVAC sensor'].plot(ax=ax5, color='cyan', linestyle='None', marker='.', 
    #                                           markersize=12, label=f'Parameter HVAC sensor')
    #lll2 = tt2.get_legend_handles_labels()[0][-1]
    tt3 = df_t_[f'Hall Element'].plot(ax=ax5, color='brown', linestyle='None', marker='.', 
                                               markersize=12, label=f'Hall Element')
    lll3 = tt3.get_legend_handles_labels()[0][-1]
    
    ax.set_ylabel(f'Hall probe |B| [T]')
    ax.yaxis.label.set_color('blue')
    ax2.set_ylabel(f'NMR |B| [T]')
    ax2.yaxis.label.set_color('red')
    #ax3.set_ylabel('SmarAct Angle 2 [Rad], centered')
    #ax4.set_ylabel('SmarAct Angle 1 [Rad], centered')
    ax3.set_ylabel('SmarAct Angle 2 [deg]')
    ax4.set_ylabel('SmarAct Angle 1 [deg]')
    ax3.yaxis.label.set_color('black')
    ax4.yaxis.label.set_color('green')
    #ax5.set_ylabel(f'{probe}_Cal_T [deg C]')
    ax5.set_ylabel(f'Temperature [deg C]')
    ax.set_title(f"|B| Comparisons for Hall Probe: {probe}\n")
    ax.set_xlim([df_t_.index.min() - timedelta(seconds=120), df_t_.index.max() + timedelta(seconds=120)])
    ax3.set_xlim([df_t_.index.min() - timedelta(seconds=120), df_t_.index.max() + timedelta(seconds=120)])
    ax5.set_xlim([df_t_.index.min() - timedelta(seconds=120), df_t_.index.max() + timedelta(seconds=120)])
    ax.set_ylim([df_[f'{probe}_Cal_Bmag'].mean() - 2e-4, df_[f'{probe}_Cal_Bmag'].mean() + 2e-4])
    ax2.set_ylim([df_t_['NMR [T]'].mean() - 2e-4, df_t_['NMR [T]'].mean() + 2e-4])
    ls = [l1, l2, l3, l4]
    lbs = [l.get_label() for l in ls]
    ax2.legend(ls, lbs)
    lls = [ll1, ll2]
    llbs = [l.get_label() for l in lls]
    ax3.legend(lls, llbs).set_zorder(100)
    #llls = [lll1, lll2, lll3]
    llls = [lll1, lll3]
    lllbs = [l.get_label() for l in llls]
    ax5.legend(llls, lllbs).set_zorder(100);
    axs = [ax, ax2, ax3, ax4, ax5]
    return fig, axs

def make_SmarAct_plot_plotly(df0):
    fig = go.Figure()

    # First scatter trace (uses primary y-axis)
    fig.add_trace(go.Scatter(
        x=df0.index,
        y=df0['SmarAct_Meas_Angle_1'],
        mode='markers',
        marker=dict(
            color='rgba(0, 128, 0, 1.0)',  # green with transparency
            size=4,
            symbol='circle'
        ),
        name='SmarAct_Meas_Angle_1',
        yaxis='y1'
    ))
    
    # Second scatter trace (uses secondary y-axis)
    fig.add_trace(go.Scatter(
         x=df0.index,
        y=df0['SmarAct_Meas_Angle_2'],
        mode='markers',
        marker=dict(
            color='rgba(0, 0, 0, 0.4)',  # black with transparency
            size=4,
            symbol='x'
        ),
        name='SmarAct_Meas_Angle_2',
        yaxis='y2'
    ))
    
    # Set up layout with two y-axes
    fig.update_layout(
        title='Dual Y-Axis Angular Measurements',
        xaxis=dict(title='Time'),
        yaxis=dict(title='SmarAct_Meas_Angle_1 (deg)', side='left'),
        yaxis2=dict(
            title='SmarAct_Meas_Angle_2 (deg)',
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.01, y=0.99),
        height=1000,  # increase height
        width=1000    # optional: set width
    )
    
    fig.show()
    return fig

def make_temps_plot(df_temp0, df0, probe, slowfile, plot_Hall_temp=True):
    # temps = ['Coil 1', 'Coil 2']
    #temps = ['Coil 1', 'Coil 2', 'LCW in Coil1', 'LCW in Coil 2', 'LCW out Coil 1', 'LCW out Coil 2']
    temps = ['Coil 1', 'Coil 2', 'LCW in Coil1', 'LCW in Coil 2', 'LCW out Coil 1', 'LCW out Coil 2',
         'Supply Magnet (NWC-S)', 'Return Magnet (NWC-R)']
    fig, ax = plt.subplots(figsize=(14, 12))
    for t in temps:
        vals = df_temp0.loc[:, t]
        ax.plot(df_temp0.index, vals, label=t)
    
    ax.plot(df_temp0.index, df_temp0['Parameter HVAC sensor'], label='Parameter HVAC sensor')
    if plot_Hall_temp:
        # Hall probe temp
        # FIXME! Only use times with NMR
        ax.scatter(df0.index, df0[f'{probe}_Cal_T'], label=f'{probe}_Cal_T', s=5)
    
    ax.fill_between([df_temp0.index[0], df_temp0.index[-1]], 45, 50, color='yellow', alpha=0.25, label='Warning! Overtemperature possible...')
    ax.fill_between([df_temp0.index[0], df_temp0.index[-1]], 50, 60, color='red', alpha=0.25, label='DANGER! OVERTEMPERATURE IMMINENT!')
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0))
    #ax.set_ylim([10.0, 40.0])
    #ax.set_ylim([10.0, 60.0])
    ax.set_ylim([0.0, 60.0])
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    #ax.set_yticks(np.arange(10, 42, 2))
    ###ax.set_yticks(np.arange(10, 62, 2))
    ax.set_yticks(np.arange(0, 62, 2))
    ax.tick_params(labelleft=True, labelright=True)
    ax.set_ylabel("Temperature (deg C)")
    ax.set_title(slowfile.split('/')[-1])
    return fig, ax