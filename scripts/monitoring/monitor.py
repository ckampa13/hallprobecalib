import sys
import os
import argparse
import time
import datetime
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from pandas.plotting import register_matplotlib_converters
# local imports
from plotting import config_plots, get_label
from load_slow import *
from configs import *
from email_funcs import daily_email, NMR_error_email, Magnet_Current_error_email, DataStream_error_email

register_matplotlib_converters()
config_plots()


# functions
def make_temps_dict(df):
    # construct temps_dict
    # only in the chamber
    temps_ch = [a for a in df.columns if ("CH" in a) and int(a[-2:])<21]
    temps_ch.append("Hall Element")
    temps_ch.append("Floor")
    temps_ch.append("Roof")
    temps_ch.append("Yoke (center magnet)")
    temps_ch.append("Yoke (near pole)")
    temps_ch.append("Parameter HVAC sensor")# only in the chamber
    # liquids and coils
    temps_liq = [a for a in df.columns if ("LCW" in a) or ("Coil" in a) or ("NW" in a) or ("FNAL" in a)]
    # all other temps
    temps_other = []
    for a in df.columns:
        if (a in temps_ch) or (a in temps_liq):
            pass
        else:
            temps_other.append(a)
    removals = ["Time", "Magnet Current [A]", "Magnet Voltage [V]", "NMR [T]", "NMR [FFT]", "seconds_delta", "hours_delta", "days_delta"]
    for i in removals:
        try:
            temps_other.remove(i)
        except:
            pass

    temps_dict = {'chamber': temps_ch, 'liquid': temps_liq, 'other': temps_other}

    return temps_dict

def make_NMR_plot(df, B0, dB):
    fig, axs = plt.subplots(2, 1, figsize=(18, 16))
    if len(df) > 1:
        axs[0].plot([df.index[0], df.index[-1]], [B0, B0], 'k-', label=f'Expected field={B0} [T]')
        axs[0].plot([df.index[0], df.index[-1]], [B0-dB, B0-dB], 'r--', label='Normal Range')
        axs[0].plot([df.index[0], df.index[-1]], [B0+dB, B0+dB], 'r--')

    axs[0].scatter(df.index, df['NMR [T]'], s=1, label='Measurements')

    axs[0].legend()

    axs[1].scatter(df.index, df['Magnet Current [A]'], s=1)

    # limits
    if len(df) > 1:
        for ax in axs:
            trange = df.index[-1] - df.index[0]
            ax.set_xlim([df.index[0]-0.05*trange, df.index[-1]+0.05*trange])

    axs[0].set_xlabel('Datetime')
    axs[1].set_xlabel('Datetime')
    axs[0].set_ylabel('NMR [T]')
    axs[1].set_ylabel('Magnet Current [A]')
    fig.suptitle('Magnetic Field Measurements & Proxies')

    return fig, axs

def make_Temps_plot(df, temps_dict):
    fig, axs = plt.subplots(3, 1, figsize=(18, 24))
    lgds = []
    for c in temps_dict['chamber']:
        axs[0].plot(df.index, df[c], label=c)

    axs[0].set_xlabel('Datetime')
    axs[0].set_ylabel('Temp [deg C]')
    axs[0].set_title('Chamber Temperatures')
    lg_ = axs[0].legend(ncol=1, bbox_to_anchor=(1., 1.), loc='upper left')
    lgds.append(lg_)
    #axs[0].legend()

    for c in temps_dict['liquid']:
        axs[1].plot(df.index, df[c], label=c)

    axs[1].set_xlabel('Datetime')
    axs[1].set_ylabel('Temp [deg C]')
    axs[1].set_title('Liquids, Chillers, and Coil Temperatures')
    lg_ = axs[1].legend(ncol=1, bbox_to_anchor=(1., 1.), loc='upper left')
    lgds.append(lg_)
    # axs[1].legend()

    for c in temps_dict['other']:
        if "Tripp" in c:
            if "HVAC" in c:
                z = 1
            else:
                z = 2
        else:
            z = 3
        axs[2].plot(df.index, df[c], label=c, zorder=z)

    axs[2].set_xlabel('Datetime')
    axs[2].set_ylabel('Temp [deg C]')
    axs[2].set_title('Miscellaneous Temperatures')
    lg_ = axs[2].legend(ncol=1, bbox_to_anchor=(1., 1.), loc='upper left')
    lgds.append(lg_)
    # axs[2].legend()

    return fig, axs, lgds

# quality checks
def check_NMR(df, B0, dB, N_fail=3):
    m_good = (df['NMR [T]'] >= B0 - dB) & (df['NMR [T]'] <= B0 + dB)
    m_bad = (~m_good)
    N_bad = m_bad.sum()
    if N_bad >= N_fail:
        NMR_pass = False
    else:
        NMR_pass = True
    return NMR_pass, N_bad, m_bad

def check_Magnet_Current(df, minI=1., N_fail=3):
    m_good = (df['Magnet Current [A]'] >= minI)
    m_bad = (~m_good)
    N_bad = m_bad.sum()
    if N_bad >= N_fail:
        Magnet_Current_pass = False
    else:
        Magnet_Current_pass = True
    return Magnet_Current_pass, N_bad, m_bad

def check_DataStream(df, delay_max=60.):
    now = datetime.datetime.now()
    min_since_data = (now - df.index[-1]).total_seconds() / 60.
    if min_since_data > delay_max:
        DataStream_pass = False
    else:
        DataStream_pass = True
    return DataStream_pass, min_since_data


if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--NMR',
                        help='Monitor NMR? "y"(default)/"n"')
    # FIXME! may want B0 to be inferred from previous 24 hours of data. Not sure yet.
    parser.add_argument('-B', '--B0',
                        help='What B field is expected? 1.0 (default)')
    parser.add_argument('-M', '--Magnet',
                        help='Monitor Magnet Current? "y"(default)/"n"')
    parser.add_argument('-T', '--Temps',
                        help='Monitor temperatures? "y"(default)/"n"')
    parser.add_argument('-S', '--Sleep',
                        help='Length of minutes to sleep between processing? 1 (default)')
    args = parser.parse_args()
    # fill defaults
    if args.NMR is None:
        args.NMR = True
    else:
        args.NMR = args.NMR.strip() == 'y'
    if args.B0 is None:
        args.B0 = 1.0
    else:
        args.B0 = float(args.B0.strip())
    if args.Magnet is None:
        args.Magnet = True
    else:
        args.Magnet = args.Magnet.strip() == 'y'
    if args.Temps is None:
        args.Temps = True
    else:
        args.Temps = args.Temps.strip() == 'y'
    if args.Sleep is None:
        args.Sleep = 1.0
    else:
        args.Sleep = float(args.Sleep.strip())
    # main loop
    first_loop = True
    NMR_email_sent = False
    Magnet_email_sent = False
    DataStream_email_freq_hours = 1.
    DataStream_email_last = datetime.datetime.now() - datetime.timedelta(hours=100)
    #daily_email_now = False
    email_sent_today = False
    email_date = datetime.date.today()
    daily_email_time = datetime.time(hour=7, minute=0)
    while True:
        print('Starting Loop!')
        # check for any file updates
        if first_loop:
            print('First loop, reading entire data file.')
            # read entire file
            df = load_data(current_slow_scan)
            temps_dict = make_temps_dict(df)
            N_new = len(df)
            first_loop = False
            new_data = True
        else:
            print('Not first loop, reading only new.')
            # load pickle file
            skiprows_extra = len(df)
            try:
                df2 = load_data(current_slow_scan, skiprows_extra=skiprows_extra)
            except:
                df2 = []
            # read only the new parts
            N_new = len(df2)
            if N_new == 0:
                new_data = False
            else:
                new_data = True
                df = pd.concat([df, df2])
        # save dataframe
        df.to_pickle(monitor_ddir+'current.pkl')
        print(df.tail())
        # check if we need to do a daily email
        if not email_sent_today:
            # check if it's after 7am, if yes send the email!
            crossed_email_time = (datetime.datetime.now() - datetime.datetime.combine(datetime.date.today(), daily_email_time)).total_seconds() > 0.
            if crossed_email_time:
                print('Generating daily email summary...')
                # query the right data
                tf = datetime.datetime.combine(datetime.date.today(), daily_email_time)
                t0 = tf - datetime.timedelta(days=1)
                df_ = df.query(f'"{t0}" <= Datetime <= "{tf}"')
                figs = []
                lgds_list = []
                if args.NMR:
                    # FIXME! make dB an arg?
                    fig_, axs_ = make_NMR_plot(df_, args.B0, dB=1e-4)
                    NMR_pass, N_bad, m_bad = check_NMR(df_, args.B0, dB=1e-4, N_fail=3)
                    if N_bad > 0:
                        axs_[0].scatter(df.index, df['NMR [T]'], s=10, c='red', marker='x', label='Measurements (Quality=False)')
                        axs_[0].legend()
                    figs.append(fig_)
                    lgds_list.append(None)
                if args.Temps:
                    fig_, axs_, lgds_ = make_Temps_plot(df_, temps_dict)
                    figs.append(fig_)
                    lgds_list.append(lgds_)
                # make picture book
                # check if directory exists
                # year
                ddir_year = monitor_pdir+f'daily/{email_date.year}/'
                ddir_month = monitor_pdir+f'daily/{email_date.year}/{email_date.month:0>2}/'
                for ddir in [ddir_year, ddir_month]:
                    if not os.path.exists(ddir):
                        os.makedirs(ddir)
                pdf = matplotlib.backends.backend_pdf.PdfPages(ddir_month+f'{email_date}_HPC_Picturebook.pdf')
                for fig, lgds in zip(figs, lgds_list):
                    pdf.savefig(fig, bbox_extra_artists=lgds, bbox_inches='tight')
                pdf.close()
                # send the email!
                daily_email(email_date)
                email_sent_today = True
                print('Email sent!')
        # reset when it's a new day
        if datetime.date.today() != email_date:
            email_sent_today = False
            email_date = datetime.date.today()
            # reset any warning emails
            NMR_email_sent = False
            Magnet_email_ent = False
        # only do other things if there's new data
        if new_data:
            # make recent dataframe
            now = datetime.datetime.now()
            df_rec = df.query(f'Datetime >= "{now - datetime.timedelta(hours=2)}"')
            # make any plots
            for df_, suff in zip([df, df_rec], ['_full', '_recent']):
                figs = []
                lgds_list = []
                if args.NMR:
                    # FIXME! make dB an arg?
                    fig_, axs_ = make_NMR_plot(df_, args.B0, dB=1e-4)
                    NMR_pass, N_bad, m_bad = check_NMR(df_, args.B0, dB=1e-4, N_fail=3)
                    if N_bad > 0:
                        axs_[0].scatter(df.index, df['NMR [T]'], s=10, c='red', marker='x', label='Measurements (Quality=False)')
                        axs_[0].legend()
                    figs.append(fig_)
                    lgds_list.append(None)
                if args.Temps:
                    fig_, axs_, lgds_ = make_Temps_plot(df_, temps_dict)
                    figs.append(fig_)
                    lgds_list.append(lgds_)
                # make picture book
                pdf = matplotlib.backends.backend_pdf.PdfPages(monitor_pdir+f"picturebook{suff}.pdf")
                for fig, lgds in zip(figs, lgds_list):
                    pdf.savefig(fig, bbox_extra_artists=lgds, bbox_inches='tight')
                pdf.close()
            # send any emails
            if args.NMR:
                NMR_pass, N_bad, m_bad = check_NMR(df_rec, args.B0, dB=1e-4, N_fail=3)
                if not NMR_pass:
                    if NMR_email_sent:
                        print('Please check NMR data! (Email already sent)')
                    else:
                        # email!
                        print('ERROR (sending email)! Please check NMR data!')
                        NMR_error_email(datetime.date.today(), df, m_bad)
                        NMR_email_sent = True
            if args.Magnet:
                Magnet_Current_pass, N_bad, m_bad = check_Magnet_Current(df_rec, minI=1., N_fail=3)
                if not Magnet_Current_pass:
                    if Magnet_email_sent:
                        print('Please check Magnet Current data! (Email already sent)')
                    else:
                        # email!
                        print('ERROR (sending email)! Please check Magnet Current data!')
                        Magnet_Current_error_email(datetime.date.today(), df_rec, m_bad)
                        Magnet_email_sent = True
            # DataStream
            DataStream_pass, min_since_data = check_DataStream(df, delay_max=60.)
            if not DataStream_pass:
                hours_since_DataStream_email = (datetime.datetime.now()-DataStream_email_last).total_seconds() / 60. / 60.
                if hours_since_DataStream_email >= DataStream_email_freq_hours:
                    DataStream_error_email(datetime.date.today(), df, min_since_data)
                    DataStream_email_last = datetime.datetime.now()
        else:
            print('No new data.')
            ### ADD CODE TO STOP IF THERE ISN'T NEW DATA FOR A FEW HOURS (EMAIL)
        # close any open plots
        plt.close('all')
        # hold for some amount of time
        print(f'Waiting for {args.Sleep} minutes...')
        time.sleep(args.Sleep * 60)
