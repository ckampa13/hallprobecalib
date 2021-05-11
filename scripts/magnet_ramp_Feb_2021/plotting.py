import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator
from datetime import datetime

# nicer plot formatting
def config_plots():
    #must run twice for some reason (glitch in Jupyter)
    for i in range(2):
        plt.rcParams['figure.figsize'] = [10, 8] # larger figures
        plt.rcParams['axes.grid'] = True         # turn grid lines on
        plt.rcParams['axes.axisbelow'] = True    # put grid below points
        plt.rcParams['grid.linestyle'] = '--'    # dashed grid
        plt.rcParams.update({'font.size': 14.0})   # increase plot font size
        plt.rcParams.update({"text.usetex": True})

# datetime plot format for matplotlib
def datetime_plt(ax, x_dt, y, s=5, label=None, nmaj=8):
    ax.plot_date(x_dt, y, markersize=s, label=label)
    xmax = datetime.strptime(np.max(x_dt), '%Y-%m-%d %H:%M:%S')
    xmin = datetime.strptime(row.t0,'%Y-%m-%d %H:%M:%S')
    run_hours = (xmax-xmin).total_seconds() / (60**2)
    interv = int(run_hours // nmaj)
    if interv == 0:
        interv = 1
    hours = HourLocator(interval=interv)
    ax.xaxis.set_major_locator(hours)
    formatter = DateFormatter('%m-%d %H:%M')
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_tick_params(rotation=30)

    return ax

def ticks_in(ax):
    ax.tick_params(which='both', direction='in')
    return ax

# label for histogram
def get_label(data, bins):
    over = (data > np.max(bins)).sum()
    under = (data < np.min(bins)).sum()
    data_ = data[(data <= np.max(bins)) & (data >= np.min(bins))]
    mean = f'{np.mean(data_):.3E}'
    std = f'{np.std(data_, ddof=1):.3E}'
    label = f'mean: {mean:>15}\nstddev: {std:>15}\nIntegral: {len(data):>17}\n'\
    +f'Underflow: {under:>16}\nOverflow: {over:>16}'
    return label
