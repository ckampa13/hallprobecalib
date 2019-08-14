import numpy as np
import pickle as pkl
import pandas as pd

import os

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

#import copy
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go

from mu2e import mu2e_ext_path

from hallprobecalib.hpcplots import scatter2d,scatter3d,histo

# try:
#     get_ipython
def get_trajectory_widget():
    data_dir = f"{mu2e_ext_path}trajectory/run05/"
    plot_dir = f"{mu2e_ext_path}plots/trajectory/run05/"

    # stopping target
    stop_zs = np.linspace(5500.,6300.,34)
    r = 75.
    r_in = 25.
    phis = np.linspace(0,2*np.pi,50)
    xx = r*np.cos(phis)
    yy = r*np.sin(phis)
    x_in = r_in*np.cos(phis)
    y_in = r_in*np.sin(phis)

    # tracker
    trk_zs = np.linspace(8410.,11660.,18)
    r_tr = 700.
    r_tr_in = 400.
    x_tr = r_tr*np.cos(phis)
    x_tr_in = r_tr_in*np.cos(phis)
    y_tr = r_tr*np.sin(phis)
    y_tr_in = r_tr_in*np.sin(phis)

    fig_geom = go.Figure()
    for z_s in stop_zs:
        zst = np.ones_like(phis)*z_s
        fig_geom.add_mesh3d(
            x=xx,
            y=yy,
            z=zst,
            opacity=0.3,
            color='red',
        )
        fig_geom.add_mesh3d(
            x=x_in,
            y=y_in,
            z=zst,
            opacity=0.3,
            color='white',
        )
        for z_t in trk_zs:
            ztr = np.ones_like(phis)*z_t
            fig_geom.add_mesh3d(
                x=x_tr,
                y=y_tr,
                z=ztr,
                opacity=0.15,
                color='grey',
            )
            fig_geom.add_mesh3d(
                x=x_tr_in,
                y=y_tr_in,
                z=ztr,
                opacity=0.15,
                color='white',
            )

    # subruns = sorted([i for i in os.listdir(data_dir+'raw') if "subrun" in i])


    # def invoke_widget_traj():
        # @interact
        def trajectory_plot(run=run, subrun=None ,event=0, inline=True):
            # run = "run05"
            # data_dir = mu2e_ext_path+f"trajectory/{run}/sparse/{subrun}/"
            files = os.listdir(data_dir)
            file = [f for f in files if f"{event}" in f][0]

            df_out = pd.read_pickle(data_dir+file)
            df_out = pd.concat([df_out[::2],df_out.tail(1)],axis=0)
            if df_out.Z.min() > 5500.:
                zm = 5500.
            else:
                zm = df_out.Z.min()
            fig = scatter3d([df.X for df in [df_out]],
                [df.Y for df in [df_out]],
                [df.Z for df in [df_out]],
                scale_list=[df.time for df in [df_out]],
                mode_list=['markers+lines'],units_list=[('mm','mm','mm')],inline=True,
                colors_list = ["Viridis"], aspect_auto=False, show_plot=False,
                rangex=(700+700),rangey=(700+700),rangez=(12000-zm),opacity_list=[1.],
                fig_ = fig_geom, copy_fig = True,
                            );
            camera = dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=-0.3),
                eye=dict(x=-3.6, y=1., z=-0.5)#(x=-5,y=1.5,z=-0.25)#(x=-1.8, y=0.2, z=-0.1)
            )

            fig.layout.showlegend = False
            fig.layout.dragmode = 'orbit'
            fig.layout.scene.camera = camera
            fig.layout.title = f"Run: {run[-2:]}, Subrun: {subrun[-2:]}, Event: {event}"

            if show_plot:
                if inline:
                    iplot(fig)
                else:
                    plot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')

            return fig
