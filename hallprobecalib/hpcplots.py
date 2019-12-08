import numpy as np
import copy
import pandas as pd
from hallprobecalib import hpc_ext_path
from plotly import tools
import plotly.graph_objs as go
import plotly.figure_factory as FF
from plotly.offline import plot, iplot
import plotly.express as px
from ROOT import TProfile, gDirectory
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# init_notebook_mode(connected=True)

plotly_colors = ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
                 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
                 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
                 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
                 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
                 'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
                 'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
                 'orrd', 'oryel', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg',
                 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor',
                 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy',
                 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral',
                 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose',
                 'tempo', 'temps', 'thermal', 'tropic', 'turbid','twilight',
                 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd']

def scatter2d(x_list, y_list, colors_list=None, colorscale_list=None, size_list=None, lines=True, width=2, markers=False, title=None, filename=None, show_plot=True,legend_list=None, inline=True):
    traces = []

    if type(x_list) != list:
        x_list = [x_list]
    if type(y_list) != list:
        y_list = [y_list]

    if colors_list is not None:
        markers=True
        lines=False
        if type(colors_list) is not list:
            colors_list = [colors_list]
        if colorscale_list is None:
            colorscale_list = plotly_colors[:len(colors_list)]
            # colorscale_list = len(colors_list)*["Viridis"]

    if size_list == None:
        size_list = [2 for i in x_list]
    else:
        if type(size_list) != list:
            size_list = [size_list]

    if lines:
        if markers:
            mode = 'lines+markers'
        else:
            mode = 'lines'
    else:
        mode = 'markers'

    if title == None:
        try:
            title = f'{y_list[0].name} vs. {x_list[0].name}'
        except:
            title = f'y vs. x'
    else:
        try:
            title = title + f' {y_list[0].name} vs. {x_list[0].name}'
        except:
            title = title

    for idx in range(len(x_list)):
        x = x_list[idx]
        y = y_list[idx]
        size = size_list[idx]
        if legend_list == None:
            try:
                name = y.name
            except:
                name = 'y'
        else:
            name = legend_list[idx]
        if colors_list is None:
            marker_ = dict(size=size)
        else:
            marker_ = dict(
                    size=size,
                    cmax=colors_list[idx].max(),
                    cmin=colors_list[idx].min(),
                    color=colors_list[idx],
                    colorbar=dict(
                        title=colors_list[idx].name+"<br>",
                        x=1+0.1*idx,
                    ),
                    colorscale=colorscale_list[idx])
        traces.append(
        go.Scatter(
            x = x,
            y = y,
            mode = mode,
            marker=marker_,
            line=dict(
                width=width,
            ),
            name = name,
        )
        )
    data = traces

    try:
        xtitle = x_list[0].name
    except:
        xtitle = "x"

    try:
        ytitle = y_list[0].name
    except:
        ytitle = "y"

    layout = go.Layout(
        title=title,
        xaxis= dict(
            title=xtitle
        ),
        yaxis=dict(
            title=ytitle
        ),
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)

    if colors_list is not None:
        fig.update_layout(legend=dict(y=1.2))

    if filename == None:
        filename = 'scatter2d_DEFAULT'

    if show_plot == True:
        if inline:
            iplot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')
        else:
            plot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')

    return fig


def scatter3d(x_list, y_list, z_list, scale_list=None,scale_unit='ns', mode_list=None, units_list=None, colors_list=None, colorbars_list=None, min_color_list=None, max_color_list=None, same_color=False, opacity_list=None, size_list=None, reverse_scale=False, aspect_auto = True, rangex=None, rangey=None, rangez=None ,inline=True, title=None, filename=None, show_plot=True, fig_=None,copy_fig=False):
    '''
    units_list = [('mm','mm','K'),('T','C','cm')] as an example. List contains 3 element tuples for (x units,y units, z units)
    '''
    traces = []
    if type(x_list) != list:
        x_list = [x_list]
    if type(y_list) != list:
        y_list = [y_list]
    if type(z_list) != list:
        z_list = [z_list]

    if scale_list == None:
        scale_list = z_list

    if mode_list == None:
        mode_list = ['markers' for i in z_list]

    if colorbars_list == None:
        colorbars_list = len(x_list) * [True]

    if min_color_list == None:
        min_color_list = [s.min() for s in scale_list]
        max_color_list = [s.max() for s in scale_list]

    if opacity_list == None:
        opacity_list = [0.8 for i in x_list]
    else:
        if type(opacity_list) != list:
            opacity_list = [opacity_list]
    if size_list == None:
        size_list = [2 for i in x_list]
    else:
        if type(size_list) != list:
            size_list = [size_list]

    if units_list == None:
        units_list = [("mm","mm","T") for j in x_list]
    else:
        if type(units_list) != list:
            units_list = [units_list]
    if type(units_list[0]) != tuple:
        units_list = [tuple(["mm" for i in range(3)]) for j in x_list]

    if colors_list == None:
        colors_list = ['Viridis' for i in x_list]
    else:
        if type(colors_list) != list:
            colors_list = [colors_list]
    num_colorbars = 0

    for idx in range(len(x_list)):
        opacity = opacity_list[idx]
        size = size_list[idx]
        units = units_list[idx]
        # PHI = np.radians(x_list[idx])
        # THETA = np.radians(y_list[idx])
        # R_col = z_list[idx]
        # if absval:
            # R = R_col.abs()
            #R = R_col.abs()
            #R = z_list[idx].abs()     # SHOULD THIS ALWAYS BE ABS?
        # else:
            # R = R_col
            #R = z_list[idx]
        X = x_list[idx]
        Y = y_list[idx]
        Z = z_list[idx]
        C = scale_list[idx]
        # R * np.sin(THETA) * np.cos(PHI)
        # Y = R * np.sin(THETA) * np.sin(PHI)
        # Z = R * np.cos(THETA)

        mode = mode_list[idx]

        min_color = min_color_list[idx]
        max_color = max_color_list[idx]

        colorbars = colorbars_list[idx]
        if colorbars == False:
            showscale = False
        else:
            showscale = True
            num_colorbars += 1

        name = (f'<br>x: {X.name} ({units[0]})<br>'
                f'y: {Y.name} ({units[1]})<br>'
                f'z: {Z.name} ({units[2]})<br>')

        traces.append( go.Scatter3d(
                                    x=X,
                                    y=Y,
                                    z=Z,
                                    name=name,
                                    mode=mode,
                                        marker=dict(
                                            size=size,
                                            color=C,                # set color to an array/list of desired values
                                            colorscale=colors_list[idx],   # choose a colorscale
                                            cauto = False,
                                            cmin = min_color,
                                            cmax = max_color,
                                            reversescale = reverse_scale,
                                            opacity=opacity,
                                            colorbar=dict(thickness=20, title=f'{C.name} ({scale_unit})',
                                                          x=-0.1*num_colorbars,y=0.5),
                                            showscale = showscale,
                                        )
                                    )
                     )

    data = traces

    # if title == None:
    #     title = f'3D Scatter Spherical Coord)'
    # else:
    #     title = title + f' 3D Scatter (Spherical Coord)'

    if title == None:
        title = f'{z_list[0].name} vs. {x_list[0].name}, {y_list[0].name}'
    else:
        title = title + f' {z_list[0].name} vs. {x_list[0].name}, {y_list[0].name}'

    xmin = min([x.min() for x in x_list])
    ymin = min([x.min() for x in y_list])
    zmin = min([x.min() for x in z_list])
    xmax = min([x.max() for x in x_list])
    ymax = min([x.max() for x in y_list])
    zmax = min([x.max() for x in z_list])

    if rangex == None:
        rangex = xmax - xmin
    if rangey == None:
        rangey = ymax - ymin
    if rangez == None:
        rangez = zmax - zmin

    if rangex > rangey:
        xratio = 1
        yratio = rangey/rangex
        zratio = rangez/rangex
    else:
        xratio = rangex/rangey
        yratio = 1
        zratio = rangez/rangey

    if aspect_auto == True:
        layout = go.Layout(
            title = title,
            showlegend=True,
            scene = dict(
                xaxis = dict(
                    title=f'{X.name} ({units_list[0][0]})'
                    ),
                yaxis = dict(
                    title=f'{Y.name} ({units_list[0][1]})'
                    ),
                zaxis = dict(
                    title=f'{Z.name} ({units_list[0][2]})'
                    ),
            )
        )
    else:
        layout = go.Layout(
            title = title,
            showlegend=True,
            scene = dict(
                xaxis = dict(
                    title=f'{X.name} ({units_list[0][0]})'
                    ),
                yaxis = dict(
                    title=f'{Y.name} ({units_list[0][1]})'
                    ),
                zaxis = dict(
                    title=f'{Z.name} ({units_list[0][2]})'
                    ),
                aspectmode = 'manual',
                # automatically make xy axis ratio based on data...z has wide range
                aspectratio=go.layout.scene.Aspectratio(
                    x = xratio, y = yratio, z = zratio#1
                    )
                )
        )

    if fig_ == None:
        fig = go.Figure(data=data, layout=layout)
    else:
        if copy_fig:
            fig = copy.copy(fig_)
        else:
            fig = fig_
        [fig.add_trace(d) for d in data]
        fig.layout = layout

    # fig['layout'].update(go.layout.Scene(aspectmode='data'))

    if filename == None:
        filename = 'scatter3d_DEFAULT'

    if show_plot == True:
        if inline:
            iplot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')
        else:
            plot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')

    return fig


def histo(series_list, names_list=None, xlabel=None, bins=10, same_bins=False, autobin=False, opacity=0.9,cut=[0.,1.], range_x=None, yscale='linear', barmode='overlay', horizontal=False,inline=True, title=None, verbosity=2, show_plot=True,filename=None):
    traces = []

    if type(series_list) != list:
        series_list = [series_list]

    if names_list == None:
        names_list = [i.name for i in series_list]
    else:
        if type(names_list) != list:
            names_list = list(names_list)

    if xlabel == None:
        xlabel = series_list[0].name

    x_min = series_list[0].min()
    x_max = series_list[0].max()
    bin_size = (x_max - x_min) / bins

    for idx in range(len(series_list)):
        series = series_list[idx]
        if range_x != None:
            range_mask = (series > range_x[0]) & (series < range_x[1])
            series = series[range_mask]
        c = (series >= series.quantile(cut[0])) & (series <= series.quantile(cut[1]))
        series_list[idx] = series[c]
        series = series[c]
        if verbosity == 2:
            name = (f'<br>{names_list[idx]}<br>'
                    f'mean: {(series.mean()):.7E}<br>'
                    f'std:     {(series.std()):.3E}<br>'
                    f'range: {(series.max()-series.min()):.3E}<br>'
                    # f'min:     {(series.min()-series.mean()):.3E}<br>'
                    # f'max:     {(series.max()-series.mean()):.3E}<br>'
                    f'count: {len(series)}')
        elif verbosity == 1:
            name = (f'<br>{names_list[idx]}<br>'
                    f'mean: {(series.mean()):.7E}<br>'
                    f'std:     {(series.std()):.3E}<br>')
        else:
            name = (f'<br>{names_list[idx]}<br>')

        if not horizontal:
            if autobin:
                traces.append(go.Histogram(x=series, nbinsx=bins, opacity=opacity,name=name))
            else:
                if not same_bins:
                    x_min = series.min()
                    x_max = series.max()
                    bin_size = (x_max - x_min) / bins
                traces.append(
                    go.Histogram(
                        x=series,
                        xbins=dict(
                            start=x_min,
                            end=x_max,
                            size=bin_size,
                        ),
                        autobinx=False,
                        opacity=opacity,
                        name=name
                    )
                )
        else:
            if autobin:
                traces.append(go.Histogram(y=series, nbinsy=bins, opacity=opacity,name=name))
            else:
                if not same_bins:
                    x_min = series.min()
                    x_max = series.max()
                    bin_size = (x_max - x_min) / bins
                traces.append(
                    go.Histogram(
                        y=series,
                        ybins=dict(
                            start=x_min,
                            end=x_max,
                            size=bin_size,
                        ),
                        autobiny=False,
                        opacity=opacity,
                        name=name
                    )
                )

    data = traces

    if title == None:
        title = f'Histo: {", ".join([i.name for i in series_list])}'
    else:
        title = title + f' Histo: {", ".join([i.name for i in series_list])}'

    if not horizontal:
        layout = go.Layout(
            title=title,
            xaxis= dict(
                title=xlabel,
                showgrid=True
            ),
            yaxis=dict(
                title='Counts',
                type=yscale,
                autorange=True,
                showgrid=True
            ),
            barmode=barmode,
            showlegend=True
        )
    else:
        layout = go.Layout(
            title=title,
            yaxis= dict(
                title=xlabel,
                showgrid=True
            ),
            xaxis=dict(
                title='Counts',
                type=yscale,
                autorange=True,
                showgrid=True
            ),
            barmode=barmode,
            showlegend=True
        )

    fig = go.Figure(data=data, layout=layout)

    if filename == None:
        filename = 'h_DEFAULT'

    if show_plot:
        if inline:
            iplot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')
        else:
            plot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')

    return fig


def py_profile(x, y, x_bins, xrange = None, show_plot=True, return_tprof=False, inline=True):
    """
    Args: x, y (Pandas Series), x_bins (int), xrange (None or 2 element list/tuple with xmin
    and xmax), show_plot (Boolean), return_tprof (Boolean for TProfile object), inline (Boolean)
    """
    if xrange==None:
        xrange= [x.min(), x.max()]

    # Generate TProfile
    # This might give problems if using ROOT and want things stored...may have to live with
    # 'potential memory leak' every time function is run
    root_list = gDirectory.GetList()
    if len(root_list) != 0:
        for obj in root_list:
            if obj.GetName() == "tprof":
                obj.Delete()
    tprof = TProfile('tprof', 'Profile Plot', x_bins, xrange[0], xrange[1])
    for xi,yi in zip(x,y):
        tprof.Fill(xi,yi)

    # Collect data into Pandas Series
    bin_centers = []
    bin_contents = []
    bin_errors = []

    for i in range(1,x_bins+1):
        bin_centers.append(tprof.GetBinCenter(i))
        bin_contents.append(tprof.GetBinContent(i))
        bin_errors.append(tprof.GetBinError(i))

    bin_centers = pd.Series(bin_centers, name=x.name)
    bin_contents = pd.Series(bin_contents, name=y.name)
    bin_errors = pd.Series(bin_errors, name="bin errors")

    x_error = tprof.GetBinWidth(1)/2

    # Plot with Plotly!
    # Legend to match ROOT TProfile
    name = (
        f'<br>{y.name}<br>'
        f'entries: {len(x)}<br>'
        f'mean: {x.mean():.3E}<br>'
        f'mean y: {y.mean():.3E}<br>'
        f'std dev: {x.std():.3E}<br>'
        f'std dev y: {y.std():.3E}'
        )

    fig = scatter2d(bin_centers, bin_contents, lines=False, markers=True, size_list=[1],
                    title="Profile Plot: ",show_plot=False, inline=True)
    fig.update_traces(
        error_x=dict(type="constant", value=x_error, width=0),
        error_y=dict(type="data", array=bin_errors, width=0, visible=True),
        name=name,
    )

    if show_plot:
        iplot(fig)

    if return_tprof:
        return fig, tprof
    else:
        return fig



def spherical_scatter3d(phi_list, theta_list, r_list, units_list=None, absval=True, colors_list=None, opacity_list=None, size_list=None,inline=True, title=None, filename=None):
    traces = []
    if type(phi_list) != list:
        phi_list = [phi_list]
    if type(theta_list) != list:
        theta_list = [theta_list]
    if type(r_list) != list:
        r_list = [r_list]

    if opacity_list == None:
        opacity_list = [0.8 for i in phi_list]
    else:
        if type(opacity_list) != list:
            opacity_list = [opacity_list]
    if size_list == None:
        size_list = [2 for i in phi_list]
    else:
        if type(size_list) != list:
            size_list = [size_list]

    if units_list == None:
        units_list = ["T" for i in phi_list]
    else:
        if type(units_list) != list:
            units_list = [units_list]

    if colors_list == None:
        colors_list = ['Viridis' for i in phi_list]
    else:
        if type(colors_list) != list:
            colors_list = [colors_list]

    for idx in range(len(phi_list)):
        opacity = opacity_list[idx]
        size = size_list[idx]
        PHI = np.radians(phi_list[idx])
        THETA = np.radians(theta_list[idx])
        R_col = r_list[idx]
        if absval:
            R = R_col.abs()
            #R = R_col.abs()
            #R = r_list[idx].abs()     # SHOULD THIS ALWAYS BE ABS?
        else:
            R = R_col
            #R = r_list[idx]
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)

        name = (f'<br>phi: {PHI.name}<br>'
                f'theta: {THETA.name}<br>'
                f'r: {R.name} ({units_list[idx]})<br>')

        traces.append( go.Scatter3d(
                                    x=X,
                                    y=Y,
                                    z=Z,
                                    name=name,
                                    mode='markers',
                                        marker=dict(
                                            size=size,
                                            color=R_col,                # set color to an array/list of desired values
                                            colorscale=colors_list[idx],   # choose a colorscale
                                            opacity=opacity,
                                            colorbar=dict(thickness=20, title=f'{R.name} ({units_list[idx]})',
                                                          x=-0.1*idx,y=0.5)
                                        )
                                    )
                     )

    data = traces

    if title == None:
        title = f'3D Scatter (Spherical Coord)'
    else:
        title = title + f' Scatter (Spherical Coord)'


    layout = go.Layout(
        title = title,
        showlegend=True
    )
    fig = go.Figure(data=data, layout=layout)

    if filename == None:
        filename = 'spherical_scatter3d_DEFAULT'

    if inline:
        iplot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')
    else:
        plot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')

    return fig
