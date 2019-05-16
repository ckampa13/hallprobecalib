import numpy as np
from hallprobecalib import hpc_ext_path
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
from plotly.offline import plot, iplot
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# init_notebook_mode(connected=True)

def scatter2d(x_list, y_list, size_list=None, lines=True, markers=False, title=None, filename=None, inline=False):
    traces = []

    if type(x_list) != list:
        x_list = [x_list]
    if type(y_list) != list:
        y_list = [y_list]

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
        title = f'{y_list[0].name} vs. {x_list[0].name}'
    else:
        title = title + f' {y_list[0].name} vs. {x_list[0].name}'

    for idx in range(len(x_list)):
        x = x_list[idx]
        y = y_list[idx]
        size = size_list[idx]
        traces.append(
        go.Scatter(
            x = x,
            y = y,
            mode = mode,
            marker=dict(
                size=size,
            ),
            name = y.name,
        )
        )

    data = traces

    layout = go.Layout(
        title=title,
        xaxis= dict(
            title=x_list[0].name
        ),
        yaxis=dict(
            title=y_list[0].name
        ),
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)

    if filename == None:
        filename = 'scatter2d_DEFAULT'

    if inline:
        iplot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')
    else:
        plot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')

    return fig


def scatter3d(x_list, y_list, z_list, scale_list=None, mode_list=None, units_list=None, colors_list=None, colorbars_list=None, min_color_list=None, max_color_list=None, same_color=False, opacity_list=None, size_list=None, reverse_scale=False, aspect_auto = True, inline=False, title=None, filename=None, show_plot=True):
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
                                            colorbar=dict(thickness=20, title=f'{C.name} ({units[2]})',
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

    rangex = xmax - xmin
    rangey = ymax - ymin
    rangez = zmax - zmin

    if rangex > rangey:
        xratio = 1
        yratio = rangey/rangex
    else:
        xratio = rangex/rangey
        yratio = 1

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
                    x = xratio, y = yratio, z = 1
                    )
                )
        )

    fig = go.Figure(data=data, layout=layout)
    # fig['layout'].update(go.layout.Scene(aspectmode='data'))

    if filename == None:
        filename = 'scatter3d_DEFAULT'

    if show_plot == True:
        if inline:
            iplot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')
        else:
            plot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')

    return fig


def histo(series_list, names_list=None, xlabel='B (T)', bins=10, opacity=0.9,cut=[0.,1.], yscale='linear', inline=False, title=None, filename=None):
    traces = []

    if type(series_list) != list:
        series_list = [series_list]

    if names_list == None:
        names_list = [i.name for i in series_list]
    else:
        if type(names_list) != list:
            names_list = list(names_list)

    for idx in range(len(series_list)):
        series = series_list[idx]
        c = (series >= series.quantile(cut[0])) & (series <= series.quantile(cut[1]))
        series_list[idx] = series[c]
        series = series[c]
        name = (f'<br>{names_list[idx]}<br>'
                f'mean: {(series.mean()):.7E}<br>'
                f'std:     {(series.std()):.3E}<br>'
                f'min:     {series.min():.3E}<br>'
                f'max:     {series.max():.3E}<br>'
                f'count: {len(series)}')
        traces.append(go.Histogram(x=series, nbinsx=bins, opacity=opacity,name=name))

    data = traces

    if title == None:
        title = f'Histo: {", ".join([i.name for i in series_list])}'
    else:
        title = title + f' Histo: {", ".join([i.name for i in series_list])}'

    layout = go.Layout(
        title=title,
        xaxis= dict(
            title=xlabel
        ),
        yaxis=dict(
            title='Counts',
            type=yscale,
            autorange=True
        ),
        barmode='overlay',
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)

    if filename == None:
        filename = 'h_DEFAULT'

    if inline:
        iplot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')
    else:
        plot(fig, filename=hpc_ext_path+'plots/'+filename+'.html')

    return fig


def spherical_scatter3d(phi_list, theta_list, r_list, units_list=None, absval=True, colors_list=None, opacity_list=None, size_list=None,inline=False, title=None, filename=None):
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