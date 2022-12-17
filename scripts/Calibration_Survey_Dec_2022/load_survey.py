import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def inch_to_m(inch):
    return 0.0254 * inch

def inch_to_mm(inch):
    return 25.4 * inch

def mm_to_inch(inch):
    return inch/25.4

def load_group_df(filename, df_slow):
    df = pd.read_csv(filename)
    # datetime
    df.loc[:,'Datetime'] = pd.to_datetime(df.Time)
    df.sort_values(by=['Datetime'], inplace=True)
    df = df.set_index('Datetime')
    # seconds delta, on the same time frame as df_slow
    df.loc[:,'seconds_delta'] = (df.index - df_slow.index[0]).total_seconds()+df_slow['seconds_delta'].iloc[0]
    # store whether the stage was removed before this measurement
    removed = []
    for row in df.itertuples():
        #if ("Repeat" in row.Group) and (not "Check" in row.Group) and (not "Test" in row.Group) and (not "Vertical" in row.Group):
        if ("Repeat" in row.Group) and (not "Check" in row.Group) and (not "Test" in row.Group) and (not "26th Installation Home Ideal" in row.Group):
            removed.append(True)
        else:
            removed.append(False)
    df.loc[:, 'Stage_Removed'] = removed
    # interpolation for stage locations
    SmarAct1_interp = interp1d(df_slow.seconds_delta.values, df_slow['SmarAct_Meas_Angle_1'].values, kind='nearest')
    SmarAct2_interp = interp1d(df_slow.seconds_delta.values, df_slow['SmarAct_Meas_Angle_2'].values, kind='nearest')
    df.loc[:, 'SmarAct1_Meas_Deg'] = SmarAct1_interp(df.seconds_delta.values)
    df.loc[:, 'SmarAct2_Meas_Deg'] = SmarAct2_interp(df.seconds_delta.values)
    df.loc[:, 'SmarAct1_Meas_Rad'] = np.radians(df['SmarAct1_Meas_Deg'])
    df.loc[:, 'SmarAct2_Meas_Rad'] = np.radians(df['SmarAct2_Meas_Deg'])
    # group index
    df.loc[:, 'Group_Index'] = [i for i in range(len(df))]
    return df

def load_group(ind_df_group, row_df_group, ddir):
    filename = ddir+row_df_group.Group+'.txt'
    df = pd.read_csv(filename, skiprows=5, names=['Point', 'X', 'Y', 'Z'])
    for i in ['X', 'Y', 'Z']:
        df.loc[:, f'{i}_mm'] = inch_to_mm(df.loc[:, i])
    df.loc[:, 'Point'] = [i.replace('Kinematic-0.5_','') for i in df.loc[:, 'Point']]
    df.loc[:, 'Group'] = row_df_group.Group
    df.loc[:, 'Group_Index'] = row_df_group.Group_Index
    df.loc[:, 'Datetime'] = ind_df_group
    # check whether plane analysis will work
    if ("D" in df.Point.values) and (len(df) >= 3):
        can_analyze = True
    else:
        can_analyze = False
    if len(df) < 5:
        all_targets = False
    else:
        all_targets = True
    return df, can_analyze, all_targets

def construct_meas_df(df_groups, ddir):
    # assumes group/time dataframe has been loaded (see function above)
    # container for output dataframes
    df_meas = []
    can_analyze_list = []
    all_targets_list = []
    # loop through each group
    for ind, row in df_groups.iterrows():
        df_, can_analyze_, all_targets_ = load_group(ind, row, ddir)
        df_meas.append(df_)
        can_analyze_list.append(can_analyze_)
        all_targets_list.append(all_targets_)
    # concatenate
    df_meas = pd.concat(df_meas)
    # add can analyze to df_groups
    df_groups.loc[:, 'can_analyze'] = can_analyze_list
    df_groups.loc[:, 'all_targets'] = all_targets_list
    return df_meas, df_groups

#### analysis steps
def plane_fit_SVD(df):
    # collect positional points from dataframe
    points = df[['X','Y','Z']].values.T
    # subtract centroid
    centroid = (np.sum(points,axis=1) / len(df))
    points_c = points - centroid[:,None]
    # calculate svd
    u, _, _ = np.linalg.svd(points_c)
    # normal vector is left singular vector with least singular value
    norm_v = u[:,2]
    norm_v = norm_v/np.linalg.norm(norm_v)
    return norm_v, centroid

# circle fit
# 2D (from emtracks)
'''
def calc_R(xc, yc, x, y):
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def circ_alg_dist(center, x, y):
    Ri = calc_R(*center, x, y)
    return Ri - Ri.mean()

def reco_circle(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_est = x_m, y_m
    center_fit, ier = optimize.leastsq(circ_alg_dist, center_est, args=(x, y))
    Ri_fit = calc_R(*center_fit, x, y)
    R_fit = np.mean(Ri_fit)
    R_residual = np.sum((Ri_fit - R_fit)**2)
    return center_fit, R_fit, Ri_fit, R_residual

# cent, R_guess, Ri_fit, R_residual = reco_circle(track_data_rot[1], track_data_rot[2])
# C_x_guess, C_y_guess = cent
'''

#### analysis driver
#def analyze_group(ind_group, df_groups, df_meas):
def analyze_group_plane(ind_group, df_meas):
    #df_g = df_groups.query(f'Group_Index == {ind_group}').iloc[0]
    df_m = df_meas.query(f'Group_Index == {ind_group}')
    # plane fit
    norm_v, centroid = plane_fit_SVD(df_m)
    # force "x" of normal vector to be >0.
    # note this normal vector will define "z" of kinematic plate.
    if norm_v[0] < 0:
        norm_v = -1 * norm_v
    # determine "y" vector in kinematic coordinates (up on the Hall card)
    point_D_ = df_m.query('Point == "D"').iloc[0]
    point_D = np.array([point_D_.X, point_D_.Y, point_D_.Z])
    vec_y = centroid - point_D
    # normalize
    vec_y = vec_y / np.linalg.norm(vec_y)
    # intuit "x" vector from other two vectors
    vec_x = np.cross(vec_y, norm_v)
    return {'norm_v': norm_v, 'norm_v_x': norm_v[0], 'norm_v_y': norm_v[1], 'norm_v_z': norm_v[2],
            'centroid': centroid, 'centroid_x': centroid[0], 'centroid_y': centroid[1], 'centroid_z': centroid[2],
            'vec_y': vec_y, 'vec_y_x': vec_y[0], 'vec_y_y': vec_y[1], 'vec_y_z': vec_y[2],
            'vec_x': vec_x, 'vec_x_x': vec_x[0], 'vec_x_y': vec_x[1], 'vec_x_z': vec_x[2],
           }

def analyze_all_groups_plane(df_groups, df_meas):
    results = {
        'norm_v': [], 'norm_v_x': [], 'norm_v_y': [], 'norm_v_z': [],
        'centroid': [], 'centroid_x': [], 'centroid_y': [], 'centroid_z': [],
        'vec_y': [], 'vec_y_x': [], 'vec_y_y': [], 'vec_y_z': [],
        'vec_x': [], 'vec_x_x': [], 'vec_x_y': [], 'vec_x_z': [],
              }
    group_inds = df_groups.Group_Index.values
    for gid in group_inds:
        # analyze group
#         new_result = analyze_group_plane(gid, df_groups, df_meas)
        new_result = analyze_group_plane(gid, df_meas)
        # add to final results
        for k, v in new_result.items():
            results[k].append(v)
    # add into groups dataframe
    print(f'Groups: {len(df_groups)}')
    for k, v in results.items():
        print(f'column: {k}, len of results: {len(v)}')
        df_groups.loc[:, k] = v
    # check angle between norm vector and the first norm vector
    # and distance between centroid and first centroid
    centroid_0 = df_groups.iloc[0].centroid
    norm_v_0 = df_groups.iloc[0].norm_v
    dcentroid = []
    dnorm_v = []
    for gid in group_inds:
        row = df_groups.query(f'Group_Index == {gid}').iloc[0]
        centroid_ = row.centroid
        dcentroid_ = np.linalg.norm(centroid_-centroid_0)
        dcentroid.append(dcentroid_)
        norm_v_ = row.norm_v
        dnorm_v_ = np.arccos(np.dot(norm_v_0, norm_v_))
        dnorm_v.append(dnorm_v_)
    df_groups.loc[:, 'dnorm_v_Rad'] = dnorm_v
    df_groups.loc[:, 'dnorm_v_Deg'] = np.degrees(dnorm_v)
    df_groups.loc[:, 'dcentroid'] = dcentroid
    return df_groups