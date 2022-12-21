import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib.pyplot as plt
from collections.abc import Iterable

# local imports
from load_survey import plane_fit_SVD

class CircleFitter3D(object):
    def __init__(self,):
        self.z_axis = np.array([0., 0., 1.])
    
    def reco_circle_coope(self, x, y, param_settings):
        self.model_circ = lm.Model(mod_circle, independent_vars=['x', 'y'])
        params = lm.Parameters()
        params.add('A', value=param_settings['A0'], min=param_settings['Amin'], max=param_settings['Amax'])
        params.add('B', value=param_settings['B0'], min=param_settings['Bmin'], max=param_settings['Bmax'])
        params.add('R2', value=param_settings['R20'], min=param_settings['R2min'], max=param_settings['R2max'])
        params.add('C', expr='R2 -(A/2)**2 - (B/2)**2')
        self.result = self.model_circ.fit(x**2 + y**2, x=x, y=y, params=params)
        Xc = self.result.params['A'].value/2
        Yc = self.result.params['B'].value/2
        R = (self.result.params['R2'].value)**(1/2)
        self.result_params = {'Xc':Xc, 'Yc':Yc, 'R': R}
        return self.result_params, self.result
    
    def run_fit(self, df, flip_condition_index=0, flip_condition='<0', param_settings=None):
        # default param settings
        if param_settings is None:
            param_settings = {
                'A0': 0., 'Amin': None, 'Amax': None,
                'B0': 0., 'Bmin': None, 'Bmax': None,
                'R20': 4., 'R2min': 0., 'R2max': None,
            }
        # dataframe, and points
        self.df = df.copy()
        self.vecs = self.df[['X', 'Y', 'Z']].values
        # SVD for fitting plane
        self.norm_v, self.centroid = plane_fit_SVD(self.df)
        # flip normal vector, if in the wrong direction
        if flip_condition == '<0':
            if self.norm_v[flip_condition_index] < 0:
                self.norm_v *= -1
        elif flip_condition == '>0':
            if self.norm_v[flip_condition_index] > 0:
                self.norm_v *= -1
        elif flip_condition is None:
            pass
        else:
            raise ValueError(f'"{flip_condition}" is not a valid flip_condition. Please choose a supported flip_condition: ["<0", ">0", None]')
        # normalize
        self.norm_v = self.norm_v/np.linalg.norm(self.norm_v)
        # center points
        self.vecs_cen = self.vecs - self.centroid
        # rodrigues rotation -- project points onto best fit plane (x, y)
        self.vec_cen_rot = rodrigues_rotation(self.vecs_cen, self.norm_v, self.z_axis)
        # fit circle
        _, _ = self.reco_circle_coope(self.vec_cen_rot[:, 0], self.vec_cen_rot[:, 1], param_settings=param_settings)
        # rodrigues rotation -- rotate back to original coordinates
        self.center_fit_cen_rot = np.array([self.result_params['Xc'], self.result_params['Yc'], 0.])
        self.center_fit_cen = rodrigues_rotation(self.center_fit_cen_rot, self.z_axis, self.norm_v)[0]
        # translate back to orginal coordinates
        self.center_fit = self.center_fit_cen + self.centroid
        # calculate other parameters
        self.phi = np.arctan2(self.norm_v[1], self.norm_v[0])
        self.theta = np.arccos(self.norm_v[2])
        self.u_fit = np.array([-np.sin(self.phi), np.cos(self.phi), 0.])
        # aggregate
        self.final_results = {
            'r': self.result_params['R'],
            'n': self.norm_v,
            'u': self.u_fit,
            'C': self.center_fit,
            'theta': self.theta,
            'phi': self.phi,
        }
        return self.final_results
        
    def plot_results_3D(self, mode='mpl'):
        # must run self.run_fit first.
        if mode == 'mpl':
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            vecs_fit = make_circle_3d_from_params(N=500, **self.final_results)

            ax.plot(*vecs_fit.T, color='red', linestyle='--', label='Fit')
            ax.scatter3D(*self.vecs.T, color='blue', label='data',)
            # plot cube of points to get correct aspect ratio
            vecs_all = np.concatenate([self.vecs, vecs_fit])
            ranges = np.ptp(vecs_all, axis=0)
            mins = np.min(vecs_all, axis=0)
            maxrange = np.max(ranges)
            cens = mins + ranges/2
            p_x = [cens[0]-maxrange/2, cens[0]-maxrange/2, cens[0]-maxrange/2, cens[0]-maxrange/2,
                   cens[0]+maxrange/2, cens[0]+maxrange/2, cens[0]+maxrange/2, cens[0]+maxrange/2,]
            p_y = [cens[1]-maxrange/2, cens[1]-maxrange/2, cens[1]+maxrange/2, cens[1]+maxrange/2,
                   cens[1]-maxrange/2, cens[1]-maxrange/2, cens[1]+maxrange/2, cens[1]+maxrange/2,]
            p_z = [cens[2]-maxrange/2, cens[2]+maxrange/2, cens[2]-maxrange/2, cens[2]+maxrange/2,
                   cens[2]-maxrange/2, cens[2]+maxrange/2, cens[2]-maxrange/2, cens[2]+maxrange/2,]
            ax.scatter3D(p_x, p_y, p_z, alpha=0.0)
            # formatting
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.legend()
            p = self.final_results
#             title = f'Circle Fit Results: '+\
#             ', '.join([f'{k}={v:0.3f}' if not isinstance(v, Iterable) else f'{k}=({v[0]:0.3f}, {v[1]:0.3f}, {v[2]:0.3f})' for k, v in p.items()])
            title = f'Circle Fit Results: '
            for i, item in enumerate(p.items()):
                k, v = item
                if not isinstance(v, Iterable):
                    title += f'{k}={v:0.3f}, '
                else:
                    title += f'{k}=({v[0]:0.3f}, {v[1]:0.3f}, {v[2]:0.3f}), '
                if (i+1)%2 == 0:
                    title+='\n'
            title.rstrip('\n').rstrip(',')
            ax.set_title(title)
            return fig, ax
        elif mode == 'plotly':
            pass
        else:
            raise ValueError(f'"{mode}" is not a valid mode. Please choose a supported flip_condition: ["mpl", "plotly"]')
        
#     def plot_results_2D(self):
#         # must run self.run_fit first.

#     def plot_circle_fit_2D(self):
#         # must run self.reco_circle_coope (automatically run by self.run_fit)
#         # plot the step of actually doing a circle fit
        
        
     
        
# any additional functions
def circle_3d(t, **params):
    # t must be a scalar
    return params['r'] * np.cos(t) * params['u'] + params['r'] * np.sin(t) * np.cross(params['n'], params['u']) + params['C']

def make_circle_3d_from_params(N=500, **params):
    ths = np.linspace(0, 2*np.pi, N)
    vecs = np.array([circle_3d(t, **params) for t in ths])
    return vecs

def gen_circle(params_truth, N_points, theta_range=[0.,2*np.pi], do_noise=False, stddev_noise=0.1):
    t_gen = np.linspace(theta_range[0], theta_range[1], N_points)
    vecs_gen = np.array([circle_3d(t, **params_truth) for t in t_gen])
    if do_noise:
        vecs_gen += np.random.normal(loc=0, scale=stddev_noise, size=(N_points, 3))
    df = pd.DataFrame({'X':vecs_gen[:,0], 'Y':vecs_gen[:,1], 'Z':vecs_gen[:,2]})
    return df

def rodrigues_rotation(points, vec0, vec1):
    # points should be N x 3
    # rotate from vec0 to vec1
    # note vec0 and vec1 should be normalized
    k_vec = np.cross(vec0, vec1)
    k_vec = k_vec / np.linalg.norm(k_vec)
    costheta = np.dot(vec0, vec1)
    sintheta = (1-costheta**2)**(1/2)
    points_rot = points * costheta + np.cross(k_vec, points) * sintheta + np.outer(np.dot(points, k_vec), k_vec) * (1 - costheta)
    return points_rot

def mod_circle(x, y, **params):
    return params['A'] * x + params['B'] * y + params['C']

def run_circle_fit_survey(self, df_groups_queried, df_meas_queried, flip_condition_index, flip_condition,
                          param_settings=None):
    #df_m = df_meas_queried
    #df_g = df_groups_queried
    myCircleFitter = CircleFitter3D()
    myCircleFitter.run_fit(df_meas_queried, flip_condition_index, flip_condition, param_settings)
    return myCircleFitter