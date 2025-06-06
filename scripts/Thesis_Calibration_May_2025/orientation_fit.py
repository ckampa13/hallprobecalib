import numpy as np
#import pandas as pd
import lmfit as lm
from collections.abc import Iterable
from scipy.spatial.transform import Rotation

def model_func_CalibrationOrientationFit(axis0, axis1, Bmag, **params):
    if not isinstance(axis0, Iterable):
        axis0 = np.array([axis0])
        axis1 = np.array([axis1])
        Bmag = np.array([Bmag])
    # scale axis0, axis1
    axis0 = params['a0_sf'] * axis0
    axis1 = params['a1_sf'] * axis1
    # calculate normal vectors
    # Hall element is w.r.t. "SmarAct coordinates" (or better, Kinematic plate coordinates) and is fixed, regardless of axis0, axis1
    norm_Hall = np.array([np.sin(params['theta'])*np.cos(params['phi']), np.sin(params['theta'])*np.sin(params['phi']), np.cos(params['theta'])])
    # x_Hall is arbitary...
    x_Hall = np.array([-np.sin(params['phi']), np.cos(params['phi']), 0.])
    # B field has some nominal orientation w.r.t. kinematic plate, and is further rotated by axis0, axis1
    norm_B = np.array([np.sin(params['alpha'])*np.cos(params['beta']), np.sin(params['alpha'])*np.sin(params['beta']), np.cos(params['alpha'])])
    # rotate the field vector w.r.t.
    rot = Rotation.from_euler('yz', angles=np.array([-axis0, -axis1]).T, degrees=True) # ok
    norm_B_rot = rot.apply(np.repeat(norm_B[:, np.newaxis], len(axis0), axis=1).T)
    # calculate signed fraction of B vector along Hall normal i.e. Hall only reads perpendicular field (to first order)
    costh = np.dot(norm_B_rot, norm_Hall)
    sinth = (1 - costh**2)**(1/2)
    theta = np.arccos(costh)
    BT_rot90 = np.cross(norm_B_rot, norm_Hall)
    BT_rot90 = BT_rot90 / np.linalg.norm(BT_rot90, axis=1)[:, None]
    BT_rot = np.cross(norm_Hall, BT_rot90)
    # must normalize BT_rot
    BT_rot = BT_rot / np.linalg.norm(BT_rot, axis=1)[:, None]
    phi_Hall = np.arccos(np.dot(BT_rot, x_Hall))
    sign_phi = 2*(np.dot(np.cross(x_Hall, BT_rot), norm_Hall) > 0.).astype(int) - 1.
    phi_Hall = sign_phi * phi_Hall + params['phase']
    # recenter phi
    phi_Hall[phi_Hall > np.pi] = phi_Hall[phi_Hall > np.pi] - 2*np.pi
    phi_Hall[phi_Hall < -np.pi] = phi_Hall[phi_Hall < -np.pi] + 2*np.pi
    ###
    Bx = Bmag * sinth * np.cos(phi_Hall)
    By = Bmag * sinth * np.sin(phi_Hall)
    Bz = Bmag * costh
    B_meas = np.concatenate([Bx, By, Bz])
    if params['verbose'] == 1.:
        return B_meas, norm_Hall, norm_B, norm_B_rot, BT_rot, theta, phi_Hall
    else:
        return B_meas