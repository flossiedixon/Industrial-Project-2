import numpy as np
import importlib 

from modules import base_model as bm
importlib.reload(bm)
from modules import obstacles as obs
importlib.reload(obs)

def update_v(vx, vy, vx_c, vy_c, vx_a, vy_a, vx_m, vy_m):
    '''
    Update the velocities according to
        velocity += centre of mass (com), avoidance, matching;
        Then introduce the impact of wind on the velocity.
    Input:
        vx, vy (ndarray): the x and y velocities.
        vx_c, vy_c (ndarray): the x y and com velocities.
        vx_a, vy_a (ndarray): the x and y avoidance velocities.
        vx_m, vy_m (ndarray): the x and y matching velocities.

    Output: 
        u_vx, u_vy (ndarray): the updated velocities.
    '''

    u_vx = vx + vx_c + vx_a + vx_m
    u_vy = vy + vy_c + vy_a + vy_m

    # 1. Wind drift
    if drift_wind:
        u_vx += wind_vx
        u_vy += wind_vy

    # 2. Directional wind bias
    if direction_wind:
        u_vx += wind_strength * np.cos(wind_theta)
        u_vy += wind_strength * np.sin(wind_theta)
    return u_vx, u_vy
