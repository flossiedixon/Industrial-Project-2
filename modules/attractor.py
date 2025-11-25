import sys
import os

# Get the path of the current file's directory (modules/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (Industrial-Project-2/) to the system path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import importlib 

from modules import base_model as bm
importlib.reload(bm)
from modules import obstacles as obs

importlib.reload(obs)

def global_attractor(lam_att, x, y, attractor_pos):
    '''
    1. Initialise number of birds and attractor velocity of the birds.
    2. Velocity is updated as strength factor * (Point of attraction - current position)
    3. Return the x and y velocities

    Input:
        lam_att: This is the factor we apply to the velocity update. This should be kept quite low. <0.2
        x, y: x, y position
        attractor_pos [x, y] = point of attraction (point the birds are migrating towards). This does not have to be within the L*L grid
    '''

    N = x.shape[0]
    vx_att = np.zeros((N, 1))
    vy_att = np.zeros((N, 1))
    for bird in range(N):
        vx_att[bird] = lam_att * (attractor_pos[0] - x[bird]) * 2
        vy_att[bird] = lam_att * (attractor_pos[1] - y[bird]) * 2

    return vx_att, vy_att
# -----

def step(x, y, vx, vy, theta, dt, L, A, lam_c, lam_a, lam_m, lam_att,
          eta, v0, R, obstacle_params, attractor_pos):
    ''' 
    1. Update positions.
    2. Calculate cohesion/avoidance/matching velocities.
    3. Update the velocity and angle.
    4. Limit speeds.

    Input:
        x, y (ndarray): the positions of birds.
        vx, vy (ndarray): the velocities of birds.
        theta (ndarray): the angle of the birds.
        dt (float): the timestep.
        L (float): the size of the box.
        A (float): the radius of bird avoidance.
        O (float): the radius of obstacle avoidance.
        lam_c, lam_a, lam_m, lam_att (float): the strength of each
            velocity update.
        eta (float): the strength of noise update.
        v0 (float): the initial speed of all birds.
        R (float): the radius of neighbours.
        x_obs (float): the x-position of the obstacle.
        y_obs (float): the y-position of the obstacle.
        attractor_Pos: migration point [x, y]
    Output:
        x, y (ndarray): the positions of the birds.
        vx, vy (ndarray): the velocities of the birds.
        theta (ndarray): the angles of the birds.
    '''

    x, y = bm.update_positions(x, y, vx, vy, dt, L)

    # Calculate cohesion, avoidance, matching velocities.
    vx_c, vy_c = bm.centre_of_mass(lam_c, x, y, R)
    vx_a, vy_a = bm.avoid_birds(lam_a, A, x, y)
    vx_m, vy_m = bm.match_birds(lam_m, x, y, vx, vy, theta, R)
    vx_att, vy_att = global_attractor(lam_att, x, y, attractor_pos)

    vx_o, vy_o = obs.avoid_obstacle(x, y, obstacle_params)
    

    u_vx = vx +  vx_c + vx_a + vx_att + vx_m + vx_o
    u_vy = vy  + vy_c + vy_a + vy_att + vy_m + vy_o
    theta = bm.update_theta(x, y, theta, eta, R**2)

    # Limit speeds.
    vx_max, vy_max = bm.max_velocity(v0)
    u_vx, u_vy = bm.lim_s(u_vx, u_vy, vx_max, vy_max)

    return x, y, u_vx, u_vy, theta