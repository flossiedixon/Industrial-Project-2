import numpy as np
import importlib 

import base_model as bm
importlib.reload(bm)
import obstacles as obs
importlib.reload(bm)

def global_attractor(lam_att, x, y, attractor_pos):

    N = x.shape[0]
    vx_att = np.zeros((N, 1))
    vy_att = np.zeros((N, 1))
    for bird in range(N):
        vx_att[bird] = lam_att* (attractor_pos[0] - x[bird])
        vy_att[bird] = lam_att * (attractor_pos[1] - y[bird])

    return vx_att, vy_att
# -----

def step(x, y, vx, vy, theta, dt, L, A, O, lam_c, lam_a, lam_m, lam_o, lam_att,
          eta, v0, R, x_obs, y_obs, attractor_pos):
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
        lam_c, lam_a, lam_m (float): the strength of each
            velocity update.
        eta (float): the strength of noise update.
        v0 (float): the initial speed of all birds.
        R (float): the radius of neighbours.
        x_obs (float): the x-position of the obstacle.
        y_obs (float): the y-position of the obstacle.
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
    vx_o, vy_o = obs.avoid_obstacle(lam_o, x, y, x_obs, y_obs, O)
    vx_att, vy_att = global_attractor(lam_att, x, y, attractor_pos)
    # Update velocities and angles.
    # REMOVED the update_v function - felt unnecessary.
    u_vx = vx + vx_c + vx_a + vx_m + vx_o + vx_att
    u_vy = vy + vy_c + vy_a + vy_m + vy_o + vy_att
    theta = bm.update_theta(x, y, theta, eta, R**2)

    # Limit speeds.
    vx_max, vy_max = bm.max_velocity(v0)
    u_vx, u_vy = bm.lim_s(u_vx, u_vy, vx_max, vy_max)

    return x, y, u_vx, u_vy, theta

# -----

