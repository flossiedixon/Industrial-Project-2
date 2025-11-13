import numpy as np
import importlib 

from modules import base_model as bm
importlib.reload(bm)

def avoid_obstacle(lam_o, x, y, x_obs, y_obs, O):
    ''' 
    Determines the avoid-obstacle velocity update,
    simialr to that of avoid_birds(); that is,
        lambda_o * (proportional distance to obstacle).

    Input:
        lam_o (float): the strength of the obstacle-avoidance update.
        x, y (ndarray): positions of all birds.
        x_obs, y_obs (float): the position of the obstacle.
        O (float): the radius to avoid the obstacle from.
    '''

    N = x.shape[0]
    vx_o = np.zeros((N, 1))
    vy_o = np.zeros((N, 1))

    for bird in range(N):
        # Being too near means within the radius O of the obstacle.
        euclid_dist = np.sqrt((x[bird] - x_obs)**2 + (y[bird] - y_obs)**2)
        too_close = euclid_dist < O

        if (too_close):
            # The strength is inversely proportional to the distance?
            # Don't divide by zero - add something small to denominator.
            repulsion_strength = lam_o / (euclid_dist**2 + 1e-5)
            vx_o[bird] = repulsion_strength * ((x[bird] - x_obs) / euclid_dist)
            vy_o[bird] = repulsion_strength * ((y[bird] - y_obs) / euclid_dist)
    
    # If they are not too close, this will be zero.
    return vx_o, vy_o

# -----

def step(x, y, vx, vy, theta, dt, L, A, O, lam_c, lam_a, lam_m, lam_o,
          eta, v0, R, x_obs, y_obs):
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
    vx_o, vy_o = avoid_obstacle(lam_o, x, y, x_obs, y_obs, O)

    # Update velocities and angles.
    # REMOVED the update_v function - felt unnecessary.
    u_vx = vx + vx_c + vx_a + vx_m + vx_o
    u_vy = vy + vy_c + vy_a + vy_m + vy_o
    theta = bm.update_theta(x, y, theta, eta, R**2)

    # Limit speeds.
    vx_max, vy_max = bm.max_velocity(v0)
    u_vx, u_vy = bm.lim_s(u_vx, u_vy, vx_max, vy_max)

    return x, y, u_vx, u_vy, theta

# -----

