import numpy as np
import importlib 

from modules import base_model as bm
importlib.reload(bm)

deflection_count = 0

def avoid_obstacle(x, y, vx, vy, L, obstacle_params, obs_method = "forcefield"):
    ''' 
    Determines the avoid-obstacle velocity update,
    simialr to that of avoid_birds(); that is,
        lambda_o * (proportional distance to obstacle).
    Iterated on each obstacle given.

    Input:
        x, y (ndarray): positions of all birds.
    '''
    global deflection_count
    if obs_method not in ["forcefield", "steer2avoid"]:
        raise ValueError(f'Invalid obstacle avoidance method received: {obs_method}.')

    N = x.shape[0]
    vx_o = np.zeros((N, 1))
    vy_o = np.zeros((N, 1))

    if (obs_method == "forcefield"):
        for bird in range(N):
            for obstacle_param in obstacle_params:
                lam_o, x_obs, y_obs, O = obstacle_param
                # Being too near means within the radius O of the obstacle.
                euclid_dist = np.sqrt((x[bird] - x_obs)**2 + (y[bird] - y_obs)**2)
                too_close = euclid_dist < O

                if (too_close):
                    #deflection count
                    deflection_count += 1
                    
                    # The strength is inversely proportional to the distance?
                    # Don't divide by zero - add something small to denominator.
                    repulsion_strength = lam_o / (euclid_dist**2 + 1e-2)
                    vx_o[bird] = repulsion_strength * ((x[bird] - x_obs) / euclid_dist)
                    vy_o[bird] = repulsion_strength * ((y[bird] - y_obs) / euclid_dist)

                    # Break - this assumes each bird is only near ONE OBSTACLE at a time.
                    break
            
    # Steer to avoid in else block.
    elif (obs_method == "steer2avoid"):
        for bird in range(N):
            # Check each predicted trajectory against all obstacles. 
            # If a trajectory hits an obstacle, move on to the next bird.
            cur_x, cur_y = x[bird], y[bird]
            cur_vx, cur_vy = vx[bird], vy[bird]

            if (np.isnan(cur_vx) or np.isnan(cur_vy)):
                continue

            # Restrict the range of alphas so that the predicted trajectories do not
            # go more than L/2 away of the bird. In this sense L/2 is the "eyesight range".
            alpha_max = (L/2) / (np.sqrt(cur_vx**2 + cur_vy**2) + 1e-2)
            
            # Arbitrary maximum for now.
            alpha_max = min(alpha_max, L)

            # print(f'The alpha max is {alpha_max}.')
            alpha_step = (L/2) / 100
            alphas = np.arange(1, np.abs(alpha_max), alpha_step)

            # Trajectories = alpha(vx + vy) for each alpha.
            trajec_x = cur_x + alphas * cur_vx
            trajec_y = cur_y + alphas * cur_vy

            # trajec_x = [cur_x + alpha*cur_vx for alpha in alphas]
            # trajec_y = [cur_y + alpha*cur_vy for alpha in alphas]

            for i, (pred_x, pred_y) in enumerate(zip(trajec_x, trajec_y)):
                will_collide = False

                for obstacle_param in obstacle_params:
                    lam_o, x_obs, y_obs, O = obstacle_param

                    # Being too near means PREDICTED TRAJEC iswithin the radius O of the obstacle.
                    euclid_dist = np.sqrt((pred_x - x_obs)**2 + (pred_y - y_obs)**2)
                    too_close = euclid_dist < O

                    if (too_close):
                        will_collide = True 
                        alpha = alphas[i]

                        # Originally the same as the force-field, but then divide by alpha
                        # so the further away you are, the less you change.
                        repulsion_strength = lam_o / (euclid_dist**2 + 1e-5)
                        repulsion_strength *= (1 / alpha)

                        vx_o[bird] = repulsion_strength * ((pred_x - x_obs) / euclid_dist + 1e-2)
                        vy_o[bird] = repulsion_strength * ((pred_y - y_obs) / euclid_dist + 1e-2)

                        # Don't check any other obstacles
                        break

                if (will_collide):
                    # Don't check any other trajectories - move on to the next bird.
                    break 

    # If they are not too close, this will be zero.
    return vx_o, vy_o

# -----
def get_deflection_count():
    """Get the current deflection count and reset it"""
    global deflection_count
    count = deflection_count
    deflection_count = 0  # Reset for next simulation
    return count
# -----
def step(x, y, vx, vy, theta, dt, L, A, lam_c, lam_a, lam_m,
            eta, v0, R, obstacle_params):
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
        lam_c, lam_a, lam_m (float): the strength of each
            velocity update.
        eta (float): the strength of noise update.
        v0 (float): the initial speed of all birds.
        R (float): the radius of neighbours.
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

    vx_o, vy_o = avoid_obstacle(x, y, obstacle_params)

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

