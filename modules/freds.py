import numpy as np
import importlib 
from IPython.display import display, clear_output
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from modules import base_model as bm
from modules import obstacles as obs
from modules import attractor as att
from modules import run_simulation as run_sim

importlib.reload(obs); importlib.reload(bm); 
importlib.reload(att);

# 

def wind(x, wind_params):
    ''' 
    
    '''

    # Drift wind (bool) and wind drift velocities,
    # then directional wind (bool) and wind theta/strength.
    drift_wind, w_vx, w_vy, dir_wind, wind_theta, lam_w = wind_params  

    # Initialise empty arrays that will contain the update.
    N = x.shape[0]
    vx_w = np.zeros((N, 1))
    vy_w = np.zeros((N, 1))

    if (drift_wind):
        vx_w += w_vx
        vy_w += w_vy

    if (dir_wind):
        vx_w += lam_w * np.cos(wind_theta)
        vy_w += lam_w * np.cos(wind_theta)

    return vx_w, vy_w

    
# Put this into a separate .py file ALL BY ITS LONESOME when we're done.
def step(x, y, vx, vy, theta, dt, L, A, lam_c, lam_a, lam_m, lam_att,
          eta, v0, R, obstacle_params, obs_method, attractor_pos, wind_params):
    ''' 
    =
    To complete docstring!
    '''

    x, y = bm.update_positions(x, y, vx, vy, dt, L)

    # Calculate cohesion, avoidance, matching velocities.
    vx_c, vy_c = bm.centre_of_mass(lam_c, x, y, R)
    vx_a, vy_a = bm.avoid_birds(lam_a, A, x, y)
    vx_m, vy_m = bm.match_birds(lam_m, x, y, vx, vy, theta, R)

    # Global attractor.
    vx_att, vy_att = att.global_attractor(lam_att, x, y, attractor_pos)

    # Obstacle avoidance - obs_method is either 'forcefield' or 'steer2avoid'.
    vx_o, vy_o = obs.avoid_obstacle(x, y, vx, vy, L, obstacle_params, obs_method)

    # NEW - Add wind.
    vx_w, vy_w = wind(x, wind_params)

    # Update velocities and theta.
    u_vx = vx + vx_c + vx_a + vx_att + vx_m + vx_o + vx_w
    u_vy = vy + vy_c + vy_a + vy_att + vy_m + vy_o + vy_w
    theta = bm.update_theta(x, y, theta, eta, R**2)

    # Limit speeds.
    vx_max, vy_max = bm.max_velocity(v0)
    u_vx, u_vy = bm.lim_s(u_vx, u_vy, vx_max, vy_max)

    # Return information to plotting function.
    return x, y, u_vx, u_vy, theta

# 
#
#
#
#

def plot_simulation(model_params, strength_params, obstacle_params, wind_params, attractor_pos, 
                        init_left = True, obs_method = "forcefield", fig = None, ax = None, seed = 10, save = False):
    ''' 
    Runs a simulation depending on the parameters.
    If a figure is provided, it plots it there. Otherwise it creates one.

    Input:
        model_params (list or tuple): contains v0, eta, L, dt, Nt, N.
        strength_params (list or tuple): contains lam_c, lam_a, lam_m, A, R.

    Output:
        Currently nothing. If we want to store the fig/axis objects, or the final
            positions and velocities, can be returned.
    '''

    rng_state = np.random.get_state()
    np.random.seed(seed)

    # Unpack the arguments.
    v0, eta, L, dt, Nt, N = model_params
    lam_c, lam_a, lam_m, lam_att, A, R = strength_params 

    # Lighter obstacle outer circle - don't show if S2A.
    show_boundary = True if obs_method == "forcefield" else False

    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(figsize = (10, 10))

    # Get the initial configuration
    x, y, vx, vy, theta = bm.initialize_birds(N, L, v0, init_left)

    # Do an initial plot and set up the axes.
    q = ax.quiver(x, y, vx, vy, scale = 50)
    ax.set(xlim = (0, L), ylim = (0, L))
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # See new helper function defined above.
    for obstacle_param in obstacle_params:
        # Strength, centre, centre, 'too close' radius.
        lam_o, x_obs, y_obs, O = obstacle_param
        run_sim.add_obstacle(ax, x_obs, y_obs, O, show_boundary)

    # ADDED FOR saving the animation - needs a mutable (changeable) structure.
    state = [x, y, vx, vy, theta, q]
    
    # You can ignore this block - it's only used if wanting to save the animation as mp4.
    if (save):
        def update(frame):
            # Same process as in loop, 'global' tells Python to use the variables outside
            # the loop. Could make it a list/dict instead
            state[0], state[1], state[2], state[3], state[4] = step(
                state[0], state[1], state[2], state[3], state[4],
                dt, L, A, lam_c, lam_a, lam_m, lam_att, eta, v0, R, obstacle_params, obs_method, 
                attractor_pos, wind_params)
            
            run_sim.update_quiver(q, state[0], state[1], state[2], state[3])

            # Need to return a tuple for animation function.
            return (q, )

        bird_ani = animation.FuncAnimation(fig, update, frames = Nt, interval = 50, blit = False)

        # Random number on path so they don't overwrite (idk, ad-hoc).
        np.random.set_state(rng_state)
        randint = np.random.randint(1, 1e5)

        path = f'movies/flocking{randint}.gif'
        bird_ani.save(path, writer = "ffmpeg", fps = 20)
    else:
        for iT in range(Nt):
            # Use obstacles step, not the base model.
            x, y, vx, vy, theta = step(x, y, vx, vy, theta, dt, L, A, 
                                    lam_c, lam_a, lam_m, lam_att, eta, v0, R, 
                                    obstacle_params, obs_method, attractor_pos, wind_params)
            
            q = run_sim.update_quiver(q, x, y, vx, vy)
            clear_output(wait = True)
            display(fig)
        
        plt.close(fig)
        return fig, ax