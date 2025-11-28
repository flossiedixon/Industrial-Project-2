import numpy as np
import importlib 
from IPython.display import display, clear_output
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from modules import base_model as bm
from modules import obstacles as obs
from modules import attractor as att
from modules import wind as wind

importlib.reload(obs); importlib.reload(bm); 
importlib.reload(att); importlib.reload(wind); 

# 

def step(x, y, vx, vy, theta, dt, L, A, lam_c, lam_a, lam_m, lam_att,
          eta, v0, R, obstacle_params, obs_method, attractor_pos, wind_params):
    ''' 
    ======================
    Each step of the model.
        1. Update positions from last step.
        2. Calculate new velocity updates according to 
            - cohesion, avoidance, matching
            - global attractor
            - obstacle avoidance
            - wind
        3. Limit the speeds.
        4. Return position, velocity, and angle to the plotting function.
    ======================

    *** Really it would be better for these to still be in terms of model_params, etc, as it is
    in run_simulation(). But this is fine :)

    Input:
        x, y, vx, vy, theta (ndarray): the current bird components.
        dt (float): the step size (how much to update position by).
        L (float): grid size.
        A (float): avoidance radius.
        lam_{c, a, m, att}: the cohesion/avoidance/matching/attractor strengths.
        eta (float): randomness of bird movement.
        v0 (float): initial bird speed (NB maximum speed is based on this).
        R (float): neighbour radius.
        obstacle_params (list or ndarray):
            A list of lists, each containing obstacle parameters:
            - lam_o (float): strength of obstacle avoidance.
            - x_obs, y_obs (float): obstacle centre.
            - O (float): obstacle avoidance radius.
        obs_method (string): forcefield or steer2avoid.
        attractor_pos (list or tuple): x and y coords of the attractor.
        wind_params (list or ndarray):
            - drift_wind (bool): if there is wind drift or not
            - w_xv (float): wind drift in x component
            - w_vy (float): wind drift in y component

            - dir_wind (bool): if there is directional wind or not.
            - wind_theta (float): the angle of wind.
            - lam_w (float): the wind strength.
    Output:
        x, y, u_vx, u_vy, theta (ndarray): the updated components.
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
    vx_w, vy_w = wind.add_wind(x, wind_params)

    # Update velocities and theta.
    u_vx = vx + vx_c + vx_a + vx_att + vx_m + vx_o + vx_w
    u_vy = vy + vy_c + vy_a + vy_att + vy_m + vy_o + vy_w
    theta = bm.update_theta(x, y, theta, eta, R**2)

    # NEW - maximum theta update (@Flossie).
    """ 
    I have removed for now because it was giving weird results.
    Need to talk 2 Flossie.

    old_theta = np.arctan2(vy, vx)
    updated_theta = np.arctan2(u_vy, u_vx)
    change_theta = ((((updated_theta - old_theta) + np.pi) % 2 * np.pi) - np.pi)
    
    # Assuming maximum angle change is pi/8?
    new_speed = np.sqrt(u_vy**2 + u_vx**2)
    limited_change = old_theta + np.clip(change_theta, -1/8*np.pi, 1/8*np.pi)

    u_vx = new_speed * np.cos(limited_change)
    u_vy = new_speed * np.sin(limited_change)

    # End of new. ============
    """

    # Limit speeds.
    vx_max, vy_max = bm.max_velocity(v0)
    u_vx, u_vy = bm.lim_s(u_vx, u_vy, vx_max, vy_max)

    # Return information to plotting function.
    return x, y, u_vx, u_vy, theta

def update_quiver(q, x, y, vx, vy):
    # Updates the velocity vectors on the plot.
    q.set_offsets(np.column_stack([x, y]))
    q.set_UVC(vx, vy)
    
    return q


def plot_simulation(model_params, strength_params, obstacle_params, wind_params, attractor_pos, 
                        init_left = True, obs_method = "forcefield", fig = None, ax = None, 
                        seed = 10, save = False):
    ''' 
    Runs a simulation depending on the parameters.
    If a figure is provided, it plots it there. Otherwise it creates one.

    For more detailed parameter explanation see step() above.

    Input:
        model_params (list): contains v0, eta, L, dt, Nt, N.
        strength_params (list): contains lam_c, lam_a, lam_m, A, R.
        obstacle_params (list): contains lam_o, x_obs, y_obs, O.
        wind_params (list): contains drift_wind, drift_x, drift_y, directional_wind, dir_x, dir_y.
        attractor_pos (list or tuple): x and y coordinates of the attractor.
        init_left (bool): to initialise the birds on the left half-plane, currently true.
        obs_method (str): forcefield or steer2avoid.
        fig, ax (pyplot objects): if provided, the figure/axes to plot on.
        seed (int): seed set for bird initalisation.
        save (bool): if True, saves the simulation as a gif. Otherwise displays locally in notebook.

    Output:
        Currently nothing. If we want to store the fig/axis objects, or the final
            positions and velocities, can be returned.
    '''

    # For bird initialisation.
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
    # *** If changing L you should think about changing the scale.
    q = ax.quiver(x, y, vx, vy, scale = 50)
    ax.set(xlim = (0, L), ylim = (0, L))
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    for obstacle_param in obstacle_params:
        # Strength, centre, centre, 'too close' radius.
        lam_o, x_obs, y_obs, O = obstacle_param
        obs.add_obstacle(ax, x_obs, y_obs, O, show_boundary)

    # ADDED FOR saving the animation - needs a mutable (changeable) structure.
    state = [x, y, vx, vy, theta]
    
    # You can ignore this block - it's only used if wanting to save the animation as mp4/gif.
    if (save):
        def update(frame):
            # Same process as in loop.
            # x, y, vx, vy, theta
            state[0], state[1], state[2], state[3], state[4] = step(
                state[0], state[1], state[2], state[3], state[4],
                dt, L, A, lam_c, lam_a, lam_m, lam_att, eta, v0, R, obstacle_params, obs_method, 
                attractor_pos, wind_params)
            
            update_quiver(q, state[0], state[1], state[2], state[3])

            # Need to return a tuple for animation function.
            return (q, )

        bird_ani = animation.FuncAnimation(fig, update, frames = Nt, interval = 50, blit = False)

        # Random number on file path so they don't overwrite (idk, ad-hoc).
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
            
            q = update_quiver(q, x, y, vx, vy)
            clear_output(wait = True)
            display(fig)
        
        plt.close(fig)
        return fig, ax
    
# ===================================================
# ===================================================
# ===================================================
# ===================================================
# ===================================================

# Recommend to Cathal: you will need to add more parameters to count_obstacle_deflections() so that it works. 
# Just match the parameters called to plot_simulation (you can ignore fix, ax, and save)
# That way you can see how it changes with wind/different initialisations/etc.

# Recommend to Colm: add a new function 'run_simulation()' that runs the simulation without ever plotting.
# OR just change what you have in testing.py to match the step function here. Haven't had chance to look at what you've done,
# ik you've done something similar in there.

# DM if questions! Docstring in step() should explain all the parameters. 

def count_obstacle_deflections(model_params, strength_params, obstacle_params, wind_params, attractor_pos, 
                        init_left = True, obs_method = "forcefield", seed = 10):
    """
    Run a simulation and count how many times birds are deflected by obstacles
    
    Returns:
        total_deflections (int): Total number of obstacle deflections during simulation
    """
    np.random.seed(seed)

    # Unpack the arguments
    v0, eta, L, dt, Nt, N = model_params
    lam_c, lam_a, lam_m, A, R = strength_params 

    # Reset deflection counter
    from modules.obstacles import get_deflection_count
    get_deflection_count()  # This resets the counter to 0

    # Get initial configuration
    x, y, vx, vy, theta = bm.initialize_birds(N, L, v0)

    # Run simulation
    for iT in range(Nt):
        x, y, vx, vy, theta = step(x, y, vx, vy, theta, dt, L, A, 
                            lam_c, lam_a, lam_m, eta, v0, R, obstacle_params)

    # Get final deflection count
    total_deflections = get_deflection_count()
    return total_deflections
