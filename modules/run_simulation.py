import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import matplotlib.animation as animation

import importlib
from modules import base_model as bm 
from modules import obstacles as obs
from modules import attractor as att
importlib.reload(bm)
importlib.reload(obs)
importlib.reload(att)

def update_quiver(q, x, y, vx, vy):
    # Updates the arrows on the plot.
    q.set_offsets(np.column_stack([x,y]))
    q.set_UVC(vx,vy)
    
    return q

def plot_simulation(model_params, strength_params, fig = None, ax = None, seed = 10):
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

    np.random.seed(seed)

    # Unpack the arguments.
    v0, eta, L, dt, Nt, N = model_params
    lam_c, lam_a, lam_m, A, R = strength_params 

    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(figsize = (10, 10))

    # Get the initial configuration
    x, y, vx, vy, theta = bm.initialize_birds(N, L, v0)

    # Save the initial values to be used later
    x_init, y_init, vx_init, vy_init, theta_init = (
        x.copy(), y.copy(), vx.copy(), vy.copy(), theta.copy()
    )

    # Do an initial plot and set up the axes.
    q = ax.quiver(x, y, vx, vy, scale = 50)
    ax.set(xlim = (0, L), ylim = (0, L))
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    for iT in range(Nt):
        x, y, vx, vy, theta = bm.step(x, y, vx, vy, theta, dt, L, A, lam_c, lam_a, lam_m, eta, v0, R)
        q = update_quiver(q, x, y, vx, vy)
        clear_output(wait = True)
        display(fig)
    
    plt.close(fig)
    return fig, ax

# -----

def add_obstacle(ax, x_obs, y_obs, O, show_boundary = True):
    ''' 
    Helper function to add a circle to the plots.
    Input:
        ax (plt axes): the figure axis.
        centre (float): the centre of the obstacle.
    '''

    # Plot the obstacle and it's 'force-field' effect.
    # The inner circle represents the obstacle, the outer circle the 'too close' zone.
    inner_circle = plt.Circle((x_obs, y_obs), 0.25*O, color = 'orange', alpha = 0.8)
    outer_circle = plt.Circle((x_obs, y_obs), O, color = 'orange', alpha = 0.1)
    ax.add_patch(inner_circle)
    
    if (show_boundary):
        ax.add_patch(outer_circle)

    return ax


def plot_simulation_obs(model_params, strength_params, obstacle_params, fig = None, ax = None, seed = 10, save = False):
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
    lam_c, lam_a, lam_m, A, R = strength_params 

    # Want it to be a list (or NumPy array) of lists.
    if not(all(isinstance(i, list) for i in obstacle_params)):
        # If it's just a list, make it a list of lists.
        print(f'Changing the input {obstacle_params} to be a list of lists.')
        obstacle_params = np.array([obstacle_params])

    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(figsize = (10, 10))

    # Get the initial configuration
    x, y, vx, vy, theta = bm.initialize_birds(N, L, v0)

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
        add_obstacle(ax, x_obs, y_obs, O)

    # ADDED FOR saving the animation - needs a mutable (changeable) structure.
    state = [x, y, vx, vy, theta, q]
    
    # You can ignore this block - it's only used if wanting to save the animation as mp4.
    if (save):
        def update(frame):
            # Same process as in loop, 'global' tells Python to use the variables outside
            # the loop. Could make it a list/dict instead
            state[0], state[1], state[2], state[3], state[4] = obs.step(
                state[0], state[1], state[2], state[3], state[4],
                dt, L, A, lam_c, lam_a, lam_m, eta, v0, R, obstacle_params)
            
            update_quiver(q, state[0], state[1], state[2], state[3])

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
            x, y, vx, vy, theta = obs.step(x, y, vx, vy, theta, dt, L, A, 
                                lam_c, lam_a, lam_m, eta, v0, R, obstacle_params)
            
            q = update_quiver(q, x, y, vx, vy)
            clear_output(wait = True)
            display(fig)
        
        plt.close(fig)
        return fig, ax
    
def plot_simulation_att(model_params, strength_params, obstacle_params, attractor_pos, 
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
        add_obstacle(ax, x_obs, y_obs, O, show_boundary)

    # ADDED FOR saving the animation - needs a mutable (changeable) structure.
    state = [x, y, vx, vy, theta, q]
    
    # You can ignore this block - it's only used if wanting to save the animation as mp4.
    if (save):
        def update(frame):
            # Same process as in loop, 'global' tells Python to use the variables outside
            # the loop. Could make it a list/dict instead
            state[0], state[1], state[2], state[3], state[4] = att.step(
                state[0], state[1], state[2], state[3], state[4],
                dt, L, A, lam_c, lam_a, lam_m, lam_att, eta, v0, R, obstacle_params, obs_method, attractor_pos)
            
            update_quiver(q, state[0], state[1], state[2], state[3])

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
            x, y, vx, vy, theta = att.step(x, y, vx, vy, theta, dt, L, A, 
                                    lam_c, lam_a, lam_m, lam_att, eta, v0, R, 
                                    obstacle_params, obs_method, attractor_pos)
            
            q = update_quiver(q, x, y, vx, vy)
            clear_output(wait = True)
            display(fig)
        
        plt.close(fig)
        return fig, ax


def count_obstacle_deflections(model_params, strength_params, obstacle_params, seed=10):
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
        x, y, vx, vy, theta = obs.step(x, y, vx, vy, theta, dt, L, A, 
                            lam_c, lam_a, lam_m, eta, v0, R, obstacle_params)

    # Get final deflection count
    total_deflections = get_deflection_count()
    return total_deflections


