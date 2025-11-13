import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import importlib
from modules import base_model as bm 
importlib.reload(bm)

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

    # return fig, ax, (x, y, vx, vy, theta)



