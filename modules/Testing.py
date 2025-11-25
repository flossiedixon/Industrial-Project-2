#Testing here
#Should have two functions after a bit of research/thinking
#One will be similar to run_simulation, however, I won't produce a simualtion. This seems to be time consuming and
#I think a second function focused on repaeting the first functions and giving stats such as mean will be neccessary

#This code is the plot simulation just updated to remove plot and check weather we reach the end goal
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import matplotlib.animation as animation

import sys
import os

# Get the path of the current file's directory (modules/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (Industrial-Project-2/) to the system path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import importlib
from modules import base_model as bm 
from modules import obstacles as obs
from modules import attractor as att
from modules import run_simulation as rs
importlib.reload(bm)
importlib.reload(obs)
importlib.reload(att)
importlib.reload(rs)

def check_boundary(x, goal):
    return x >= goal



def simulation_test(model_params, strength_params, obstacle_params, attractor_pos, goal: float, seed = 10 ):

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

    # if (fig is None) or (ax is None):
    #     fig, ax = plt.subplots(figsize = (10, 10))

    # Get the initial configuration
    x, y, vx, vy, theta = bm.initialize_birds(N, L, v0)


    
    total_boids = N #count total birds
    boids_finished = 0 #start a birds counter
    time_steps_taken = Nt


  

    for iT in range(Nt):
        #Get all birds who have crossed the boundary
        #This should be updated to create a grid or take into account y position
        finished_mask = check_boundary(x, goal)
        
        #Add these birds to a counter so we can check when finished
        #sum the numbers of birds which have crossed. if > 0 we repeat the process
        #at each iteration the remaing should be decreasing down to 0
        #We only inbcrease the counter if new birds cross. Could be the case taht birds remain but dont cross persistently
        crossed = np.sum(finished_mask)
        if crossed > 0:
            boids_finished += crossed
            
            #The remaining will be not finished_mask
            #~ is the inverse, so not crossed the x boundary
            remaining = ~finished_mask

            #Mask of only remaing birds applied to all parameters/arguments
            x = x[remaining]
            y = y[remaining]
            vx = vx[remaining]
            vy = vy[remaining]
            theta = theta[remaining]

            #This is gemini helping with debugging. Seems to be an issue related to one of the velocity functions, e.g match birds
            #One of the velocities seems to not be an (N, 1) array, but a 3D array. Reshape fixes this. Should figure out this bug for future
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            vx = vx.reshape(-1, 1)
            vy = vy.reshape(-1, 1)
            theta = theta.reshape(-1, 1)

            #Simple break if make birds is reached. Also count the current iterations as time step and return
            if boids_finished == total_boids:
                time_steps_taken = iT
                break
        # if there is remaining birds apply step function an dgenerate new step for birds. Then repeat masking/crossing process
        if len(x) > 0:
            # Use step, not the base model.
            x, y, vx, vy, theta = att.step(x, y, vx, vy, theta, dt, L, A, 
                lam_c, lam_a, lam_m, lam_att, eta, v0, R, obstacle_params, attractor_pos)
        #This may not be neccessary and code will break given x = 0, but just incase
        else:
            break
        
    #Reset random state to before simulation so subsequent simulations are unaffected
    np.random.set_state(rng_state)
    return time_steps_taken, boids_finished, total_boids



