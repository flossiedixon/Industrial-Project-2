import numpy as np

def add_wind(vx, vy, wind_params):
    ''' 
    Adds wind to the system.
    Two types: drift and directional.
        Drift: a constant force irrespective of current velocities.
        Directional: wind strength is proportional to velocities.

    Input:
        x (ndarray): the bird x-positions. (NB only used to get N).
        wind_params (ndarray or list): the wind parameters:
            - drift_wind (bool): if there is wind drift or not
            - w_xv (float): wind drift in x component
            - w_vy (float): wind drift in y component

            - dir_wind (bool): if there is directional wind or not.
            - wind_theta (float): the angle of wind.
            - lam_w (float): the wind strength.
    Output:
        vx_w (ndarray): the wind velocity contribution in x.
        vy_w (ndarray): the wind velocity contribution in y.
    '''

    # Drift wind (bool) and wind drift velocities,
    # then directional wind (bool) and wind theta/strength.
    drift_wind, w_vx, w_vy, dir_wind, wind_theta, lam_w = wind_params  

    # Initialise empty arrays that will contain the update.
    N = vx.shape[0]
    vx_w = np.zeros((N, 1))
    vy_w = np.zeros((N, 1))

    if (drift_wind):
        vx_w += w_vx
        vy_w += w_vy

    if (dir_wind):
        vx_w += lam_w * vx * np.cos(wind_theta)
        vy_w += lam_w * vy * np.cos(wind_theta)

    return vx_w, vy_w
