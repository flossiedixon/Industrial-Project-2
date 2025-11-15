import numpy as np

def initialize_birds(N, L, v0):
    '''
    Set initial positions, direction, and velocities of birds.
    Input: 
        N (int): the number of birds.
        L (float): the size of the box.
        v0 (float): initial velocity of birds.
    Output:
        x (ndarray): an Nx1 array of random starting x-positions
        y (ndarray): an Nx1 array of random starting y-positions
        vx (ndarray): an Nx1 array of random x-velocities
        vy (ndarray): an Nx1 array of random y-velocities
        theta (float): the angle matching the initial velocity
    '''

    # Bird positions - an N-dimensional ndarray.
    #x = np.random.rand(N, 1)*2
    #y = np.random.rand(N, 1)*2
    #------------------------------------------------
    #Could add input arguments, say x_range, y_range. List/array with a min and max values. Should obviously be within our grid L
    #initial position of each bird will be min + a random value between min and max
    x_range = [0, 4]
    y_range = [0, 4]
    x = x_range[0] + ( np.random.rand(N, 1) * (x_range[1] - x_range[0]) )
    y = y_range[0] + ( np.random.rand(N, 1) * (y_range[1] - y_range[0]) )
    #-------------------------------------------------------

    # Matching bird velocities, split into x and y components.
    theta = 2 * np.pi * np.random.rand(N, 1)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    return x, y, vx, vy, theta

# -----

def apply_boundary_conditions(x, y, L):
    '''
    Apply periodic boundary conditions.
    Input:
        x, y (ndarray): the Nx1 arrays of bird positions.
        L (float): the length of the box.
    Output:
        x, y (ndarray): the Nx1 arrays of (possibly) updated positions.
    '''

    # This 'wraps' the birds around - if L = 10, then (10.2, 5)
    # becomes (0.2, 5), for example. If the birds within the box
    # then nothing happens.

    x = x % L
    y = y % L
    return x, y

# -----

def update_positions(x, y, vx, vy, dt, L):
    '''
    Update the positions moving dt in the direction of the velocity
    and apply the boundary conditions.
    Input:
        x, y (ndarray): the Nx1 arrays of bird positions.
        vx, vy (ndarray); the Nx1 arrays of bird velocities.
        dt (float): the timestep size.
        L (float): the length of the box.
    Output:
        x, y (ndarray): the updated Nx1 positions.
    '''
    
    # Update positions
    x += vx*dt
    y += vy*dt
    
    # Apply boundary conditions
    x, y = apply_boundary_conditions(x, y, L)
    return x, y

# -----

def centre_of_mass(lam_c, x, y, R = 1.25):
    '''
    Determine the centre of mass velocity change, v_i^c, given
        lambda_c * [sum(x_j)/N_i  - x_i], where the sum is over N_i, the neighbours.
        This is the sam as lambda_c * [(mean of neighbours) - x_i]
        
    Input:
        lam_c (float): strength of centre-of-mass influence.
        x, y (ndarray): x and y positions of all birds.
        R (float): the 'neighbour' range.
    Output:
        vx_c, vx_y: the velocity changes in x and y according to centre of mass update.
    '''

    # Initialise empty arrays that will contain the update.
    N = x.shape[0]
    vx_c = np.zeros((N, 1))
    vy_c = np.zeros((N, 1))

    for bird in range(N): 
        # Neighbours are all birds within a circle of radius R.
        neighbours = (x - x[bird])**2 + (y - y[bird])**2 < R**2 
        x_mean = np.mean(x[neighbours])
        y_mean = np.mean(y[neighbours])

        # Velocity update = const*(difference from mean velocity)
        vx_c[bird] = lam_c * (x_mean - x[bird])
        vy_c[bird] = lam_c * (y_mean  - y[bird])

    return vx_c, vy_c

# -----


def avoid_birds(lam_a, A, x, y): 
    '''
    Determines the avoid-collisions velocity change, v_i^a, given
        lambda_a * sum(x_i - x_j)
        sum is over A, the birds that are 'too close'.
    
    Input: 
        lam_a (float): strength of avoidance influence.
        A (float): the distance defining 'too close'.
        x, y (ndarray): the current x and y positions.

    Output:
        vx_a, vx_y (ndarray): the velocity changes in x and y according to avoidance.
    '''

    # Initialise empty arrays that will contain the update.
    N = x.shape[0]
    vx_a = np.zeros((N, 1))
    vy_a = np.zeros((N, 1))

    for bird in range(N):
        # "Too close" is defined as being within the circle of radius A.
        too_close = (x - x[bird])**2 + (y - y[bird])**2 < A**2 
        x_sum = np.sum(x[bird] - x[too_close])
        y_sum = np.sum(y[bird] - y[too_close])

        # Velocity update = const*(total closeness)
        vx_a[bird]= lam_a * x_sum
        vy_a[bird] = lam_a * y_sum

    return vx_a, vy_a 

# -----

def match_birds(lam_m, x, y, vx, vy, theta, R = 1.25): 
    '''
    Determines the match-velocity change, according to
        lambda_m * (sum v_j/N - v_i) where sum is over the neighbours;
        the same as lambda_m * (mean velocity of neighbours - velocity_i).
    Input:
        lam_m (float): strength of match velocity influence
        x, y (ndarray): the x and y positions of all birds.
        vx, vy (ndarray): the x and y velocities of all birds.
        theta (ndarray): the current angles of all birds.
        R (float): the radius determining neighbours.
    Output:
        vx_m, vy_m (ndarray): the velocity updates according to cohesion (matching velocities).
    '''

    # Initialise empty arrays that will contain the update.
    N = x.shape[0]
    sx_mean = np.zeros((N, 1))
    sy_mean = np.zeros((N, 1))
        
    for bird in range (N):
        # Neighbours are birds within a circle of radius R.
        neighbours = (x - x[bird])**2 + (y - y[bird])**2 < R**2 

        # Uses Euclidean norm by default (good).
        # Calculate the average velocities of all neighbouring birds.
        sx_mean[bird] = np.mean(vx[neighbours])
        sy_mean[bird] = np.mean(vy[neighbours])

    # Velocity update const*(mean of velocities*current direction).
    vx_m = lam_m*(sx_mean*np.cos(theta))
    vy_m = lam_m*(sy_mean*np.sin(theta))
   
    return vx_m, vy_m 

# -----

def update_theta(x, y, theta, eta, R = 1.25):
    '''
    Compute the local average angle in a circle of radius R around
    each bird. Use this average to update the angle, providing some noise.

    Input:
        x, y (ndarray): the x and y positions of birds.
        theta (ndarray): the angles of birds.
        eta (float): the parameter controlling noise variance.
        R (float): the radius determining neighbours.
    Output:
        theta_update (ndarray): the new updated angles, with some noise.
    '''

    N = x.shape[0]
    mean_theta = np.zeros((N, 1))

    for bird in range(N):
        # Calculate the average angle within the neighbourhood.
        neighbors = (x - x[bird])**2 + (y - y[bird])**2 < R**2
        sum_x = np.sum(np.cos(theta[neighbors]))
        sum_y = np.sum(np.sin(theta[neighbors]))
        mean_theta[bird] = np.arctan2(sum_y, sum_x)

    # Noise is a sample from a Gaussian (0, eta/2_)
    theta_update = mean_theta + (eta/2)*(np.random.randn(N, 1))
    return theta_update

# -----

def max_velocity(v0):
    '''
    Determine the maximum velocities in x and y.
    Currently trivial - could be a function of lambda.
    '''

    vx_max, vy_max = 2*v0, 2*v0
    return vx_max, vy_max

# -----

def lim_s(vx, vy, vx_max, vy_max):
    '''
    Limit the speed of each individual bird by the maximum velocity.
    Note this is done in each component (x and y) instead of the total 
    magnitude.
    Input:
        vx, vy (ndarray): the proposed bird velocities.
        vx_max, vy_max (float): the maximum velocity in both x and y.
    Output:
        ux, uy (ndarray): the restricted bird velocities.
    '''

    # CHANGED to magnitude/total speed.
    speed = np.sqrt(vx**2 + vy**2)
    factor = np.maximum(1, speed / vx_max)

    # Update the speeds according to this factor.
    ux = np.divide(vx, factor)
    uy = np.divide(vy, factor)

    return ux, uy

# if vx>v_max then set vx to be vxmax by vx / (vx/vxmax)

# -----

def update_v(vx, vy, vx_c, vy_c, vx_a, vy_a, vx_m, vy_m):
    '''
    Update the velocities according to
        velocity += centre of mass (com), avoidance, matching
    Input:
        vx, vy (ndarray): the x and y velocities.
        vx_c, vy_c (ndarray): the x y and com velocities.
        vx_a, vy_a (ndarray): the x and y avoidance velocities.
        vx_m, vy_m (ndarray): the x and y matching velocities.

    Output: 
        u_vx, u_vy (ndarray): the updated velocities.
    '''

    u_vx = vx + vx_c + vx_a + vx_m
    u_vy = vy + vy_c + vy_a + vy_m

    return u_vx, u_vy

# -----

def step(x, y, vx, vy, theta, dt, L, A, lam_c, lam_a, lam_m, eta, v0, R = 1.25):
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
        A (float): the radius of avoidance.
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

    x, y = update_positions(x, y, vx, vy, dt, L)

    # Calculate cohesion, avoidance, matching velocities.
    vx_c, vy_c = centre_of_mass(lam_c, x, y, R)
    vx_a, vy_a = avoid_birds(lam_a, A, x, y)
    vx_m, vy_m = match_birds(lam_m, x, y, vx, vy, theta, R)

    # Update velocities and angles.
    u_vx, u_vy = update_v(vx, vy, vx_c, vy_c, vx_a, vy_a, vx_m, vy_m)
    theta = update_theta(x, y, theta, eta, R**2)

    # Limit speeds.
    vx_max, vy_max = max_velocity(v0)
    u_vx, u_vy = lim_s(u_vx, u_vy, vx_max, vy_max)

    return x, y, u_vx, u_vy, theta


    




    