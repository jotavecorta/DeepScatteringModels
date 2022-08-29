import numpy as np
from t_matrix import SpmSurface
from itertools import product

def init_parameters_grid(grid_length=35):
    # Constants parameters: incident wave length and number [cm]
    lambda_ = .245
    k = 2 * np.pi / lambda_

    # Incident Azimut and polar angle [radians]
    theta, phi = 38.5*np.pi/180, 0

    # Parameters to sample [cm]: distance btw layers, rms high and corr length
    distance = np.linspace(.1, .7, 35)
    rms_high1 = np.linspace(.004, .012, 35)
    rms_high2 = np.linspace(.004, .012, 35)
    corr_length1, corr_length2 = 6*rms_high1, 6*rms_high2

    # Layers 1 and 2 Dielectric Constant
    init_epsilon = 3
    epsilon1 = np.arange(init_epsilon, init_epsilon + grid_length) 
    epsilon2 = np.arange(init_epsilon, init_epsilon + grid_length)

    # Returns a dictionary with all parameters to form grid
    space_grid = {
        'rms_high' : rms_high1,
        'corr_length' : corr_length1, 
        'epsilon' : epsilon1,
        'epsilon_2' : epsilon2, 
        'rms_high_2' : rms_high2, 
        'corr_length_2' : corr_length2,
        'distance' : distance,
        'theta_inc' : theta,
        'phi_inc' : phi
    }

    return space_grid

def make_data(realizations=20480):
    params_to_sample = init_parameters_grid()

    # Form a list of hyperparameters values
    parameters_list = list(params_to_sample.values())

    # Get all posible combinations of parameters
    all_combinations = []
    for items in product(*parameters_list):
        all_combinations.append(dict(zip(params_to_sample.keys(), items)))

    # Sample n_samples randomly from all combinations
    rng = np.random.default_rng(123)
    sampled_params = rng.choice(all_combinations, realizations)
    
    pass

    