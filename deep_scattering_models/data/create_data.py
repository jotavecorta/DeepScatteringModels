#%%
"""Script with functions to generate and save raw data for use
in train and evaluation of unsupervised deep learning models. 

Generated data is composed of polarimetric signatures of random rough
surfaces, calculated using small perturbation method."""
from itertools import product
import os

import numpy as np

from deep_scattering_models.small_perturbation_method.t_matrix import SpmSurface


def init_parameters_grid(grid_length=35, seed=123):
    """Returns a dictionary with all parameters values to initialize
    multiples SpmSurface class instances.
    
    Parameters
    ----------
    grid_length : ``int``, default: 35         
        Number of elements of every parameter array.
    seed : ``int``       
        A seed passed to numpy.random.default_rng call.

    Returns
    -------
    ``dict`` containing ``numpy.ndarray``'s      
        Contanins numpy arrays of length -grid_length- corresponding
        to each SpmSurface class argument -see documentation-.
    """

    # Parameters to sample [cm]: distance btw layers, rms high and corr length
    distance = np.linspace(.1, .7, grid_length)
    rms_high1 = np.linspace(.004, .012, grid_length)
    rms_high2 = np.linspace(.004, .012, grid_length)
    corr_length1, corr_length2 = 6*rms_high1, 6*rms_high2

    # Layers 1 and 2 Dielectric Constant
    init_epsilon = 3
    epsilon1 = np.arange(init_epsilon, init_epsilon + grid_length) 
    epsilon2 = np.arange(init_epsilon, init_epsilon + grid_length)

    # Returns a dictionary with all parameters shuffled to form grid
    rng = np.random.default_rng(seed)
    space_grid = {
        'rms_high' : rng.permutation(rms_high1),
        'corr_length' : rng.permutation(corr_length1), 
        'epsilon' : rng.permutation(epsilon1),
        'epsilon_2' : rng.permutation(epsilon2), 
        'rms_high_2' : rng.permutation(rms_high2), 
        'corr_length_2' : rng.permutation(corr_length2),
        'distance' : rng.permutation(distance),
    }

    return space_grid

def sample_parameters(**kwargs):
    """Returns a generator with all parameters values to initialize
    SpmSurface class.
    
    Parameters
    ----------
    **kwargs :        
        All additional keyword arguments are passed to 
        numpy.random.default_rng call.

    Returns
    -------
    ``generator``     
        Dictionary with all SpmSurface class argument values sampled.
    """    

    # Initialize parameters space grid
    params_to_sample = init_parameters_grid(**kwargs)

    # Form a list of hyperparameters values
    parameters_list = list(params_to_sample.values())

    # Get all posible combinations of parameters
    for items in product(*parameters_list):
        # Form dicts of parameters for SpmSurface and yield it
        surf_params = dict(zip(params_to_sample.keys(), items))

        yield  surf_params

def make_data(realizations=20480, noise=False, size=(45, 90), **kwargs):
    """Create stacked arrays of polarization signatures data from two layer 
    random rough surface, for use in deep learning unsupervised models. 
    
    Parameters
    ----------
    realizations : ``int``, default : 20480
        Number of experiments to run.
    noise: ``bool``, default : False
        Add wishard noise to each signature.
    size : ``int tuple``, default : (45, 90)
        Shape of each polarization signature.
        (a, b) where: ::
        - a, ellipticity angle length.
        - b, orientation angle length.  
    **kwargs :      
        All additional keyword arguments are passed to 
        numpy.random.default_rng call.

    Returns
    -------
    data : ``numpy.ndarray``     
        Array of shape (realizations, shape[0], shape[1]) 
        containing generated data.
    """ 

    # Constants parameters: incident wave length and number [cm]
    lambda_ = .245
    k = 2 * np.pi / lambda_

    # Incident Azimut and polar angle [radians]
    theta, phi = 38.5*np.pi/180, 0

    # Initialize surface parameters generator
    parameters_generator = sample_parameters(**kwargs)

    # Initialize ndarray for stack polarization signatures
    data = np.zeros((realizations, size[0], size[1]))

    for i in range(realizations):
        # Sample realization of parameters
        surf_parameters = next(parameters_generator)
        surf_parameters.update({'two_layer' : True})

        # Realization of surface and polarizarion signature
        surface = SpmSurface(**surf_parameters)
        signature = surface.polarization_signature(
            lambda_, 
            theta,
            phi,
            wishard_noise=noise,
            grid_size=size
        )

        # Stack result in data
        data[i, :, :] = np.real(signature)

    return data

def save_data(file_name, data):
    """Saves data into data directory, on the repository.
    
    Parameters
    ----------
    file_name : ``str``       
        Name of the file.

    data : ``numpy.ndarray``       
        Array containing data.     
    """

    # Path to parent src directory
    src_dir = os.path.normpath(os.getcwd() + '/../..')

    # Path to data parent folder
    data_dir = os.path.join(src_dir, 'data/spm')

    # Saves data in file
    file_path = os.path.join(data_dir, file_name)
    np.save(file_path, data)

    print(f'Data saved at {file_path}')

def load_data(file_name):
    """Load data from data directory, on the repository.
    
    Parameters
    ----------
    file_name : ``str``       
        Name of the file.

    Returns
    -------
    data : ``numpy.ndarray``       
        Array containing data.     
    """
    
    # Path to parent src directory
    src_dir = os.path.normpath(os.getcwd() + '/../..')

    # Path to data parent folder
    data_dir = os.path.join(src_dir, 'data/spm')

    # Saves data in file
    file_path = os.path.join(data_dir, f'{file_name}.npy')
    data = np.load(file_path)

    return data


# %%
