"""Module with functions to generate and save raw data for use
in train and evaluation of unsupervised deep learning models. 

Generated data is composed of polarimetric signatures of random rough
surfaces, calculated using small perturbation method."""
import os
from warnings import warn

import numpy as np
import scipy

from deep_scattering_models.small_perturbation_method.t_matrix import SpmSurface


def init_parameters_grid(grid_length=35):
    """Returns a dictionary with all parameters values to initialize
    multiples SpmSurface class instances.

    Parameters
    ----------
    grid_length : ``int``, default: 35
        Number of elements of every parameter array.

    Returns
    -------
    ``dict`` containing ``numpy.ndarray``'s
        Contanins numpy arrays of length -grid_length- corresponding
        to each SpmSurface class argument -see documentation-.
    """

    # Parameters to sample [cm]: distance btw layers, rms high and corr length
    distance = np.linspace(0.1, 0.7, grid_length)
    rms_high1 = np.linspace(0.004, 0.012, grid_length)
    rms_high2 = np.linspace(0.004, 0.012, grid_length)
    corr_length1, corr_length2 = 6 * rms_high1, 6 * rms_high2

    # Layers 1 and 2 Dielectric Constant
    init_epsilon = 3
    epsilon1 = np.arange(init_epsilon, init_epsilon + grid_length)
    epsilon2 = np.arange(init_epsilon, init_epsilon + grid_length)

    # Returns a dictionary with all parameters shuffled to form grid

    space_grid = {
        "rms_high": rms_high1,
        "corr_length": corr_length1,
        "epsilon": epsilon1,
        "epsilon_2": epsilon2,
        "rms_high_2": rms_high2,
        "corr_length_2": corr_length2,
        "distance": distance,
    }

    return space_grid


def sample_parameters(realizations=20480, seed=123, **kwargs):
    """Returns a generator that yields a dictionary with all parameters 
    names and values necesary to initialize an object of type <SpmSurface class>.
    All values are sampled from a grid generated using ``init_parameters_grid``.

    Parameters
    ----------
    realizations : ``int``, default : 20480
        Number of experiments to run.
    seed : ``int``
        A seed passed to numpy.random.default_rng call.
    **kwargs :
        All additional keyword arguments are passed to
        numpy.random.default_rng call.

    Returns
    -------
    ``generator``
        Dictionary with all SpmSurface class constructor arguments with 
        sampled values.
    """

    # Initialize parameters space grid
    params_to_sample = init_parameters_grid(**kwargs)

    # Form a list of hyperparameters values
    parameters_list = list(params_to_sample.values())

    # Select
    rng = np.random.default_rng(seed)

    # Get all posible combinations of parameters
    for _ in range(realizations):
        # Generates list of randomly picked parameters
        rand_idx = rng.integers(35, size=7)

        if rand_idx[2] == rand_idx[3]:
            # If epsilon1==epsilon2, replace former
            new_set = list(set(np.arange(35)).difference([rand_idx[3]]))
            rand_idx[3] = rng.choice(new_set)

        items = [param[rand_idx[i]] for i, param in enumerate(parameters_list)]

        # Form dicts of parameters for SpmSurface and yield it
        surf_params = dict(zip(params_to_sample.keys(), items))

        yield surf_params


def make_data(realizations=20480, noise=False, size=(45, 90), **kwargs):
    """Create stacked arrays of polarization signatures data from two layer
    random rough surface, for use in deep learning unsupervised models.

    Parameters
    ----------
    realizations : ``int``, default : 20480
        Number of experiments to run.
    noise: ``bool``, default : False
        Add wishart noise to each signature.
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
    lambda_ = 0.245
    k = 2 * np.pi / lambda_

    # Incident Azimut and polar angle [radians]
    theta, phi = 38.5 * np.pi / 180, 0

    # Initialize surface parameters generator
    parameters_generator = sample_parameters(**kwargs)

    # Initialize ndarray for stack polarization signatures
    data = np.zeros((realizations, size[0], size[1]))

    for i in range(realizations):
        # Sample realization of parameters
        surf_parameters = next(parameters_generator)
        surf_parameters.update({"two_layer": True})

        # Realization of surface and polarizarion signature
        surface = SpmSurface(**surf_parameters)
        try:
            signature = surface.polarization_signature(
                lambda_, theta, phi, wishard_noise=noise, grid_size=size
            )
        except scipy.linalg.LinAlgError:
            warn(
                "La Matriz T no satisface las hipótesis "
                f"de Cholesky para los parámetros \n {surf_parameters}",
                RuntimeWarning,
            )
        else:
            # Stack result in data
            data[i, :, :] = signature

    return data


def fixed_parameter_grid(realizations, interval, seed=123, p_name="epsilon"):
    """Returns a generator that yields a dictionary with all parameters 
    names and values necesary to initialize an object of type <SpmSurface class>. 
    All parameters are fixed, exept 'p_name', wich is sample from a flat 
    distribution of boundarys equals to the elements of 'interval'.

    Parameters
    ----------
    realizations : ``int``
        Number of experiments to run.
    interval : ``tuple``
        Interval from which is sample "p_name".    
    seed : ``int``, default : 123
        A seed passed to numpy.random.default_rng call.
    p_name : ``str``
        Name of the <SpmSurface class> constructor argument
        to be sampled. 

    Returns
    -------
    ``generator``
        Dictionary with all SpmSurface class constructor arguments with 
        sampled values.
    """
    
    # Set fixed parameters
    grid = {
        "rms_high": 0.08,
        "corr_length": 6 * 0.008,
        "epsilon": 9.0,
        "epsilon_2": 16.0,
        "rms_high_2": 0.008,
        "corr_length_2": 6 * 0.008,
        "distance": 0.35,
    }

    rng = np.random.default_rng(seed)

    for _ in range(realizations):

        p_value = rng.uniform(*interval)

        if p_name == "epsilon" and p_value == grid["epsilon_2"]:
            # Re-sample if both surfaces has same epsilon
            p_value = rng.uniform(*interval)

        grid.update({p_name: rng.uniform(*interval)})

        yield grid


def make_labeled_data(
    realizations, interval, parameter_name="epsilon", size_out=(45, 90), **kwargs
):
    """Create a tuple that contains a stacked arrays of polarization signatures 
    data from two layer random rough surface and an array with the dielectric constant
    of each surface. For use in machine learning supervised models.

    Parameters
    ----------
    realizations : ``int``
        Number of experiments to run.
    interval : ``int tuple``
        Interval from which ``parameter_name`` is sampled.
    size_out : ``int tuple``, default : (45, 90)
        Shape of each polarization signature.
        (a, b) where: ::
        - a, ellipticity angle length.
        - b, orientation angle length.
    **kwargs :
        All additional keyword arguments are passed to
        t_matrix.polarization_signature call.

    Returns
    -------
    data : ``numpy.ndarray``
        Array of shape (realizations, shape[0], shape[1])
        containing generated data.
    """
    # Surface parameters
    surf_parameters_generator = fixed_parameter_grid(
        realizations, interval, p_name=parameter_name
    )
    label = np.zeros(realizations)

    # Constants parameters: incident wave length and number [cm]
    lambda_ = 0.245
    k = 2 * np.pi / lambda_

    # Incident Azimut and polar angle [radians]
    theta, phi = 38.5 * np.pi / 180, 0

    # Initialize ndarray for stack polarization signatures
    data = np.zeros((realizations, size_out[0], size_out[1]))

    for idx in range(realizations):
        # Realization of surface and polarizarion signature
        param = next(surf_parameters_generator)
        surface = SpmSurface(**param)

        try:
            signature = surface.polarization_signature(
                lambda_,
                theta,
                phi,
                wishard_noise=kwargs.get("noise", False),
                grid_size=size_out,
            )
        except scipy.linalg.LinAlgError:
            warn(
                "La Matriz T no satisface las hipótesis "
                f"de Cholesky para los parámetros \n {surf_parameters_generator}",
                RuntimeWarning,
            )
        else:
            # Stack result in data
            data[idx, :, :] = signature
            label[idx] = param[parameter_name]

    return (data, label)


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
    src_dir = os.path.normpath(os.getcwd() + "/..")

    # Path to data parent folder
    data_dir = os.path.join(src_dir, "data/spm")

    # Saves data in file
    file_path = os.path.join(data_dir, f"{file_name}.npy")

    with open(file_path, "wb") as file_:
        np.save(file_, data)

    print(f"Data saved at {file_path}")


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
    src_dir = os.path.normpath(os.getcwd() + "/..")

    # Path to data parent folder
    data_dir = os.path.join(src_dir, "data/spm")

    # Saves data in file
    file_path = os.path.join(data_dir, f"{file_name}.npy")
    data = np.load(file_path)

    return data