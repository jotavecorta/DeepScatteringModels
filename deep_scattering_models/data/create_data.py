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

    # Layers 1 and 2 Dielectric Constant
    init_epsilon = 3
    epsilon1 = np.arange(init_epsilon, init_epsilon + grid_length)
    epsilon2 = np.arange(init_epsilon, init_epsilon + grid_length)

    # Returns a dictionary with all parameters shuffled to form grid

    space_grid = {
        "rms_high": rms_high1,
        "epsilon": epsilon1,
        "epsilon_2": epsilon2,
        "rms_high_2": rms_high2,
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
        rand_idx = rng.integers(35, size=5)

        if rand_idx[1] == rand_idx[2]:
            # If epsilon1==epsilon2, replace former
            new_set = list(set(np.arange(35)).difference([rand_idx[2]]))
            rand_idx[2] = rng.choice(new_set)

        items = [param[rand_idx[i]] for i, param in enumerate(parameters_list)]

        # Form dicts of parameters for SpmSurface and yield it
        surf_params = dict(zip(params_to_sample.keys(), items))
        surf_params.update({"corr_length" : surf_params["rms_high"] * 6})
        surf_params.update({"corr_length_2" : surf_params["rms_high_2"] * 6})

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


def fixed_parameter_grid(realizations, variables, seed=123):
    """Returns a generator that yields a dictionary with all parameters 
    names and values necesary to initialize an object of type <SpmSurface class>. 
    All parameters are fixed, exept 'p_name', wich is sample from a flat 
    distribution of boundarys equals to the elements of 'interval'.

    Parameters
    ----------
    realizations : ``int``
        Number of experiments to run.
    variables : ``dict``
        Keys, name of the <SpmSurface class> constructor argument
        to be sampled. Values, interval from which the variable will
        be sample.    
    seed : ``int``, default = 123
        A seed passed to numpy.random.default_rng call.

    Returns
    -------
    ``generator``
        Dictionary with all SpmSurface class constructor arguments with 
        sampled values.
    """

    # Set fixed parameters
    grid = {
        "rms_high": 0.008,
        "corr_length": 6 * 0.008,
        "epsilon": 9.0,
        "epsilon_2": 17.0,
        "rms_high_2": 0.006,
        "corr_length_2": 6 * 0.006,
        "distance": 0.2,
    }

    rng = np.random.default_rng(seed)

    for _ in range(realizations):
        
        for key, interval in variables.items():

            p_value = rng.uniform(*interval)

            # Re-sample if both surfaces has same epsilon
            while (key == "epsilon" and p_value == grid["epsilon_2"]):
                
                p_value = rng.uniform(*interval)

            grid.update({key: p_value})

            # Corrects rms_high - corr_length quotent
            if ("rms_high" in key):
                grid.update({"corr_length": grid["rms_high"] * 6})
                grid.update({"corr_length_2": grid["rms_high_2"] * 6})

        yield grid


def make_labeled_data(
    realizations, variables=None, size_out=(45, 90), **kwargs
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
        realizations, variables = variables
    )

    # Set default variables
    variables = {"epsilon" : (3, 35)} if variables is None else variables

    # Constants parameters: incident wave length and number [cm]
    lambda_ = 0.245
    k = 2 * np.pi / lambda_

    # Incident Azimut and polar angle [radians]
    theta, phi = 38.5 * np.pi / 180, 0

    # Initialize ndarrays for stack polarization signatures and labels
    data = np.zeros((realizations, size_out[0], size_out[1]))
    labels = np.zeros((realizations, len(variables)))

    for idx in range(realizations):
        # Realization of surface and polarizarion signature
        params = next(surf_parameters_generator)
        params.update({"two_layer": True})
        surface = SpmSurface(**params)

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
            labels[idx, :] = np.array([params[key] for key in variables])

    return (data, labels)


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

# def main():
    # # Constants parameters: incident wave length and number [cm]
    # lambda_ = 0.245

    # # Incident Azimut and polar angle [radians]
    # theta, phi = 38.5 * np.pi / 180, 0

    # Test make_labeled_surface
    # # Initialize surface parameters generator 
    # data, label = make_labeled_data(
    #     realizations=10,
    #     variables={
    #         "epsilon" : (3, 35),
    #         "rms_high" : (0.004, 0.012)
    #     },        
    # )

    # print(f"Tamaño de los datos: {data.shape}")
    # print(f"Tamaño de las etiquetas: {label.shape}")
    # print(label)

    # # Test fixed_parameter_grid
    # parameters_generator = fixed_parameter_grid(
    #     realizations= 2048,
    #     variables={
    #         "epsilon" : (3, 35),
    #         "rms_high" : (0.004, 0.012)
    #     },
    #     seed=435
    #     )  
    # parameters = next(parameters_generator)
    # parameters.update({"two_layer": True})

    # # Calculate polarimetric signature    
    # surf = SpmSurface(**parameters)
    # signature = surf.polarization_signature(lambda_, theta, phi)

    # iterations = 0
    # while (np.any(signature <= 0.0)):
    #     iterations += 1
    #     try:
    #         parameters = next(parameters_generator)
    #         parameters.update({"two_layer": True})
    #     except:
    #         raise (
    #             "No se encontraron los parámetros luego de"
    #             f"{iterations} iteraciones."
    #         )    
    #     else:    
    #         surf = SpmSurface(**parameters)
    #         signature = surf.polarization_signature(lambda_, theta, phi)

    # print(parameters)
    # print(iterations)

# if __name__ == "__main__":
#     main()        

