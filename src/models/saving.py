""" Code for model and configuration loading/saving. """
import json
import os

from tensorflow import keras


def save_configuration(
    configuration_dict, filename="model_configuration", scattering_model="spm"
):
    """Saves a model configuration as a json file.

    Parameters
    ----------
    configuration_dict : ``dict``
        Dictionary with model parameters as keys.
    filename : ``str``, default: 'model_configuration'
        Name of the file.
    scattering_model : ``str``, default: 'spm'
        The EMS model used to produce data.
    """
    # Get models directory path
    src_dir = os.path.normpath(os.getcwd() + "/..")
    model_dir = os.path.join(src_dir, "models")

    # Create path to json file
    filename = f"{filename}_{scattering_model}.json"
    json_path = os.path.join(model_dir, filename)

    with open(json_path, "w") as file_:
        json.dump(configuration_dict, file_, indent=4)

    print(f"Configuration saved at {json_path}")


def load_configuration(config_filename, scattering_model="spm"):
    """Load a model configuration json file.

    Parameters
    ----------
    config_filename : ``str``
        Name of the file containing the CAE configuration
        in JSON format.

    Returns
    -------
    configuration_dict : ``dict``
        Dictionary with model parameters as keys.
    """
    # Get models directory path
    src_dir = os.path.normpath(os.getcwd() + "/..")
    model_dir = os.path.join(src_dir, "models")

    filename = f"{config_filename}_{scattering_model}.json"
    json_path = os.path.join(model_dir, filename)

    with open(json_path, "r") as file_:
        config_dict = json.load(file_)

    return config_dict


def save_model(model, configuration_dict, name="cae"):
    """Saves a model configuration as a json file.

    Parameters
    ----------
    configuration_dict : ``dict``
        Dictionary with model parameters as keys.
    filename : ``str``, default: 'model_configuration'
        Name of the file.
    """
    # Get models directory path
    src_dir = os.path.normpath(os.getcwd() + "/..")
    model_dir = os.path.join(src_dir, f"models")

    # Save model and weights into hdf5 file
    model_filename = f"{name}_model_weights"
    model_path = os.path.join(model_dir, model_filename)
    model.save(model_path, save_format="tf")

    # Save configuration into json file
    config_filename = f"{name}_configuration"
    save_configuration(configuration_dict, filename=config_filename)

    print(f"Model and weights saved at {model_path}")


def load_model(name="cae"):
    """Saves a model configuration as a json file.

    Parameters
    ----------
    configuration_dict : ``dict``
        Dictionary with model parameters as keys.
    filename : ``str``, default: 'model_configuration'
        Name of the file.
    """
    # Get models directory path
    src_dir = os.path.normpath(os.getcwd() + "/../")
    model_dir = os.path.join(src_dir, f"models")

    # Load model and weights
    model_filename = f"{name}_model_weights"
    model_path = os.path.join(model_dir, model_filename)
    model = keras.load_model(model_path)

    # Load configuration from json file
    config_filename = f"{name}_configuration"
    configuration_dict = load_configuration(config_filename=config_filename)

    return model, configuration_dict
