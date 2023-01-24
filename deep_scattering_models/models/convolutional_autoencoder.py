import os
import warnings

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
from tensorflow.python.util import deprecation

from deep_scattering_models.models.select_model import (
    save_configuration, 
    load_configuration
    )

# Config tf verbosesity
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Config warnings verbosity
deprecation._PRINT_DEPRECATION_WARNINGS = False

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ConvAutoencoder(Model):
    def __init__(self, latent_dim, input_shape, conv_layers=None, dense_layers=None, sparse=False):
        super().__init__()
        self.latent_dim = latent_dim
        
        self._input_shape = input_shape
        self._match_shape = 0
        
        self.conv_layers = {} if conv_layers is None else conv_layers
        self.dense_layers = {} if dense_layers is None else dense_layers

        self.sparse = sparse

        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()


    def _create_encoder(self):
        # Rename layers configuration
        dense_config = self.dense_layers
        conv_config = self.conv_layers

        input_shape = self._input_shape

        # Create a secuential model with input shape
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=input_shape))

        # Add Convolutional layers
        conv_activation = conv_config.get("activation", "relu")
        conv_init = conv_config.get("kernel_initializer", "normal")

        conv_pooling = conv_config.get("max_pooling", False)

        for filter, kernel, stride in conv_config.get(
            "layers_config", [(4, (6, 6), 1)]
        ):
            # Convolutional Layer
            model.add(
                layers.Conv2D(
                    filter,
                    kernel,
                    strides=stride,
                    activation=conv_activation,
                    kernel_initializer=conv_init,
                )
            )

            # Add Max Pooling layer
            if conv_pooling:
                model.add(
                    layers.MaxPooling2D((2, 2), strides=1, padding = 'same')
                )

        # Add dense layers
        # Get activation and kernel initializer
        dense_activation = dense_config.get("activation", "relu")
        kernel_init = dense_config.get("kernel_initializer", "normal")

        drop_out = dense_config.get("dropout", False)

        # Add intermediate layer to match dense and convolutinal layers
        self._match_shape = model.layers[-1].output_shape[1:]
        model.add(layers.Flatten())
        model.add(layers.Dense(units=np.prod(self._match_shape)))
        

        for neurons in dense_config.get("layers_units", (16,)):
            model.add(
                layers.Dense(
                    units=neurons,
                    activation=dense_activation,
                    kernel_initializer=kernel_init,
                )
            )

            # Add dropout
            if drop_out:
                model.add(
                    layers.Dropout(.2)
                )

        # Add latent space layer
        model.add(layers.Dense(units=self.latent_dim, activation="linear"))

        if self.sparse:
            model.add(layers.ActivityRegularization(l1=1e-4))

        return model

    def _create_decoder(self):
        # Rename layers configuration
        dense_config = self.dense_layers
        conv_config = self.conv_layers
        
        # Create a secuential model with input shape
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(self.latent_dim,)))
       
        # Add dense layers
        # Get activation and kernel initializer
        dense_activation = dense_config.get("activation", "relu")
        kernel_init = dense_config.get("kernel_initializer", "normal")

        drop_out = dense_config.get("dropout", False)

        for neurons in reversed(dense_config.get("layers_units", (16,))):
            # Dense layer
            model.add(
                layers.Dense(
                    units=neurons,
                    activation=dense_activation,
                    kernel_initializer=kernel_init,
                )
            )

            # Add dropout
            if drop_out:
                model.add(
                    layers.Dropout(.2)
                )        
        
        # Add intermediate layer to match dense and convolutinal layers
        model.add(layers.Dense(units=np.prod(self._match_shape)))    
        model.add(layers.Reshape(target_shape=self._match_shape))
        
        # Add Convolutional layers
        conv_activation = conv_config.get("activation", "relu")
        conv_init = conv_config.get("kernel_initializer", "normal")

        conv_pooling = conv_config.get("max_pooling", False)

        for filter, kernel, stride in reversed(conv_config.get(
            "layers_config", [(4, (6, 6), 1)])
        ):
            # Convolutional layer
            model.add(
                layers.Conv2DTranspose(
                    filter,
                    kernel,
                    strides=stride,
                    activation=conv_activation,
                    kernel_initializer=conv_init,
                )
            )

            # Add Max Pooling layer
            if conv_pooling:
                model.add(
                    layers.MaxPooling2D((2, 2), strides=1, padding = 'same')
                )            

        # Add reshape layer
        model.add(layers.Conv2D(1, (3, 3), activation="linear", padding="same"))

        return model

    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded

    def summary(self):
        print(self.encoder.summary())
        print(self.decoder.summary())


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
    model.save(model_path, save_format='tf')

    # Save configuration into json file
    config_filename = f"{name}_configuration"
    save_configuration(
        configuration_dict, 
        filename=config_filename
        )  

    print(f"Model and weights saved at {model_path}")


def load_cae_model(name="cae"):
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
    model = load_model(model_path)

    # Load configuration from json file
    config_filename = f"{name}_configuration"
    configuration_dict = load_configuration( 
        config_filename=config_filename
        )  

    return model, configuration_dict
