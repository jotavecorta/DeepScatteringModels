import os
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.python.util import deprecation

# Config tf verbosesity
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Config warnings verbosity
deprecation._PRINT_DEPRECATION_WARNINGS = False

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ConvAutoencoder(Model):
    def __init__(self, latent_dim, input_shape, conv_layers=None, dense_layers=None):
        super().__init__()
        self.latent_dim = latent_dim
        
        self._input_shape = input_shape
        self._match_shape = 0
        
        self.conv_layers = {} if conv_layers is None else conv_layers
        self.dense_layers = {} if dense_layers is None else dense_layers

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

        for filter, kernel, stride in conv_config.get(
            "layers_config", [(4, (6, 6), 1)]
        ):

            model.add(
                layers.Conv2D(
                    filter,
                    kernel,
                    strides=stride,
                    activation=conv_activation,
                    kernel_initializer=conv_init,
                )
            )

        # Add dense layers
        # Get activation and kernel initializer
        dense_activation = dense_config.get("activation", "relu")
        kernel_init = dense_config.get("kernel_initializer", "normal")

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

        # Add latent space layer
        model.add(layers.Dense(units=self.latent_dim, activation="linear"))

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

        for neurons in reversed(dense_config.get("layers_units", (16,))):
            model.add(
                layers.Dense(
                    units=neurons,
                    activation=dense_activation,
                    kernel_initializer=kernel_init,
                )
            )        
        
        # Add intermediate layer to match dense and convolutinal layers
        model.add(layers.Dense(units=np.prod(self._match_shape)))    
        model.add(layers.Reshape(target_shape=self._match_shape))
        
        # Add Convolutional layers
        conv_activation = conv_config.get("activation", "relu")
        conv_init = conv_config.get("kernel_initializer", "normal")

        for filter, kernel, stride in reversed(conv_config.get(
            "layers_config", [(4, (6, 6), 1)])
        ):

            model.add(
                layers.Conv2DTranspose(
                    filter,
                    kernel,
                    strides=stride,
                    activation=conv_activation,
                    kernel_initializer=conv_init,
                )
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


def save_model(model, configuration):
    pass
