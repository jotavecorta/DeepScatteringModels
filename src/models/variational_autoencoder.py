import tensorflow as tf
from tensorflow.keras import layers

from .convolutional_autoencoder import ConvAutoencoder


def kl_loss(z_mean, z_log_var):
    return -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalAutoencoder(ConvAutoencoder):
    def __init__(self, latent_dim, input_shape, conv_layers=None, dense_layers=None):
        super().__init__(
            latent_dim, input_shape, conv_layers=conv_layers, dense_layers=dense_layers
        )
        self._latent_mean = layers.Dense(latent_dim)
        self._latent_log_var = layers.Dense(latent_dim)
        self._sampling = Sampling()
        self.encoder = self._create_encoder()

    def _create_encoder(self):
        # Use inheritate architecture
        encoder = super()._create_encoder()

        # Remove latent-space layer and add variational layer
        encoder.pop()

        return encoder

    def call(self, inputs):
        z = self.encoder(inputs)
        z_mean = self._latent_mean(z)
        z_log_var = self._latent_log_var(z)
        encoded = self._sampling(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        self.add_loss(kl_loss(z_mean, z_log_var))
        self.add_loss(tf.keras.losses.mean_squared_error(inputs, decoded))
        return decoded
