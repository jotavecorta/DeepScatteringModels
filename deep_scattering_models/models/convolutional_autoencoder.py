import os
import warnings

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from tensorflow.python.util import deprecation

# Config tf verbosesity 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Config warnings verbosity
deprecation._PRINT_DEPRECATION_WARNINGS = False

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class ConvAutoencoder(Model):
    
    def __init__(self, latent_dim, kernel_init='glorot_uniform'):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([layers.InputLayer(input_shape = (64, 64, 1)), 
                                            layers.Conv2D(4, (6, 6), strides=1, activation = 'relu'),
                                            layers.MaxPooling2D((2, 2), strides=1, padding = 'same'),
                                            
                                            layers.Conv2D(16, (5, 5), strides=2, activation = 'relu'),
                                            layers.MaxPooling2D((2, 2), strides=1, padding = 'same'),
                                            
                                            layers.Conv2D(32, (4, 4), strides=2, activation = 'relu'),
                                            layers.MaxPooling2D((2, 2), strides=1, padding = 'same'),

                                            layers.Conv2D(32, (3, 3), strides=2, activation = 'relu'),
                                            layers.MaxPooling2D((2, 2), strides=1, padding = 'same'),

                                            layers.Flatten(),
                                            
                                            layers.Dense(units=6*6*32, activation='relu', kernel_initializer=kernel_init), 
                                            #kernel_regularizer=tf.keras.regularizers.l2(l=0.1)),
                                            layers.Dropout(.2),

                                            layers.Dense(units=256, activation='relu', kernel_initializer=kernel_init), 
                                            #kernel_regularizer=tf.keras.regularizers.l2(l=0.1)),
                                            layers.Dropout(.2),
                                            
                                            layers.Dense(units=128, activation='relu', kernel_initializer=kernel_init), 

                                            layers.Dense(units=64, activation='relu', kernel_initializer=kernel_init), 
                                            #kernel_regularizer=tf.keras.regularizers.l2(l=0.1)),
                                            #layers.Dropout(.2),

                                            layers.Dense(latent_dim, activation = 'linear')])
      
        
        self.decoder = tf.keras.Sequential([layers.InputLayer(input_shape=(latent_dim,)),

                                            layers.Dense(units=64, activation='relu', kernel_initializer=kernel_init), 
                                            #layers.Dropout(.2),

                                            layers.Dense(units=128, activation='relu', kernel_initializer=kernel_init), 

                                            layers.Dense(units=256, activation='relu', kernel_initializer=kernel_init), 
                                            #kernel_regularizer=tf.keras.regularizers.l2(l=0.1)),
                                            layers.Dropout(.2),                                          

                                            layers.Dense(units=6*6*32, activation='relu', kernel_initializer=kernel_init), 
                                            #kernel_regularizer=tf.keras.regularizers.l2(l=0.1)),
                                            layers.Dropout(.2),
                                            
                                            layers.Reshape(target_shape=(6,6,32)),
                                            
                                            layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu"),

                                            layers.Conv2DTranspose(32, (4, 4), strides=2, activation="relu"),
                                            
                                            layers.Conv2DTranspose(16, (5, 5), strides=2, activation="relu"),

                                            layers.Conv2DTranspose(4, (6, 6), strides=1, activation='relu'),

                                            layers.Conv2D(1, (3, 3), activation="linear", padding="same")])

    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded
  
    def summary(self):
        print(self.encoder.summary())
        print(self.decoder.summary())
