"""Module with build functions to use as input in Keras wrappers,
in particular in KerasRegressor object.

Different builders were write to use in different optimization
stages, such as finding the best model architecture or tune 
hyperparameters like optimizer type, learning rate, batch size, etc.

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers as opt
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers.schedules import PolynomialDecay, PiecewiseConstantDecay
from tensorflow.keras.optimizers.schedules import InverseTimeDecay, ExponentialDecay

from src.models.convolutional_autoencoder import ConvAutoencoder

# Global parameters
FINAL_LEARNING_RATE = 0.00005
TRAIN_SIZE = 7000

def build_cae_architecture(
    latent_dimension=10,
    conv_layers_config=None, 
    dense_layers_config=None,
    input_shape = (45, 90, 1)
    ):
    """Returns a compiled ConvAutoencoder model with
    the desired architecture.

    Parameters
    ----------
    latent_dimension : ``int``, default: 10         
        Number of units that connects encoder and decoder in the
        model.
    conv_layers_config : ``dict``, default: None
        To use as input in ``conv_layers`` in ConvAutoencoder object
        (see documentation).
    dense_layers_config : ``dict``, default: None
        To use as input in ``dense_layers`` in ConvAutoencoder object
        (see documentation).        
    input_shape : ``tuple``, default: (45, 90, 1)
        Shape of the data to be used in training.
                
    Returns
    -------
    model : ``models.convolutional_autoencoder.ConvAutoencoder``      
        Compiled ConvAutoencoder object.
    
    """
    # Set TensorFlow seed for reproducible results                   
    tf.random.set_seed(123) 

    # Create and Compile Model
    model = ConvAutoencoder(
        latent_dimension, 
        input_shape,
        conv_layers=conv_layers_config,
        dense_layers=dense_layers_config
    )
    model.compile(
        optimizer="adam",
        loss=MeanSquaredError(),
        metrics=['mean_squared_error', 'mean_absolute_error']
        )

    return model    

def build_vae_architecture(
    latent_dimension=10,
    conv_layers_config=None, 
    dense_layers_config=None,
    optimizer = "adam",
    input_shape = (45, 90, 1)
    ):
    """Returns a compiled Variational Autoencoder model with
    the desired architecture.

    Parameters
    ----------
    latent_dimension : ``int``, default: 10         
        Number of units that connects encoder and decoder in the
        model.
    conv_layers_config : ``dict``, default: None
        To use as input in ``conv_layers`` in ConvAutoencoder object
        (see documentation).
    dense_layers_config : ``dict``, default: None
        To use as input in ``dense_layers`` in ConvAutoencoder object
        (see documentation).  
    optimizer : ``str`` or ``tf.keras.optimizers.Optimizer``
        Gradient Descent optimizer. Could be a string identifier e.g 'adam'.         
    input_shape : ``tuple``, default: (45, 90, 1)
        Shape of the data to be used in training.
                
    Returns
    -------
    model : ``models.variational_autoencoder.VariationalAutoencoder``      
        Compiled VariationalAutoencoder object.
    
    """
    # Set TensorFlow seed for reproducible results                   
    tf.random.set_seed(123) 

    # Create and Compile Model
    model = VariationalAutoencoder(
        latent_dimension, 
        input_shape,
        conv_layers=conv_layers_config,
        dense_layers=dense_layers_config
    )
    model.compile(
        optimizer=optimizer,
        loss=MeanSquaredError()
        )

    return model    


def create_autoencoder(optimizer='adam',
                       learning_rate=0.0001,
                       beta_momentum=.9,
                       init='glorot_uniform'):
    # Set TensorFlow seed for reproducible results                   
    tf.random.set_seed(123)                   

    # Create optimazer
    select_optimizer = {
        'adam': opt.Adam(learning_rate=learning_rate, beta_1=beta_momentum),
        'RMSProp': opt.RMSprop(learning_rate=learning_rate),
        'sgd': opt.SGD(learning_rate=learning_rate)
                        }
    gradient_descent = select_optimizer.get(optimizer, 'adam')

    # Create and Compile Model
    model = ConvAutoencoder(12, kernel_init=init)
    model.compile(
        optimizer=gradient_descent,
        loss=MeanSquaredError(),
        metrics=['mean_squared_error', 'mean_absolute_error']
        )

    return model


def rmsProp_with_decay(
    latent_dimension = 12,
    initial_value=0.0003, 
    final_value=FINAL_LEARNING_RATE, 
    train_samples=TRAIN_SIZE,
    epochs=300, 
    batch_size=32, 
    decay_type='polynomial',
    centered=False
    ):
    # Set TensorFlow seed for reproducible results
    tf.random.set_seed(123)

    # Continous decay parameters
    decay_rate = (initial_value - final_value)/epochs
    decay_steps = int(train_samples / batch_size) 
    total_steps = decay_steps*epochs
    inverse_decay_steps = (final_value*decay_rate*total_steps / 
                            (initial_value - final_value))
    
    # Piecewise decay parameters
    steps = 6
    step_len = int(total_steps / steps)
    step_decay = (initial_value - final_value) / steps

    schedulers_type = {
        'polynomial' : PolynomialDecay(
            initial_learning_rate=initial_value, 
            decay_steps=total_steps, 
            end_learning_rate=FINAL_LEARNING_RATE, 
            power=3
            ), 
        'step' : PiecewiseConstantDecay(
            boundaries=[bound for bound in range(step_len, total_steps, step_len)],
            values=[value for value in np.arange(initial_value, final_value, -step_decay)]
            ),
        'inverse' : InverseTimeDecay(
            initial_learning_rate=initial_value, 
            decay_steps=inverse_decay_steps, 
            decay_rate=decay_rate,
            ), 
        'exponential' : ExponentialDecay(
            initial_learning_rate=initial_value,
            decay_steps=total_steps,
            decay_rate=final_value/initial_value
            ),
        'constant' : initial_value
        }

    scheduler = schedulers_type.get(decay_type, 'polynomial')

    # Gradient descent optimizer
    opt = tf.keras.optimizers.RMSprop(    
        learning_rate=scheduler,
        rho=0.9,
        momentum=0.0,
        centered=centered,
        name=f'RMSprop_{decay_type}'
        )

    # Create and Compile Model
    model = ConvAutoencoder(latent_dimension, kernel_init='normal')
    model.compile(
        optimizer=opt,
        loss=MeanSquaredError(),
        metrics=['mean_squared_error', 'mean_absolute_error']
        )    

    return model