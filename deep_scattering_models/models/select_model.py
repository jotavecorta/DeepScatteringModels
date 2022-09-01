from itertools import product

import numpy as np

from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor

from ..features.preprocess_data import Scaler

# Hyperparameter Space
l_rate = [0.00005, 0.0001, 0.0005, 0.001, 0.003]
kernel_init = ['glorot_uniform', 'lecun_uniform', 'normal']
batch = [32, 64, 256, 512, 1024]
grad_optimizer = ['adam', 'RMSProp', 'sgd']

space_grid = {
    'optimizer': grad_optimizer,
    'learning_rate': l_rate,
    'init': kernel_init,
    'batch_size': batch
                   }
# N samples for randomized search                   
n_samples = 50

# Form a list of hyperparameters values
hyperparameters_list = list(space_grid.values())

# Get all posible combinations of parameters
all_combinations = []
for items in product(*hyperparameters_list):
    all_combinations.append(dict(zip(space_grid.keys(), items)))

# Sample n_samples randomly from all combinations
rng = np.random.default_rng(123)
sampled_hyperparams = rng.choice(all_combinations, n_samples)

# Add default configuration
sampled_hyperparams = np.append(
    sampled_hyperparams, 
    {
        'optimizer': 'adam', 
        'learning_rate': 0.0001, 
        'init': 'glorot_uniform', 
        'batch_size': 1024
        }
        )

# Variate all over samples 
configurations_score = []
best_score = 1e4
for params in sampled_hyperparams:

    # K-Fold cross validation with each hyperparam combination
    cv = KFold(n_splits=5)

    # Initialize list to save scores
    fold_score = []
    fold_train_score = []

    for train_index, test_index in cv.split(sigma_train[:7000]):
        # Split into train and test
        S_train, S_test = sigma_train[train_index], sigma_train[test_index]
        
        # Scale each set
        scaler = Scaler().fit(S_train)
        scaled_train = scaler.transform(S_train)
        scaled_test = scaler.transform(S_test)

        # Add extra dimension for ConvAE input
        scaled_train = np.expand_dims(scaled_train, axis=-1)
        scaled_test =  np.expand_dims(scaled_test, axis=-1)

        # Generate Model wrapper and compile it
        conv_ae = KerasRegressor(
            create_autoencoder,
            **params,
            nb_epoch=150,
            verbose=0,
            validation_data=(scaled_test, scaled_test)
        )
        history = conv_ae.fit(scaled_train, scaled_train)

        # Calculate score
        score = history.history['val_mean_squared_error'][-30:]
        fold_score.append(np.mean(score))

        # Train Score
        train_score = history.history['mean_squared_error'][-30:]
        fold_train_score.append(np.mean(train_score))

        # Clear Tensorflow graph
        tf.keras.backend.clear_session()
        del conv_ae 

    run_data = {
        'params' : params, 
        'score': np.mean(fold_score),
        'train_score': np.mean(fold_train_score)
        }
    configurations_score.append(run_data)
        
    if run_data['score'] < best_score:
        best_configuration = run_data          

    print(f'Parameters Configuration: {params}, score: {np.mean(fold_score)}')  