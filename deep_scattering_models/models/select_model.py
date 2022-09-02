from itertools import product
import json
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
from tqdm import tqdm 

from ..features.preprocess_data import Scaler

def k_fold_cv(data, model_creator, configuration):
    """"""
    # K-Fold cross validation with each hyperparam combination
    cv = KFold(n_splits=5)

    # Initialize list to save scores
    fold_score = []
    fold_train_score = []

    for train_index, test_index in cv.split(data[:7000]):
        # Split into train and test
        train_set, test_set = data[train_index], data[test_index]
        
        # Scale each set
        scaler = Scaler().fit(train_set)
        scaled_train = scaler.transform(train_set)
        scaled_test = scaler.transform(test_set)

        # Add extra dimension for ConvAE input
        scaled_train = np.expand_dims(scaled_train, axis=-1)
        scaled_test =  np.expand_dims(scaled_test, axis=-1)

        # Generate Model wrapper and compile it
        model_wrapper = KerasRegressor(
            model_creator,
            **configuration,
            nb_epoch=150,
            verbose=0,
            validation_data=(scaled_test, scaled_test)
        )
        history = model_wrapper.fit(scaled_train, scaled_train)

        # Calculate score
        score = history.history['val_mean_squared_error'][-30:]
        fold_score.append(np.mean(score))

        # Train Score
        train_score = history.history['mean_squared_error'][-30:]
        fold_train_score.append(np.mean(train_score))

        # Clear Tensorflow graph
        tf.keras.backend.clear_session()
        del model_wrapper 

        return { 
            'score': np.mean(fold_score),
            'train_score': np.mean(fold_train_score)
            }


def randomized_search(data, model_creator, parameters_grid, n_samples):                   
    """"""
    # Form a list of hyperparameters values
    hyperparameters_list = list(parameters_grid.values())

    # Get all posible combinations of parameters
    all_combinations = []
    for items in product(*hyperparameters_list):
        all_combinations.append(dict(zip(parameters_grid.keys(), items)))

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
    best_configuration = {'score' : 1e4}

    for configuration in tqdm(sampled_hyperparams):
        # Get scores of configuration via kfold cv
        fold_scores = k_fold_cv(data, model_creator, configuration)

        configuration.update(fold_scores)
        configurations_score.append(configuration)
            
        if configuration['score'] < best_configuration['score']:
            best_configuration = configuration 

        df_scores = pd.DataFrame.from_records(configurations_score).sort_values(by='score')    

        return df_scores, best_configuration

def grid_search(data, model_creator):
    pass        

def save_configuration(
    configuration_dict, 
    filename='model_configuration.json',
    scattering_model='spm'
    ):
    # Get data directory path
    src_dir = os.path.normpath(os.getcwd() + '/../..')
    data_dir = os.path.join(src_dir, f'data/{scattering_model}')

    # Guardo la mejor configuración y visualizo el ranking
    json_path = os.path.join(data_dir, filename)

    with open(json_path, 'w') as file_:
        json.dump(configuration_dict, file_, indent=4)

    print(f'Configuration saved at {json_path}')              