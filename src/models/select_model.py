"""Module with functions to evaluate models performance, and select best
set of hyperparameters.
"""
from itertools import product

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
from tqdm import tqdm

from ..features.preprocess_data import RScaler_beta


def k_fold_cv(data, model_creator, configuration, cv_splits=5):
    """K-Fold Cross Validation Capable of evaluate Convolutional
    Autoencoders and other Deep Learning Unsupervised Models using
    KerasRegresor wrapper.

    Parameters
    ----------
    data : ``numpy.ndarray``
        Data to be used for kfold cv.
    model_creator : callable function or class instance
        Should construct, compile and return a Keras model,
        which will then be used to fit/predict.
    configuration : ``dict``
        Model parameters and fitting parameters.
    cv_splits : ``int``, default: 5
        Number of folds to be used in cross validaton.

    Returns
    -------
    ``dict``
        Contanins scores mean values over all folds for test and train
        sets.
    """
    # K-Fold cross validation with each hyperparam combination
    cv = KFold(n_splits=cv_splits)

    # Initialize list to save scores
    fold_score = []
    fold_train_score = []

    # Keep only 4k samples to avoide Resourse Exahusted Error
    treshold = 4000

    if len(data) > treshold:
        # Asumes data is shuffled
        data = data[:4000]

    for train_index, test_index in cv.split(data):
        # Split into train and test
        train_set, test_set = data[train_index], data[test_index]

        # Scale each set
        scaler = RScaler_beta().fit(train_set)
        scaled_train = scaler.transform(train_set)
        scaled_test = scaler.transform(test_set)

        # Add extra dimension for ConvAE input
        scaled_train = np.expand_dims(scaled_train, axis=-1)
        scaled_test = np.expand_dims(scaled_test, axis=-1)

        # Generate Model wrapper and compile it
        model_wrapper = KerasRegressor(
            model_creator,
            **configuration,
            epochs=100,
            verbose=0,
            validation_data=(scaled_test, scaled_test)
        )
        history = model_wrapper.fit(scaled_train, scaled_train)

        # Calculate score
        score = history.history["val_mean_squared_error"][-20:]
        fold_score.append(np.mean(score))

        # Train Score
        train_score = history.history["mean_squared_error"][-20:]
        fold_train_score.append(np.mean(train_score))

        # Clear Tensorflow graph
        tf.keras.backend.clear_session()
        del model_wrapper

    return {"score": np.mean(fold_score), "train_score": np.mean(fold_train_score)}


def randomized_search(data, model_creator, parameters_grid, n_samples=25, cv_splits=5):
    """Hyperparameter tunning using random sampling over hyperparameters
    space K-fold Cross Validation is used to get scores for each configuration.

    Parameters
    ----------
    data : ``numpy.ndarray``
        Data to be used for kfold cv.
    model_creator : callable function or class instance
        Should construct, compile and return a Keras model,
        which will then be used to fit/predict.
    parameters_grid : ``dict``
        Dict with model and fitting parameters to
        be adjust. Keys correspondent to each parameter must be named
        as in keras doc.
    n_samples : ``int``, default: 25
        Number of different configurations to be scored.
    cv_splits : ``int``, default: 5
        Number of folds to be used in cross validaton.

    Returns
    -------
    ``tuple``
        pandas.DataFrame with all configuration scores and dict with
        best configuration -with highest test score- parameters.
    """
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
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "init": "glorot_uniform",
            "batch_size": 1024,
        },
    )

    # Variate all over samples
    configurations_score = []
    best_configuration = {"score": 1e4}

    for configuration in tqdm(sampled_hyperparams):
        # Get scores of configuration via kfold cv
        fold_scores = k_fold_cv(data, model_creator, configuration, cv_splits=cv_splits)

        configuration.update(fold_scores)
        configurations_score.append(configuration)

        if configuration["score"] < best_configuration["score"]:
            best_configuration = configuration

        df_scores = pd.DataFrame.from_records(configurations_score).sort_values(
            by="score"
        )

        return df_scores, best_configuration
