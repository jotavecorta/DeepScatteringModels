"""Module with functions to scale and preprocess raw data. Final data
will be use for train a Convolutial Autoencoder defined in model 
directory. 

"""
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import numpy as np

class RScaler(RobustScaler):
    """Robust Scaler ready to use with numpy.ndarrays of any
     dimentions. Inherits from sklearn.preprocessing.RobustScaler.
     """
    def __init__(self):
        super().__init__()

    def fit(self, X):
        return super().fit(X.reshape(-1, X.shape[-1]), y=None)
    
    def transform(self, X):
        X_shape = X.shape
        scaled = super().transform(X.reshape(-1, X_shape[-1]))
        return scaled.reshape(X_shape)

    def fit_transform(self, X, **fit_params):
        X_shape = X.shape
        transformed = super().fit_transform(X.reshape(-1, X_shape[-1]), y=None, **fit_params) 
        return transformed.reshape(X_shape) 

    def inverse_transform(self, X):
        X_shape = X.shape
        unscaled = super().inverse_transform(X.reshape(-1, X_shape[-1]))  
        return unscaled.reshape(X_shape)  

class MMScaler(MinMaxScaler):
    """MinMax Scaler ready to use with numpy.ndarrays of any
    dimentions. Inherits from sklearn.preprocessing.MinMaxScaler.
    """
    def __init__(self):
        super().__init__()

    def fit(self, X):
        return super().fit(X.reshape(-1, X.shape[-1]), y=None)
    
    def transform(self, X):
        X_shape = X.shape
        scaled = super().transform(X.reshape(-1, X_shape[-1]))
        return scaled.reshape(X_shape)

    def fit_transform(self, X, **fit_params):
        X_shape = X.shape
        transformed = super().fit_transform(X.reshape(-1, X_shape[-1]), y=None, **fit_params) 
        return transformed.reshape(X_shape)   

    def inverse_transform(self, X):
        X_shape = X.shape
        unscaled = super().inverse_transform(X.reshape(-1, X_shape[-1]))  
        return unscaled.reshape(X_shape)         

def to_dB(data):
    # Replace zeros to take log
    no_zeros_data = np.where(data>0.0, data, 1e-9)
    
    return 10*np.log10(no_zeros_data)
