"""Module with functions to scale and preprocess raw data. Final data
will be use for train a Convolutial Autoencoder defined in model 
directory. 

"""
from sklearn.preprocessing import RobustScaler

class Scaler(RobustScaler):
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