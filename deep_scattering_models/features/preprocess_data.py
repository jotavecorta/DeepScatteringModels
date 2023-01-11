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
        return super().fit(X.reshape(X.shape[0], -1), y=None)
    
    def transform(self, X):
        X_shape = X.shape
        scaled = super().transform(X.reshape(X.shape[0], -1))
        return scaled.reshape(X_shape)

    def fit_transform(self, X, **fit_params):
        X_shape = X.shape
        transformed = super().fit_transform(X.reshape(X.shape[0], -1), y=None, **fit_params) 
        return transformed.reshape(X_shape) 

    def inverse_transform(self, X):
        X_shape = X.shape
        unscaled = super().inverse_transform(X.reshape(X.shape[0], -1))  
        return unscaled.reshape(X_shape)  

class RScaler_beta(RobustScaler):
    """Robust Scaler ready to use with numpy.ndarrays of any
     dimentions. Inherits from sklearn.preprocessing.RobustScaler.
     """
    def __init__(self):
        super().__init__()

    def fit(self, X):
        return super().fit(X.reshape((-1, 1)), y=None)
    
    def transform(self, X):
        X_shape = X.shape
        scaled = super().transform(X.reshape((-1, 1)))
        return scaled.reshape(X_shape)

    def fit_transform(self, X, **fit_params):
        X_shape = X.shape
        transformed = super().fit_transform(X.reshape((-1, 1)), y=None, **fit_params) 
        return transformed.reshape(X_shape) 

    def inverse_transform(self, X):
        X_shape = X.shape
        unscaled = super().inverse_transform(X.reshape((-1, 1)))  
        return unscaled.reshape(X_shape)        

class MMScaler(MinMaxScaler):
    """MinMax Scaler ready to use with numpy.ndarrays of any
    dimentions. Inherits from sklearn.preprocessing.MinMaxScaler.
    """
    def __init__(self):
        super().__init__()

    def fit(self, X):
        return super().fit(X.reshape((-1, 1)), y=None)
    
    def transform(self, X):
        X_shape = X.shape
        scaled = super().transform(X.reshape((-1, 1)))
        return scaled.reshape(X_shape)

    def fit_transform(self, X, **fit_params):
        X_shape = X.shape
        transformed = super().fit_transform(X.reshape((-1, 1)), y=None, **fit_params) 
        return transformed.reshape(X_shape)   

    def inverse_transform(self, X):
        X_shape = X.shape
        unscaled = super().inverse_transform(X.reshape((-1, 1)))  
        return unscaled.reshape(X_shape)         

def to_dB(data):
    # Replace zeros to take log    
    no_zeros_data = np.where(data>0.0, data, data[data>0.0].min())
    
    return 10*np.log10(no_zeros_data)

def remove_outliers(data, k=1.5):
    """Removes surfaces samples with values outside interquartile range
    
    Parameters
    ----------
    data : ``numpy.ndarray``
        Stack of surface polarization signature samples of shape
        (n_samples, 45, 90).
    k : ``int``, default: 1.5
        Scale of the range. Data outside de interval 
        q1-k*iqr < data < q3+k*iqr, with q1 and q3 first and third quartiles,
        is removed.  

    Returns
    -------
    ``numpy.ndarray``
        Stack of surface polarization signature samples with no outliers.
        shape (n_samples_no_ouliers, 45, 90).             
    """    
    shape = data.shape
    
    # Delete entries with negative data
    reshaped_data = np.reshape(data, (shape[0], -1))
    reshaped_data = reshaped_data[np.all(reshaped_data>=-1e-9, axis=1)]

    # Calculation of quartiles an interquartil range
    first_quartile = np.quantile(reshaped_data, .25)
    third_quartile = np.quantile(reshaped_data, .75)
    iqr = third_quartile - first_quartile

    # Make a mask to filter samples
    upper_bound = np.all(
        reshaped_data < third_quartile + k * iqr, 
        axis=1
        )

    lower_bound = np.all(
        reshaped_data > first_quartile - k * iqr,
        axis=1
    )    

    mask = np.logical_and(lower_bound, upper_bound)
    
    return np.reshape(reshaped_data[mask], (-1, shape[1], shape[2]))       

