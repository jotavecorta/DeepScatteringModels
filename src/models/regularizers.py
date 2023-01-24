import keras.backend
from keras.losses import kullback_leibler_divergence
from keras.regularizers import Regularizer

class KLDivergenceRegularizer(Regularizer):
    def __init__(self, weight, target=.1):
        self.weight = weight
        self.target = target

    def __call__(self, x):
        mean_tensor = keras.backend.mean(x, axis=0)
        return self.weight * (
            kullback_leibler_divergence(self.target, mean_tensor) +
            kullback_leibler_divergence(1. - self.target, 1 - mean_tensor)
            )  