import numpy
import numpy as np
from common_types import *

class NetworkDense(Network):
    layers: list[Layer]
    def __init__(self, hidden: list[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        

class Dense(Layer):
    weights: numpy.ndarray
    biases: numpy.ndarray
    def __init__(self, n_layers_in: int, n_layers_out: int):
        self.weights = np.random.randn(n_layers_out, n_layers_in)
        self.biases = np.zeros((n_layers_out, 1))


