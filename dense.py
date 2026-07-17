import numpy
import numpy as np
from common_types import *

class NetworkDense(Network):
    layers: list[Layer]
    def __init__(self, hidden: list[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = []
        prev_n = 460 * 215 * 3
        for curr_n in hidden:
            self.layers.append(Dense(prev_n, curr_n))
            self.layers.append(Sigmoid())
            prev_n = curr_n
        self.layers.append(Dense(prev_n, 100))
        self.layers.append(Sigmoid())

    def feedforward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs


class Dense(Layer):

    weights: numpy.ndarray
    biases: numpy.ndarray
    last_input: numpy.ndarray

    def __init__(self, n_layers_in: int, n_layers_out: int):
        self.weights = np.random.randn(n_layers_out, n_layers_in)
        self.biases = np.zeros((n_layers_out, 1))

    def forward(self, inputs):
        self.last_input = inputs
        return self.weights @ inputs + self.biases


class Sigmoid(Layer):
    last_output: numpy.ndarray
    def forward(self, inputs):
        self.last_output = 1 / (1 + np.exp(-inputs))
        return self.last_output

