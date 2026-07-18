import numpy
import numpy as np
from common_types import *

class NetworkDense(Network):
    layers: list[Layer]
    learning_rate: float
    def __init__(self, hidden: list[int], learning_rate: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = []
        self.learning_rate = learning_rate
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

    def backprop(self, grad):
        for i in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[i].backward(grad)
        return grad

    def sgd(self):
        for i in range(len(self.layers) - 1, -1, -1):
            curr = self.layers[i]
            if isinstance(curr, Dense):
                curr.weights -= curr.dC_dW * self.learning_rate
                curr.biases -= curr.dC_dB * self.learning_rate


class Dense(Layer):

    weights: np.ndarray
    biases: np.ndarray
    last_input: np.ndarray
    dC_dW: np.ndarray
    dC_dB: np.ndarray

    def __init__(self, n_layers_in: int, n_layers_out: int):
        self.weights = np.random.randn(n_layers_out, n_layers_in)
        self.biases = np.zeros((n_layers_out, 1))

    def forward(self, inputs):
        self.last_input = inputs
        return self.weights @ inputs + self.biases

    def backward(self, grad):
        self.dC_dW = grad @ self.last_input.T
        self.dC_dB = grad
        dC_dA = self.weights.T @ grad
        return dC_dA


class Sigmoid(Layer):
    last_output: np.ndarray
    def forward(self, inputs):
        self.last_output = 1 / (1 + np.exp(-inputs))
        return self.last_output

    def backward(self, grad):
        sigmoid_prime = self.last_output * (1 - self.last_output)
        return sigmoid_prime * grad


if __name__ == "__main__":
    X = NetworkDense(name="arc", network_type="dense", hidden=[64, 8])
    a = np.random.randn(460 * 215 * 3, 1)
    b = np.random.randn(100, 1)

