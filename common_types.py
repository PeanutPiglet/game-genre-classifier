

class Network:
    name: str
    network_type: str
    def __init__(self, name: str, network_type: str, *args, **kwargs):
        self.name = name
        self.network_type = network_type

    def feedforward(self, inputs):
        raise NotImplementedError

    def backprop(self, grad):
        raise NotImplementedError


class Layer:
    def forward(self, inputs):
        raise NotImplementedError
    def backward(self, grad):
        raise NotImplementedError



