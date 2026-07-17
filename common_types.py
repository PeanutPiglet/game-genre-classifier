

class Network:
    name: str
    network_type: str
    def __init__(self, name: str, network_type: str, *args, **kwargs):
        self.name = name
        self.network_type = network_type


class Layer:
    def forward(self, inputted):
        raise NotImplementedError
    def backward(self, grad):
        raise NotImplementedError



