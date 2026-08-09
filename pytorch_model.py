import numpy as np

from common_types import Network

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for pytorch_model.py. Install it with `pip install torch` or `pip install torch torchvision`.") from exc

IMAGE_W = 128
IMAGE_H = 64
OUTPUT_N = 100


class NetworkPyTorch(Network):
    def __init__(self, hidden: list[int] = None, learning_rate: float = 0.01, momentum_factor: float = 0.9,
                 source: str = "", device: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.momentum_factor = momentum_factor
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.last_input = None
        self.last_output = None

        self.hidden = hidden if hidden is not None else [1024, 512]
        self._build_model(self.hidden)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum_factor)

        if source:
            self.load(source)

    def _build_model(self, hidden: list[int]):
        if not hidden:
            hidden = [1024, 512]

        layers = []
        input_features = IMAGE_W * IMAGE_H * 3
        for layer_size in hidden:
            layers.append(nn.Linear(input_features, layer_size))
            layers.append(nn.ReLU())
            input_features = layer_size

        layers.append(nn.Linear(input_features, OUTPUT_N))
        self.model = nn.Sequential(*layers).to(self.device)

    def __str__(self):
        layer_sizes = [p.shape for p in self.model.parameters() if p.dim() == 2]
        return f"{self.name} : {self.network_type} | {layer_sizes} | {self.device}"

    def feedforward(self, inputs):
        np_inputs = np.asarray(inputs, dtype=np.float32)
        if np_inputs.ndim == 1:
            np_inputs = np_inputs.reshape(-1, 1)
        if np_inputs.ndim != 2:
            raise ValueError("Expected 2D input array")

        tensor = torch.from_numpy(np_inputs.T).to(self.device)
        tensor.requires_grad_(True)
        self.last_input = tensor

        logits = self.model(tensor)
        self.last_output = torch.sigmoid(logits)
        return self.last_output.detach().cpu().numpy().T

    def backprop(self, grad):
        if self.last_output is None:
            raise RuntimeError("No forward pass available for backprop")

        grad_tensor = torch.from_numpy(np.asarray(grad, dtype=np.float32).T).to(self.device)
        self.last_output.backward(grad_tensor)
        return None

    def sgd(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def gd_momentum(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def dump(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str):
        state = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def evaluate(self, image_path):
        from PIL import Image

        with Image.open(image_path) as image:
            data = np.asarray(image, dtype=np.float32)
            data = data.reshape(-1, 1) / 255.0
            output = self.feedforward(data)
            res = [f"{i:02d} : {output[i, 0]:.6f}" for i in range(output.shape[0])]
            print("\n".join(res))


if __name__ == "__main__":
    network = NetworkPyTorch(name="pt_net", network_type="pytorch", hidden=[1024, 512])
    print(network)
