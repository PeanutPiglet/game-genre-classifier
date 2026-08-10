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
IMAGE_C = 3
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
        input_features = IMAGE_W * IMAGE_H * IMAGE_C
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
        from dataloader import load_image

        data = load_image(image_path)
        output = self.feedforward(data)
        res = [f"{i:02d} : {output[i, 0]:.6f}" for i in range(output.shape[0])]
        print("\n".join(res))


class NetworkPyTorchConv(NetworkPyTorch):
    def __init__(self, hidden: list[int] = None, conv_channels: list[int] = None,
                 learning_rate: float = 0.01, momentum_factor: float = 0.9,
                 source: str = "", device: str | None = None, *args, **kwargs):
        self.conv_channels = conv_channels if conv_channels is not None else [16, 32, 64]
        super().__init__(hidden=hidden, learning_rate=learning_rate, momentum_factor=momentum_factor,
                         source=source, device=device, *args, **kwargs)

    def _build_model(self, hidden: list[int]):
        if not hidden:
            hidden = [512]

        layers = []
        in_channels = IMAGE_C
        for out_channels in self.conv_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        pool_factor = 2 ** len(self.conv_channels)
        flattened_size = in_channels * (IMAGE_W // pool_factor) * (IMAGE_H // pool_factor)
        layers.append(nn.Flatten())

        for layer_size in hidden:
            layers.append(nn.Linear(flattened_size, layer_size))
            layers.append(nn.ReLU())
            flattened_size = layer_size

        layers.append(nn.Linear(flattened_size, OUTPUT_N))
        self.model = nn.Sequential(*layers).to(self.device)

    def feedforward(self, inputs):
        np_inputs = np.asarray(inputs, dtype=np.float32)

        if np_inputs.ndim == 2 and np_inputs.shape[1] == 1 and np_inputs.shape[0] == IMAGE_W * IMAGE_H * IMAGE_C:
            np_inputs = np_inputs.reshape(IMAGE_H, IMAGE_W, IMAGE_C)
        elif np_inputs.ndim == 1 and np_inputs.size == IMAGE_W * IMAGE_H * IMAGE_C:
            np_inputs = np_inputs.reshape(IMAGE_H, IMAGE_W, IMAGE_C)

        if np_inputs.ndim == 3:
            if np_inputs.shape == (IMAGE_H, IMAGE_W, IMAGE_C):
                np_inputs = np_inputs.transpose(2, 0, 1)
            elif np_inputs.shape == (IMAGE_C, IMAGE_H, IMAGE_W):
                pass
            elif np_inputs.shape == (IMAGE_H, IMAGE_W, IMAGE_C):
                np_inputs = np_inputs.transpose(2, 0, 1)
            else:
                raise ValueError("Expected image shape (C,H,W) or (H,W,C)")
            np_inputs = np_inputs[None, ...]
        elif np_inputs.ndim == 4:
            if np_inputs.shape[1] != IMAGE_C and np_inputs.shape[-1] == IMAGE_C:
                np_inputs = np_inputs.transpose(0, 3, 1, 2)
        else:
            raise ValueError("Expected 2D flattened input, 3D image tensor, or 4D image batch")

        tensor = torch.from_numpy(np_inputs).to(self.device)
        tensor.requires_grad_(True)
        self.last_input = tensor

        logits = self.model(tensor)
        self.last_output = torch.sigmoid(logits)
        return self.last_output.detach().cpu().numpy().T

    def evaluate(self, image_path):
        from dataloader import load_image_2d

        data = load_image_2d(image_path)
        output = self.feedforward(data)
        res = [f"{i:02d} : {output[i, 0]:.6f}" for i in range(output.shape[0])]
        print("\n".join(res))


if __name__ == "__main__":
    network = NetworkPyTorch(name="pt_net", network_type="pytorch", hidden=[1024, 512])
    print(network)
