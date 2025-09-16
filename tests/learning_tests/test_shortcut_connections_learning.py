import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple


# pytest -sv tests/learning_tests/test_shortcut_connections_learning.py
class TestShortcutConnectionsLearning:

    class ExampleDeepNeuralNetwork(nn.Module):

        _use_shortcut: bool
        _layers: nn.ModuleList
        _grad_means = ()

        def __init__(
            self, layer_sizes: Tuple[int, int, int, int, int, int], use_shortcut: bool
        ) -> None:
            super().__init__()
            self._use_shortcut = use_shortcut
            self._layers = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), nn.GELU()),
                    nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), nn.GELU()),
                    nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), nn.GELU()),
                    nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), nn.GELU()),
                    nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), nn.GELU()),
                ]
            )

        def forward(self, x):
            for layer in self._layers:
                layer_output = layer(x)
                if self._use_shortcut and x.shape == layer_output.shape:
                    x = layer_output + x
                else:
                    x = layer_output
            return x

        def print_gradients(self, x: Tensor) -> None:
            output = self(x)
            target = torch.tensor([[0.0]])
            loss = nn.MSELoss()
            loss = loss(output, target)
            loss.backward()
            print()
            for name, param in self.named_parameters():
                if "weight" in name:
                    assert type(param.grad) is Tensor
                    grad_mean = param.grad.abs().mean().item()
                    print(f"{name} has gradient mean of {grad_mean}")
                    self._grad_means += (grad_mean,)

        @property
        def grad_means(self):
            return self._grad_means

    _seed_num = 123

    def test_vanishing_gradient(self):
        layer_sizes = 3, 3, 3, 3, 3, 1
        sample_input = torch.tensor([[1.0, 0.0, -1.0]])
        torch.manual_seed(self._seed_num)
        model_without_shortcut = self.ExampleDeepNeuralNetwork(
            layer_sizes, use_shortcut=False
        )
        model_without_shortcut.print_gradients(sample_input)
        torch.manual_seed(self._seed_num)
        model_with_shortcut = self.ExampleDeepNeuralNetwork(
            layer_sizes, use_shortcut=True
        )
        model_with_shortcut.print_gradients(sample_input)
        assert all(
            grad_mean_without_shortcut < grad_mean_with_shortcut
            for grad_mean_without_shortcut, grad_mean_with_shortcut in zip(
                model_without_shortcut.grad_means, model_with_shortcut.grad_means
            )
        )
