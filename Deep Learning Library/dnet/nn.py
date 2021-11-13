"""
A neural net is just a collection of layers
It behaves a lot like a layer itself
"""

from typing import Iterator, Sequence, Tuple
import numpy as np
from dnet.tensor import Tensor
from dnet.layers import Layer

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, gradient: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad