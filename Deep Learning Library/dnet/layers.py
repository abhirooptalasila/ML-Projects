"""
Our neural net will be made up of layers. 
Each layer needs to pass its  inputs forward 
and propagate gradients backward

For example: input -> Linear -> Tanh -> Linear -> output
"""

from typing import Callable, Dict
import numpy as np
from dnet.tensor import Tensor

F = Callable[[Tensor], Tensor]

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, gradient: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError

class Linear(Layer):
    """
    Computes inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params['w'] = np.random.randn(input_size, output_size)
        self.params['b'] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params['w'] + self.params['b']
    
    def backward(self, gradient: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        dy/da = f'(x) * b
        dy/db = f'(x) * a
        dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        dy/da = f'(x) @ b.T
        dy/db = a.T @ f'(x)
        dy/dc = f'(x)
        """

        self.grads['b'] = np.sum(gradient, axis=0)
        self.grads['w'] = self.inputs.T @ gradient
        return gradient @ self.params['w'].T


class Activation(Layer):
    """
    An activation layer just applies a function 
    elementwise to its inputs
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, gradient: Tensor) -> Tensor:
        """Chain Rule
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * gradient


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)
    
