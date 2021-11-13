"""
A loss function measures how good out predictions are
We can use it adjust the parameters of our network
"""

import numpy as np
from dnet.tensor import Tensor

class Loss:
    def loss(self, target: Tensor, predicted: Tensor) -> float:
        raise NotImplementedError
    
    def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):
    """
    MSE is mean-squared error
    """
    def loss(self, target: Tensor, predicted: Tensor) -> float:
        return np.sum((predicted - target) ** 2)
    
    def gradient(self, target: Tensor, predicted: Tensor) -> Tensor:
        return 2 * (predicted - target)