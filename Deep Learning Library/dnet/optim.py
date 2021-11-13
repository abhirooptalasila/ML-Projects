"""
We use an optimizer to adjust the parameters 
of our network based on the gradients computed 
during backpropagation
"""

from dnet.nn import NeuralNet

class Optimizer:
    def step(self, model: NeuralNet) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
    
    def step(self, model: NeuralNet) -> None:
        for param, grad in model.params_and_grads():
            param -= self.lr * grad