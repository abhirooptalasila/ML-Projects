"""
Here's a function that can train a neural net
"""

from dnet.tensor import Tensor
from dnet.nn import NeuralNet
from dnet.loss import Loss, MSE
from dnet.optim import Optimizer, SGD
from dnet.data import BatchIterator, DataIterator


def train(
    model: NeuralNet, inputs: Tensor, targets: Tensor, 
    num_epochs: int = 5000, iterator: DataIterator = BatchIterator(), 
    loss: Loss = MSE(), optimizer: Optimizer = SGD()) -> None:
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = model.forward(batch.inputs)
            epoch_loss += loss.loss(batch.targets, predicted)
            grad = loss.gradient(batch.targets, predicted)
            model.backward(grad)
            optimizer.step(model)
        if epoch % 10 == 0:
            print(epoch, epoch_loss)
