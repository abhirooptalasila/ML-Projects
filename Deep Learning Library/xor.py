"""
The canonical example of a function that can't be 
learned with a simple linear model is XOR. 
"""

import numpy as np
from dnet.train import train
from dnet.nn import NeuralNet
from dnet.layers import Linear, Tanh

inputs = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])

targets = np.array([
    [1, 0], 
    [0, 1],
    [1, 0],
    [0, 1]
])

model = NeuralNet([
    Linear(2, 2), 
    Tanh(), 
    Linear(2, 2)
])

train(model, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = model.forward(x)
    print(x, y, predicted)
