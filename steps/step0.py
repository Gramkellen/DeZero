from globaldefine import Variable, Function
import numpy as np


class Exp(Function):
    def forward(self, inputs):
        return np.exp(inputs)

    def backward(self, grad):
        x = self.inputs.data
        return np.exp(x) * grad


def exp(x):
    return Exp()(x)
