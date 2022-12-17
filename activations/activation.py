from abc import abstractmethod

import numpy as np

from utils import REGISTRY_ACTIVATIONS


class Activation(object):
    """
    Базовый класс для функции активации
    """
    def __init__(self):
        self.last_forward = None

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x=None):
        pass

    def __str__(self):
        return self.__class__.__name__


@REGISTRY_ACTIVATIONS.register_module
class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.last_forward = x
        return np.maximum(0.0, x)

    def backward(self, x=None):
        last_forward = x if x else self.last_forward
        res = np.zeros(last_forward.shape, "float32")
        res[last_forward > 0] = 1.0
        return res


@REGISTRY_ACTIVATIONS.register_module
class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x, *args, **kwargs):
        ones = np.ones_like(x)
        self.last_forward = ones / (ones + np.exp(-x))
        return self.last_forward

    def backward(self, x=None):
        last_forward = self.forward(x) if input else self.last_forward
        return np.multiply(last_forward, 1 - last_forward)


@REGISTRY_ACTIVATIONS.register_module
class SoftMax(Activation):
    def __init__(self):
        super(SoftMax, self).__init__()

    def forward(self, x):
        assert np.ndim(x) == 2
        self.last_forward = x
        _x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(_x)
        s = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return s

    def backward(self, x=None):
        last_forward = x if x else self.last_forward
        return np.ones(last_forward.shape, dtype="float32")
