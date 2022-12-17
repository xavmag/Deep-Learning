import numpy as np

import activations
from activations import ReLU, Activation
from initialize import Normal, Initializer, init_zero
from utils import REGISTRY_TYPE
from .base_layer import BaseLayer


@REGISTRY_TYPE.register_module
class FullyConnected(BaseLayer):
    """
    Полносвязный слой
    """

    def __init__(self, n_out, n_in=None, initialize: Initializer = Normal(), activation: Activation = ReLU()):
        """

        :param n_out: Форма выходящего слоя
        :param n_in: Форма входящего слоя
        :param initialize: Инициализатор
        :param activation: Функция активации
        """
        super().__init__()
        self.n_out = n_out
        self.n_in = n_in
        self.out_shape = (None, n_out)

        self.initialize = initialize
        self.activation = activation

        self.W, self.dW = None, None
        self.b, self.db = None, None
        self.last_input = None

    def connect_to(self, previous_layer=None):
        if previous_layer is None:
            assert self.n_in is not None
            n_in = self.n_in
        else:
            assert len(previous_layer.out_shape) == 2
            n_in = previous_layer.out_shape[-1]
        self.W = self.initialize((n_in, self.n_out))
        self.b = init_zero((self.n_out,))

    def forward(self, x, *args, **kwargs):
        self.last_input = x
        linear_out = np.dot(x, self.W) + self.b
        return self.activation.forward(linear_out)

    def backward(self, dy, *args, **kwargs):
        act_grad = dy * self.activation.backward()
        self.dW = np.dot(self.last_input.T, act_grad) / len(dy)
        self.db = np.mean(act_grad, axis=0)
        # self.update_weights()
        if not self.is_first:
            return np.dot(act_grad, self.W.T)

    def update_weights(self):
        self.W -= self.dW * 0.01
        self.b -= self.b * 0.01

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db

    @params.setter
    def params(self, p):
        self.W, self.b = p


@REGISTRY_TYPE.register_module
class SoftMax(FullyConnected):
    def __init__(self, n_out, n_in=None, init=Normal()):
        super(SoftMax, self).__init__(n_out, n_in, init, activation=activations.SoftMax())


@REGISTRY_TYPE.register_module
class Flatten(BaseLayer):
    """
    Вытягивание входящего слоя в вектор
    """
    def __init__(self, outdim=2):
        super().__init__()
        self.outdim = outdim
        assert outdim > 0, "Размер выходящего слоя не может быть меньше единицы"

        self.last_input_shape = None
        self.out_shape = None

    def connect_to(self, previous_layer):
        assert len(previous_layer.out_shape) > 2

        to_flatten = np.prod(previous_layer.out_shape[self.outdim-1:])
        self.out_shape = previous_layer.out_shape[:self.outdim - 1] + (to_flatten,)

    def forward(self, x, *args, **kwargs):
        self.last_input_shape = x.shape

        flattened_shape = x.shape[:self.outdim - 1] + (-1,)
        return np.reshape(x, flattened_shape)

    def backward(self, dy, *args, **kwargs):
        return np.reshape(dy, self.last_input_shape)