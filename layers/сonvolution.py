import numpy as np

from activations import ReLU
from initialize import Uniform, init_zero
from layers import BaseLayer
from utils import REGISTRY_TYPE


@REGISTRY_TYPE.register_module
class Convolution(BaseLayer):
    """Сверточный слой.

    При использовании данного слоя в модели в качестве первого
    необходимо передать в `input_shape` кортеж из None (размер входящей выборки)
    и 3-х int (глубина изображения, высота, ширина)
    """

    def __init__(self, nb_filter, filter_size, input_shape=None, stride=1, init=Uniform(), activation=ReLU()):
        super().__init__()
        self.nb_filter = nb_filter
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.stride = stride

        self.W, self.dW = None, None
        self.b, self.db = None, None
        self.out_shape = None
        self.last_output = None
        self.last_input = None

        self.init = init
        self.activation = activation

    def connect_to(self, prev_layer=None):
        if prev_layer is None:
            assert self.input_shape is not None
            input_shape = self.input_shape
        else:
            input_shape = prev_layer.out_shape

        # input_shape: (batch_size, depth, image_h, image_w)
        assert len(input_shape) == 4

        nb_batch, pre_nb_filter, pre_height, pre_width = input_shape
        filter_height, filter_width = self.filter_size

        height = (pre_height - filter_height) // self.stride + 1
        width = (pre_width - filter_width) // self.stride + 1

        # Форма выхода
        self.out_shape = (nb_batch, self.nb_filter, height, width)

        # Фильтры
        self.W = self.init((self.nb_filter, pre_nb_filter, filter_height, filter_width))
        self.b = init_zero((self.nb_filter,))

    def forward(self, input, *args, **kwargs):

        self.last_input = input

        nb_batch, input_depth, old_img_h, old_img_w = input.shape
        filter_h, filter_w = self.filter_size
        new_img_h, new_img_w = self.out_shape[2:]

        outputs = init_zero((nb_batch, self.nb_filter, new_img_h, new_img_w))

        # Свертка
        for x in np.arange(nb_batch):
            for y in np.arange(self.nb_filter):
                for h in np.arange(new_img_h):
                    for w in np.arange(new_img_w):
                        h_shift, w_shift = h * self.stride, w * self.stride
                        # patch: (input_depth, filter_h, filter_w)
                        patch = input[x, :, h_shift: h_shift + filter_h, w_shift: w_shift + filter_w]
                        outputs[x, y, h, w] = np.sum(patch * self.W[y]) + self.b[y]

        # self.last_output: (nb_batch, output_depth, image_h, image_w)
        self.last_output = self.activation.forward(outputs)

        return self.last_output

    def backward(self, pre_grad, *args, **kwargs):

        assert pre_grad.shape == self.last_output.shape
        nb_batch, input_depth, old_img_h, old_img_w = self.last_input.shape
        new_img_h, new_img_w = self.out_shape[2:]
        filter_h, filter_w = self.filter_size
        old_img_h, old_img_w = self.last_input.shape[-2:]

        self.dW = init_zero(self.W.shape)
        self.db = init_zero(self.b.shape)
        delta = pre_grad * self.activation.backward()

        # dW
        for r in np.arange(self.nb_filter):
            for t in np.arange(input_depth):
                for h in np.arange(filter_h):
                    for w in np.arange(filter_w):
                        input_window = self.last_input[:, t,
                                       h:old_img_h - filter_h + h + 1:self.stride,
                                       w:old_img_w - filter_w + w + 1:self.stride]
                        delta_window = delta[:, r]
                        self.dW[r, t, h, w] = np.sum(input_window * delta_window) / nb_batch

        # db
        for r in np.arange(self.nb_filter):
            self.db[r] = np.sum(delta[:, r]) / nb_batch

        # dX
        if not self.is_first:
            layer_grads = init_zero(self.last_input.shape)
            for b in np.arange(nb_batch):
                for r in np.arange(self.nb_filter):
                    for t in np.arange(input_depth):
                        for h in np.arange(new_img_h):
                            for w in np.arange(new_img_w):
                                h_shift, w_shift = h * self.stride, w * self.stride
                                layer_grads[b, t, h_shift:h_shift + filter_h, w_shift:w_shift + filter_w] += \
                                    self.W[r, t] * delta[b, r, h, w]
            return layer_grads

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db
