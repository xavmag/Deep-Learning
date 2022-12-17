import numpy as np

from initialize import init_zero
from layers import BaseLayer
from utils import REGISTRY_TYPE


@REGISTRY_TYPE.register_module
class MaxPooling(BaseLayer):
    """
    :param pool_size: множитель понижения размерности входящего тенора
    :return: numpy.array `(nb_batch, depth, pooled_h, pooled_w)`
    """
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

        self.input_shape = None
        self.out_shape = None
        self.last_input = None

    def connect_to(self, prev_layer):
        # prev_layer.out_shape: (nb_batch, depth, height, width)
        assert len(prev_layer.out_shape) >= 3

        old_h, old_w = prev_layer.out_shape[-2:]
        pool_h, pool_w = self.pool_size
        new_h, new_w = old_h // pool_h, old_w // pool_w

        assert old_h % pool_h == old_w % pool_w == 0

        self.out_shape = prev_layer.out_shape[:-2] + (new_h, new_w)

    def forward(self, input, *args, **kwargs):
        # shape
        self.input_shape = input.shape
        pool_h, pool_w = self.pool_size
        new_h, new_w = self.out_shape[-2:]

        # forward
        self.last_input = input
        outputs = init_zero(self.input_shape[:-2] + self.out_shape[-2:])

        if np.ndim(input) == 4:
            nb_batch, nb_axis, _, _ = input.shape

            for a in np.arange(nb_batch):
                for b in np.arange(nb_axis):
                    for h in np.arange(new_h):
                        for w in np.arange(new_w):
                            outputs[a, b, h, w] = np.max(input[a, b, h:h + pool_h, w:w + pool_w])

        elif np.ndim(input) == 3:
            nb_batch, _, _ = input.shape

            for a in np.arange(nb_batch):
                for h in np.arange(new_h):
                    for w in np.arange(new_w):
                        outputs[a, h, w] = np.max(input[a, h:h + pool_h, w:w + pool_w])

        else:
            raise ValueError()

        return outputs

    def backward(self, pre_grad, *args, **kwargs):
        new_h, new_w = self.out_shape[-2:]
        pool_h, pool_w = self.pool_size

        layer_grads = init_zero(self.input_shape)

        if np.ndim(pre_grad) == 4:
            nb_batch, nb_axis, _, _ = pre_grad.shape

            for a in np.arange(nb_batch):
                for b in np.arange(nb_axis):
                    for h in np.arange(new_h):
                        for w in np.arange(new_w):
                            patch = self.last_input[a, b, h:h + pool_h, w:w + pool_w]
                            max_idx = np.unravel_index(np.nanargmax(patch), patch.shape)
                            h_shift, w_shift = h * pool_h + max_idx[0], w * pool_w + max_idx[1]
                            layer_grads[a, b, h_shift, w_shift] = pre_grad[a, b, h, w]

        elif np.ndim(pre_grad) == 3:
            nb_batch, _, _ = pre_grad.shape

            for a in np.arange(nb_batch):
                for h in np.arange(new_h):
                    for w in np.arange(new_w):
                        patch = self.last_input[a, h:h + pool_h, w:w + pool_w]
                        max_idx = np.unravel_index(patch.argmax(), patch.shape)
                        h_shift, w_shift = h * pool_h + max_idx[0], w * pool_w + max_idx[1]
                        layer_grads[a, h_shift, w_shift] = pre_grad[a, h, w]

        else:
            raise ValueError()

        return layer_grads
