import numpy as np

from optimizers import Optimizer


class SGD(Optimizer):
    """
    Стохастический градиентный спуск
    """
    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)

    def minimize(self, params, grads):
        # Параметры и градиенты передаются по ссылке, все ок
        for p, g in zip(params, grads):
            c = g
            if self.clip > 0:
                c = np.clip(g, -self.clip, self.clip)
            p -= self.lr * c
