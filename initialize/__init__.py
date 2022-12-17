from abc import abstractmethod

import numpy as np


class Initializer(object):
    """
    Базовый класс для инициализаторов обучаемых параметров
    """
    def __call__(self, size):
        return self._call(size)

    @abstractmethod
    def _call(self, size):
        pass

    def __str__(self):
        return self.__class__.__name__


class Zero(Initializer):
    """Initialize weights with zero value.
    """
    def _call(self, size):
        return np.array(np.zeros(size), dtype="float32")


class One(Initializer):
    """Initialize weights with one value.
    """
    def _call(self, size):
        return np.array(np.ones(size), dtype="float32")


init_zero = Zero()
init_one = One()


class Uniform(Initializer):
    """Sample initial weights from the uniform distribution.

    Parameters are sampled from U(a, b).

    Parameters
    ----------
    scale : float or tuple.
        When std is None then range determines a, b. If range is a float the
        weights are sampled from U(-range, range). If range is a tuple the
        weights are sampled from U(range[0], range[1]).
    """
    def __init__(self, scale=0.05):
        self.scale = scale

    def _call(self, size):
        out = np.random.default_rng().uniform(-self.scale, self.scale, size=size)
        return np.array(out, dtype="float32")


class Normal(Initializer):

    """
    Выборка начальных весов из распределения Гаусса.

    :param std: Стандартное отклонение
    :param mean: Среднее
    """
    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean

    def _call(self, size):
        out = np.random.default_rng().normal(loc=self.mean, scale=self.std, size=size)
        return np.array(out, dtype="float32")
