import numpy as np

from objectives import Objective


class SoftmaxCrossEntropy(Objective):
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def forward(self, x, y):
        """
        вычисление значения целевой функции
        :param x: выход последнего полносвязного слоя/выход слоя softmax
        :param y: ground-truth метки
        :return: значение целевой функции
        """
        # x = np.clip(x, self.epsilon, 1 - self.epsilon)
        ex = np.exp(x - np.max(x))
        ex = ex / np.sum(ex, axis=1).reshape(-1, 1)
        return np.mean(-np.sum(y * np.log(ex), axis=1))

    def backward(self, x, y):
        """
        вычисление градиентов итоговых для передачи дальше
        :param x: выход последнего полносвязного слоя/выход слоя softmax
        :param y: ground-truth метки
        :return: значение градиента текущего слоя
        """
        x = np.clip(x, self.epsilon, 1 - self.epsilon)
        return x - y
