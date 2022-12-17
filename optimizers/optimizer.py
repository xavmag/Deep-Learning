from abc import abstractmethod


class Optimizer(object):
    """
    Базовый класс для методов решения
    задачи математического программирования
    """
    @abstractmethod
    def __init__(self, lr=0.01, clip=-1):
        """
        :param clip: Если меньше нуля, не обрезать значение параметра
        :param lr: Коэффициент скорости обучения нейросети
        """
        self.lr = lr
        self.clip = clip

    @abstractmethod
    def minimize(self, params, grads):
        """
        :param params:
        :param grads:
        """

    def __str__(self):
        return self.__class__.__name__

