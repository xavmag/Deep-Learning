from abc import abstractmethod


class Objective(object):
    """
    Базовый класс для целевой функции
    """
    @abstractmethod
    def forward(self, x, y):
        pass

    @abstractmethod
    def backward(self, x, y):
        pass

    def __str__(self):
        return self.__class__.__name__
