from abc import abstractmethod


class BaseLayer(object):
    """
    Класс :class:`Layer` представляет собой один слой нейронной сети.
    Данный класс должен быть унаследован при реализации новых типов слоев.
    """
    def __init__(self):
        self.is_first = False
        self._params = None

    @property
    def grads(self):
        """
        Возвращает градиенты параметров слоя, рассчитанные с помощью backward()
        """
        return []

    @property
    def params(self):
        """
        Возвращает список обучаемых параметров.

        Если слой необучаемый, возвращается пустой список
        """
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        """
        Вычисление вывода текущего слоя при заданном вводе от предыдущего слоя

        :param x: вывод предыдущего слоя

        :return: вывод текущего слоя
        """
        pass

    @abstractmethod
    def backward(self, dy, *args, **kwargs):
        """
        Цепное правило дифференцирования.
        :param dy: значение градиента пришедшего от следующего слоя.
        :return: значение градиента этого слоя.
        """
        pass

    @abstractmethod
    def connect_to(self, previous_layer):
        """
        Присоединяет ввод данного слоя к выводу предыдущего

        (Все слои разные, поэтому отдельная функция для подключения слоев друг к другу)
        :param previous_layer: предыдущий слой
        """
        pass

    @property
    def params_with_grads(self):
        """
        Возвращает список параметров с соответствующими градиентами
        """
        return list(zip(self.params, self.grads))

    def __str__(self):
        return self.__class__.__name__
