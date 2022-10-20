
from abc import ABC, abstractmethod
import numpy as np

from configs import CFG

class Transform(ABC):
    """
    Базовый класс для трансформаций
    """
    @abstractmethod
    def __call__(self, images):
        pass


class Normalize(Transform):
    def __init__(self, mean=128, var=255) -> None:
        """
        :param mean (int or tuple): среднее значение (пикселя), которое необходимо вычесть.
        :param var (int): значение, на которое необходимо поделить.
        """
        self.mean = mean
        self.var = var

    def __call__(self, images):
        return (images - self.mean)/self.var


class View(Transform):
    """
    Преобразование изображений в векторы
    """
    def __init__(self) -> None:
        self.image_size_x = CFG["data"]["image_size_x"]
        self.image_size_y = CFG["data"]["image_size_y"]

    def __call__(self, images):
        for image in images:            
            image = np.array(image, dtype='uint8')
            image = image.reshape(self.image_size_x, self.image_size_y)
        return images