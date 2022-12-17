
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


class NormalizeConv(Transform):
    def __init__(self, mean=128.0, var=255.0) -> None:
        """
        :param mean (int or tuple): среднее значение (пикселя), которое необходимо вычесть.
        :param var (int): значение, на которое необходимо поделить.
        """
        self.mean = mean
        self.var = var
        self.image_depth = CFG["data"]["image_depth"]
        self.image_size_x = CFG["data"]["image_size_x"]
        self.image_size_y = CFG["data"]["image_size_y"]

    def __call__(self, images):
        img = images.reshape(-1, self.image_depth, self.image_size_x, self.image_size_y)
        return (img - self.mean)/self.var


class Normalize(Transform):
    def __init__(self, mean=128.0, var=255.0) -> None:
        """
        :param mean (int or tuple): среднее значение (пикселя), которое необходимо вычесть.
        :param var (int): значение, на которое необходимо поделить.
        """
        self.mean = mean
        self.var = var
        self.image_depth = CFG["data"]["image_depth"]
        self.image_size_x = CFG["data"]["image_size_x"]
        self.image_size_y = CFG["data"]["image_size_y"]

    def __call__(self, images):
        flattened_shape = images.shape[:1] + (-1,)
        return (np.reshape(images, flattened_shape) - self.mean) / self.var


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
            image.reshape(self.image_size_x, self.image_size_y)
        return images
