import numpy as np

from transform import Transform


class Dataset(object):
    def __init__(self, data: np.ndarray, labels: np.ndarray, type: str, transforms: list[Transform]):
        """
        :param data (np.ndarray): Данные
        :param labels (np.ndarray): Метки классов
        :param type (string): (['train', 'valid', 'test']).
        :param transforms (list): Список необходимых преобразований изображений.
        """
        self.transforms: list = transforms

        self.data = data
        self.labels = labels
        self.type = type
        unique = np.unique(self.labels)
        self.nrof_classes = len(unique)

    def __len__(self) -> int:
        return len(self.data)

    def one_hot_labels(self, label: int) -> np.ndarray[np.int32, np.dtype[np.int32]]:
        """
        для 10 классов метка 5-> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        :param label: метка класса
        :return: one-hot encoding вектор
        """
        out = np.zeros(self.nrof_classes, dtype=np.int32)
        out[label] = 1
        return out

    def __getitem__(self, idx: int):
        """
        :param idx: индекс элемента в выборке
        :return: preprocessed image and label
        """
        data = self.data[idx]
        labels = self.labels[idx]
        for transform in self.transforms:
            data = transform(data)
        return data, labels

    def __str__(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        out = f"""Количество элементов: {len(self.data)}
        Тип датасета: {self.type}
        Количество классов: {self.nrof_classes}
        Количество данных в каждом классе:\n"""
        for k, v in dict(zip(unique, counts)).items():
            out += f"\t{k}: {v}\n"
