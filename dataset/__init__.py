import idx2numpy
import numpy as np

from transform import Transform


class Dataset(object):
    def __init__(self, image_path: str, label_path: str, dataset_type: str, transforms: list[Transform]):
        """
        :param image_path (string): путь до файла с изображениями
        :param label_path (string): путь до файла с метками.
        :param dataset_type (string): (['train', 'valid', 'test']).
        :param transforms (list): список необходимых преобразований изображений.
        """
        self.image_path: str = image_path
        self.label_path: str = label_path
        self.dataset_type: str = dataset_type
        self.transforms: list = transforms

    def read_data(self):
        """
        Считывание данных по заданному пути.
        """
        self.images = idx2numpy.convert_from_file(self.image_path)
        self.labels = idx2numpy.convert_from_file(self.label_path)
        unique = np.unique(self.labels)
        self.nrof_classes = len(unique)

    def __len__(self) -> int:
        return len(self.images)

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
        images = self.images[idx]
        labels = self.labels[idx]
        for transform in self.transforms:
            images = transform(images)
        return images, labels

    def show_statistics(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"Количество элементов: {len(self.images)}")
        print(f"Тип датасета: {self.dataset_type}")
        print(f"Количество классов: {self.nrof_classes}")
        print(f"Количество данных в каждом классе:")
        for k, v in dict(zip(unique, counts)).items():
            print(f"\t{k}: {v}")
