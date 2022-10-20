import math
from typing import Any, Generator
from matplotlib import pyplot as plt
import numpy as np
from dataset import Dataset


class DataLoader(object):
    def __init__(self, 
        dataset: Dataset, nrof_classes: int, dataset_type: str, 
        shuffle: bool, batch_size: int, sample_type: str,
        epoch_size: int | None = None
    ):
        """
        :param dataset (Dataset): объект класса Dataset.
        :param nrof_classes (int): количество классов в датасете.
        :param dataset_type (str): (['train', 'valid', 'test']).
        :param shuffle (bool): нужно ли перемешивать данные после очередной эпохи.
        :param batch_size (int): размер батча.
        :param sample_type (string): (['default' - берем последовательно все данные])
        :param epoch_size (int or None): размер эпохи. Если None, необходимо посчитать размер эпохи (=размеру обучающей выборки/batch_size)
        """
        self.dataset = dataset
        self.nrof_classes = nrof_classes
        self.dataset_type = dataset_type
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_type = sample_type
        self.epoch_size = epoch_size if epoch_size != None else len(dataset) / batch_size


    def batch_generator(self):
        """
        Создание батчей на эпоху с учетом указанного размера эпохи и типа сэмплирования.
        """
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            yield self.dataset[indices[i: i+self.batch_size]]

    def show_batch(self, batch_img):
        img, label = batch_img

        pic_box = plt.figure(figsize=(10,6))

        for i, picture in enumerate(img):
            pic_box.add_subplot(int(math.sqrt(len(img)))+1, int(math.sqrt(len(img))),i+1)
            plt.imshow(picture)
            plt.title(label[i])
            plt.axis('off')

        plt.subplots_adjust(wspace=0.2, hspace=0.5)
        plt.show()
