# -*- coding: utf-8 -*-
""" main.py """
import idx2numpy
import numpy as np

from configs import CFG
from dataprovider import DataProvider
from dataset import Dataset
from model import Model
from transform import Normalize
from utils import REGISTRY_TYPE, REGISTRY_ACTIVATIONS


def create_layers(config: dict):
    layers = []
    for layer in config:
        layer[1]["activation"] = REGISTRY_ACTIVATIONS.get(*layer[1]["activation"])
        layers.append(REGISTRY_TYPE.get(*layer))
    return layers


def to_one_hot(labels, nb_classes=None):
    classes = np.unique(labels)
    if nb_classes is None:
        nb_classes = classes.size
    one_hot_labels = np.zeros((labels.shape[0], nb_classes))
    for i, c in enumerate(classes):
        one_hot_labels[labels == c, i] = 1
    return one_hot_labels


def run():

    train_images = idx2numpy.convert_from_file(CFG["train"]["image_path"])
    train_labels = idx2numpy.convert_from_file(CFG["train"]["label_path"])

    test_images = idx2numpy.convert_from_file(CFG["test"]["image_path"])
    test_labels = idx2numpy.convert_from_file(CFG["test"]["label_path"])

    vs = CFG["model"]["validation_split"]
    if 1.0 > vs > 0.0:
        split = int(train_labels.shape[0] * vs)
        valid_images, valid_labels = train_images[-split:], train_labels[-split:]
        train_images, train_labels = train_images[:-split], train_labels[:-split]
    else:
        valid_images, valid_labels = None, None

    train_dataset: Dataset = Dataset(train_images, to_one_hot(train_labels), "train", [Normalize()])
    valid_dataset: Dataset = Dataset(valid_images, to_one_hot(valid_labels), "valid", [Normalize()])
    test_dataset: Dataset = Dataset(test_images, to_one_hot(test_labels), "test", [Normalize()])

    train_dp = DataProvider(train_dataset, CFG["train"]["nrof_classes"], "train", CFG["train"]["shuffle"],
                            CFG["train"]["batch_size"])
    test_dp = DataProvider(test_dataset, CFG["test"]["nrof_classes"], "test", CFG["test"]["shuffle"],
                           CFG["test"]["batch_size"])
    valid_dp = DataProvider(valid_dataset, CFG["valid"]["nrof_classes"], "valid", CFG["valid"]["shuffle"],
                            CFG["valid"]["batch_size"])

    model = Model(create_layers(CFG["model"]["layers"]))
    # model.fit(train_dp, 20, valid_dp)
    # model.dump("dumps/001.pickle")
    model.load("dumps/001.pickle")
    model.test(test_dp)
    # model.overfit_on_batch(train_dp)


if __name__ == "__main__":
    run()
