# -*- coding: utf-8 -*-
""" main.py """

from configs import CFG
from dataloader import DataLoader
from dataset import Dataset
from transform import Normalize, View


def run():
    train_dataset: Dataset = Dataset(
        CFG["test"]["image_path"], 
        CFG["test"]["label_path"], 
        "train", 
        [Normalize(), View()]
    )
    train_dataset.read_data()
    train_dataset.show_statistics()

    test_dataset: Dataset = Dataset(
        CFG["test"]["image_path"], 
        CFG["test"]["label_path"], 
        "test", 
        [Normalize(), View()]
    )
    test_dataset.read_data()
    test_dataset.show_statistics()

    train_dataloader = DataLoader(train_dataset, 
        CFG["train"]["nrof_classes"],
        "train",
        CFG["train"]["shuffle"],
        CFG["train"]["batch_size"],
        CFG["train"]["sample_type"],
        CFG["train"]["epoch_size"])
    test_dataloader = DataLoader(test_dataset,
        CFG["test"]["nrof_classes"],
        "test",
        CFG["test"]["shuffle"],
        CFG["test"]["batch_size"],
        CFG["test"]["sample_type"],
        CFG["test"]["epoch_size"])
    
    
    train_dataloader.show_batch(next(train_dataloader.batch_generator()))


if __name__ == "__main__":
    run()