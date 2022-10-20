CFG = {
    "data": {
        "image_size_x": 28,
        "image_size_y": 28
    },
    "train": {
        "image_path": "assets/train-images.idx3-ubyte", 
        "label_path": "assets/train-labels.idx1-ubyte",
        "batch_size": 64,
        "epoch_size": 20,
        "nrof_classes": 10,
        "shuffle": True,
        "sample_type": "default"
    },
    "test": {
        "image_path": "assets/t10k-images.idx3-ubyte", 
        "label_path": "assets/t10k-labels.idx1-ubyte",
        "batch_size": 64,
        "epoch_size": 20,
        "nrof_classes": 10,
        "shuffle": True,
        "sample_type": "default"
    }
}
