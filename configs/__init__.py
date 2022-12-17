CFG = {
    "data": {
        "image_size_x": 28,
        "image_size_y": 28,
        "image_depth": 1
    },
    "model": {
        "validation_split": 0.1,
        # "layers": [
        #     ("Convolution", {
        #         "nb_filter": 1,
        #         "filter_size": (3, 3),
        #         "input_shape": (None, 1, 28, 28)
        #     }),
        #     ("MaxPooling", {"pool_size": (2, 2)}),
        #     ("Convolution", {
        #         "nb_filter": 2,
        #         "filter_size": (4, 4),
        #     }),
        #     ("MaxPooling", {"pool_size": (2, 2)}),
        #     ("Flatten", {"outdim": 2}),
        #     ("SoftMax", {"n_out": 10})
        # ]
        "layers": [
            ("FullyConnected", {"n_in": 784, "n_out": 128, "activation": ("ReLU", {})}),
            ("FullyConnected", {"n_out": 10, "activation": ("SoftMax", {})}),
        ]
    },
    "train": {
        "image_path": "assets/train-images.idx3-ubyte", 
        "label_path": "assets/train-labels.idx1-ubyte",
        "batch_size": 64,
        "epoch_size": 20,
        "nrof_classes": 10,
        "shuffle": True,
        "sample_type": "default",
    },
    "test": {
        "image_path": "assets/t10k-images.idx3-ubyte",
        "label_path": "assets/t10k-labels.idx1-ubyte",
        "batch_size": 64,
        "epoch_size": 20,
        "nrof_classes": 10,
        "shuffle": True,
        "sample_type": "default"
    },
    "valid": {
        "image_path": "assets/train-images.idx3-ubyte",
        "label_path": "assets/train-labels.idx1-ubyte",
        "batch_size": 64,
        "epoch_size": 20,
        "nrof_classes": 10,
        "shuffle": True,
        "sample_type": "default",
    },
}
