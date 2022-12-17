import pickle

import numpy as np
from tensorboardX import SummaryWriter

from dataprovider import DataProvider
from layers import BaseLayer
from objectives import Objective, SoftmaxCrossEntropy
from optimizers import Optimizer, SGD


class Model(object):
    def __init__(self, layers: list[BaseLayer] | None, loss: Objective = SoftmaxCrossEntropy(), optimizer: Optimizer = SGD()):
        self.layers = [] if layers is None else layers
        self.loss = loss
        self.optimizer = optimizer

        if layers is not None:
            self.layers[0].is_first = True
            self._train_losses, self._train_predictions, self._train_targets = [], [], []

            next_layer = None
            for layer in self.layers:
                layer.connect_to(next_layer)
                next_layer = layer

        self.writer = SummaryWriter("log")
        self.iteration = 0

    def fit(self, train_data: DataProvider, num_epochs=10, validation_data: DataProvider | None = None):
        for _ in range(num_epochs):
            self._train_epoch(train_data)
            if validation_data:
                self._validate(validation_data)

    def dump(self, path: str):
        params = []
        for layer in self.layers:
            params.append(layer.params)
        file = open(path, 'wb')
        pickle.dump(params, file)
        file.close()

    def load(self, path: str):
        file = open(path, 'rb')
        data = pickle.load(file)
        file.close()
        for i, layer in enumerate(self.layers):
            layer.params = data[i]

    def _train_epoch(self, train_data):
        for batch, labels in train_data.batch_generator():
            self.train_step(batch, labels, self._train_losses, self._train_predictions, self._train_targets)
        total_loss = float(np.mean(self._train_losses))
        pr = np.argmax(self._train_predictions, axis=1)
        tg = np.argmax(self._train_targets, axis=1)
        total_accuracy = float(np.mean(pr == tg))
        self.writer.add_scalar("train/total_loss", total_loss, self.iteration)
        self.writer.add_scalar("train/total_accuracy", total_accuracy, self.iteration)

    def overfit_on_batch(self, train_data):
        batch, labels = next(train_data.batch_generator())
        for i in range(1000):
            self.train_step(batch, labels, self._train_losses, self._train_predictions, self._train_targets, True)
        total_loss = float(np.mean(self._train_losses))
        pr = np.argmax(self._train_predictions, axis=1)
        tg = np.argmax(self._train_targets, axis=1)
        total_accuracy = float(np.mean(pr == tg))
        self.writer.add_scalar("train/total_loss", total_loss, self.iteration)
        self.writer.add_scalar("train/total_accuracy", total_accuracy, self.iteration)

    def train_step(self, batch, labels, losses=None, predictions=None, targets=None, print_in_console=False):
        label_pred = self._predict(batch)

        next_grad = self.loss.backward(label_pred, labels)
        for layer in self.layers[::-1]:
            next_grad = layer.backward(next_grad)

        params = []
        grads = []
        for layer in self.layers:
            params += layer.params
            grads += layer.grads

        self.optimizer.minimize(params, grads)

        batch_loss = self.loss.forward(label_pred, labels)
        batch_accuracy = np.mean(np.argmax(label_pred, axis=1) == np.argmax(labels, axis=1))

        if print_in_console:
            print(f"Loss: {batch_loss}, Accuracy: {batch_accuracy}")

        if losses is not None:
            losses.append(batch_loss)
        if predictions is not None:
            predictions.extend(label_pred)
        if targets is not None:
            targets.extend(labels)
        self.writer.add_scalar("batch_loss", batch_loss, self.iteration)
        self.writer.add_scalar("batch_accuracy", batch_accuracy, self.iteration)
        self.iteration += 1
        return batch_loss, batch_accuracy

    def _predict(self, x):
        x_next = x
        for layer in self.layers[:]:
            x_next = layer.forward(x_next)
        return x_next  # Предсказанное значение

    def _validate(self, validation_data):
        valid_losses, valid_predictions, valid_targets = [], [], []
        for batch, label in validation_data.batch_generator():
            label_pred = self._predict(batch)

            valid_losses.append(self.loss.forward(label_pred, label))
            valid_predictions.extend(label_pred)
            valid_targets.extend(label)
        total_loss = float(np.mean(self._train_losses))
        pr = np.argmax(valid_predictions, axis=1)
        tg = np.argmax(valid_targets, axis=1)
        total_accuracy = float(np.mean(pr == tg))
        self.writer.add_scalar("valid/total_loss", total_loss, self.iteration)
        self.writer.add_scalar("valid/total_accuracy", total_accuracy, self.iteration)

    def test(self, test_data):
        test_losses, test_predictions, test_targets = [], [], []
        self.iteration = 0
        for batch, labels in test_data.batch_generator():
            label_pred = self._predict(batch)
            test_loss = self.loss.forward(label_pred, labels)
            test_accuracy = np.mean(np.argmax(label_pred, axis=1) == np.argmax(labels, axis=1))
            test_losses.append(test_loss)
            test_predictions.extend(label_pred)
            test_targets.extend(labels)
            self.writer.add_scalar("test/batch_loss", test_loss, self.iteration)
            self.writer.add_scalar("test/batch_accuracy", test_accuracy, self.iteration)
            self.iteration += 1
