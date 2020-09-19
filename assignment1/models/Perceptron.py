"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int, dim: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            dim: the dimension of an input sample
        """
        self.w = np.random.random((n_class, dim + 1))
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in Lecture 3.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):
                train = np.append(X_train[i], 1).T
                label_ = y_train[i]
                for class_ in range(self.n_class):
                    if self.w[class_] @ train > self.w[label_] @ train:
                        self.w[label_] = self.w[label_] + self.lr * train
                        self.w[class_] = self.w[class_] - self.lr * train
        return self.w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        pre_label = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            test = np.append(X_test[i], 1).T
            score = self.w @ test
            pre_label[i] = np.argmax(score)
        return pre_label
