"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, dim: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
            dim: the dimension of an input sample
        """
        self.w = np.random.random((1, dim))
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        output = 1.0 / (1.0 + np.exp(-z))
        return output

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        for epoch in range(self.epochs):
            tmp = 0.0
            n = X_train.shape[0]
            for i in range(n):
                train = X_train[i].T
                label_ = -1 if y_train[i] == 0 else 1
                tmp += label_ * self.sigmoid(-label_ * self.w @ train) * train
            self.w += self.lr * (tmp / n)

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
            test = X_test[i].T
            score = self.w @ test
            pre_label[i] = 1 if score > 0 else 0
        return pre_label
