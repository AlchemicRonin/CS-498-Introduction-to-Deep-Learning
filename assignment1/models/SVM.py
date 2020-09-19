"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, dim: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
            dim: the dimension of an input sample
        """
        self.w = np.random.random((n_class, dim + 1))
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        for i in range(X_train.shape[0]):
            train = np.append(X_train[i], 1).T
            label_ = y_train[i]
            for class_ in range(self.n_class):
                self.w[class_] *= 1 - self.alpha * self.reg_const / self.n_class
                if class_ == label_:
                    continue
                if self.w[label_] @ train - self.w[class_] @ train < 1:
                    self.w[label_] += self.alpha * train
                    self.w[class_] -= self.alpha * train
        return self.w

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        for epoch in range(self.epochs):
            self.calc_gradient(X_train, y_train)

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
