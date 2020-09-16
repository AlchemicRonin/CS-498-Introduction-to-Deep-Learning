"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, w: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = np.random.random((n_class,w))  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        batch_n = X_train.shape[0]
        for epoch in range(self.epochs): 
          for i in range(len(X_train)):
            y = y_train[i]
            for c in range(self.n_class):
              self.w[c,:] = (1- self.lr*self.reg_const/batch_n)*self.w[c,:]
            for c in range (self.n_class):
              if (X_train[i,:].dot(self.w[c,:]) - X_train[i,:].dot(self.w[y,:]))>1:
                self.w[y,:] = self.w[y,:] + self.lr*X_train[i,:]
                break
            for c in range(self.n_class):
              if (X_train[i,:].dot(self.w[c,:]) - X_train[i,:].dot(self.w[y,:]))>1:
                self.w[c,:] = self.w[c,:] - self.lr*X_train[i,:]
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
        # TODO: implement me
        t = np.zeros(X_test.shape[0])
        f = np.random.random((X_test.shape[0],self.n_class))
        a = 0
        for i in range(len(X_test)):
          for c in range(self.n_class):
            f[i,c] = X_test[i,:].dot(self.w[c,:])
          a = np.argmax(f[i,:]) 
          t[i] = a
        return t
