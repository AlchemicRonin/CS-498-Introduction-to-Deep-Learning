"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = np.random.random((22,1))  # TODO: change this
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
        # TODO: implement me
        return 1/(1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        for epoch in range(self.epochs): 
          b = np.random.random((X_train.shape[0],1))
          a = np.random.random((X_train.shape[0],1))
          p = np.random.random((X_train.shape[0],X_train.shape[1]))
          for i in range(len(X_train)):
            b[i,:] = -y_train[i]*np.dot(X_train[i,:],self.w)
          a = self.sigmoid(b)
          for i in range(len(X_train)):
            p[i,:] = a[i,:]*y_train[i]*X_train[i,:]
          self.w = np.add(self.w,self.lr*np.expand_dims(np.mean(p,axis = 0), axis=1))
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
        for i in range(len(X_test)):
          if X_test[i,:].dot(self.w)>0:
            t[i] = 1
          else:
            t[i] = -1
        return t
