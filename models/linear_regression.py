#!/usr/bin/env python3

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

class LinearRegr:
    """
    Linear regression model.
    """
    def fit(self, X, Y) -> LinearRegr:
        """
        Fit the linear regression model to the training data.

        Calculates the optimal parameters (theta) by minimizing the quadratic cost function
        using the normal equation with the Moore-Penrose pseudoinverse.

        A column of ones is added to X to account for the intercept.

        :param X: np.ndarray, shape (n,m). Training data, where n is the number of samples and m is the number of features
        :param Y: np.ndarray, shape (n,). Target values corresponding to the training samples.

        :return: LinearRegr. The fitted linear regression model with parameters in self.theta.
        """ 
        n, m = X.shape
        X = np.column_stack((np.ones(n), X))
        X_transposed = np.transpose(X)

        self.theta = np.zeros((m+1))

        # (X^T * X) ^(-1) * X^T * Y
        self.theta = np.dot(np.linalg.pinv(np.dot(X_transposed, X)), np.dot(X_transposed, Y))

        return self
    
    def predict(self, X) -> np.ndarray:
        """
        Predict target values for the given input data.

        Adds a column of ones to X to account for the intercept
        and calculates predictions using the learned parameters.

        :param X: np.ndarray, shape (k, m)
        Input data, where k is the number of samples and m is the number of features.

        :return: np.ndarray, shape (k,). Predicted target values for each input sample.
        """
        k, m = X.shape
        X = np.column_stack((np.ones(k), X))

        hypothesis_function = np.dot(X, self.theta)

        return hypothesis_function


def test_RegressionInOneDim():
    X = np.array([1,3,2,5]).reshape((4,1))
    Y = np.array([2,5, 3, 8])
    a = np.array([1,2,10]).reshape((3,1))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    assert list(actual) == pytest.approx(list(expected))

def test_RegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    a = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    assert list(actual) == pytest.approx(list(expected))