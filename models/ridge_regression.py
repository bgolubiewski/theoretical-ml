#!/usr/bin/env python3

import numpy as np
import pytest
from sklearn.linear_model import Ridge

class RidgeRegr:
    """
    Ridge regression model using gradient descent.
    """
    def __init__(self, alpha = 0.0, learning_rate = 0.01):
        """
        Initialize the Ridge regression model

        :param alpha: float. Regularization strength. Controls the
                      penalty applied to the model coefficients.
        :param learning_rate: float. Step size used to update the
                              parameters during gradient descent.
        """
        self.alpha = alpha
        self.learning_rate = learning_rate

    def fit(self, X, Y) -> RidgeRegr:
        """
        Fit the Ridge regression model to the training data.

        Estimates the optimal parameters (theta) iteratively using
        gradient descent to minimize the squared error with regularization.
        The intercept is excluded from regularization.

        A column of ones is added to X to account for the intercept.

        :param X: np.ndarray, shape (n,m). Training data, where n is the number
                  of samples and m is the number of features
        :param Y: np.ndarray, shape (n,). Target values corresponding 
                  to the training samples.
        
        :return: RidgeRegr. The fitted model with parameters in self.theta.
        """
        n, m = X.shape

        X = np.column_stack((np.ones(n), X))
        X_transposed = np.transpose(X)

        self.theta = np.zeros(m + 1)

        for _ in range(20000):
            Y_pred = np.dot(X, self.theta)
            error = Y - Y_pred

            reg_penalty = np.copy(self.theta)
            reg_penalty[0] = 0

            # Gradient = -X^T * (Y - X*theta) + alpha * reg_penalty
            gradient = -np.dot(X_transposed, error) + self.alpha * reg_penalty

            self.theta = self.theta - self.learning_rate * gradient 

        return self
    
    def predict(self, X):
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


def test_RidgeRegressionInOneDim():
    X = np.array([1,3,2,5]).reshape((4,1))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,2,10]).reshape((3,1))
    alpha = 0.3
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-5)

def test_RidgeRegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    alpha = 0.4
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    assert list(actual) == pytest.approx(list(expected), rel=1e-3)
    
