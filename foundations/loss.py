import numpy as np
from numpy.typing import NDArray


class Solution:
    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # pi = 1/ (1 + np.exp( - y_pred)) 
        return round(-np.mean((y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred) )), 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return round(- np.mean( np.sum(y_true * np.log(y_pred), axis = 1)), 4)