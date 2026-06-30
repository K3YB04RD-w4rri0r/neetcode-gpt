import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    @staticmethod
    def sigmoid(z):
        print(z)
        return 1 / (1 + np.exp(-z))

    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        if activation == "relu":
            act_fn = self.relu
        if activation == "sigmoid":
            act_fn = self.sigmoid

        return np.round(act_fn(x @ w.T + b), 5)

    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        y_hat = self.forward(x, w, b, "sigmoid")

        dL_db = (y_hat - y_true)*(y_hat)*(1 - y_hat)
        dL_dw = dL_db * x
        dL_dw = np.round(dL_dw, 5)
        dL_db = np.round(dL_db, 5)
        return (dL_dw,dL_db)