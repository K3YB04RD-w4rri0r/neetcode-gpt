import numpy as np
from numpy.typing import NDArray


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
