import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector

        eps = 1e-5
        mean = np.mean(x) # ,axis=1)
        std = np.std(x)   
        return np.round(beta + gamma * (x - mean)/(np.sqrt(std ** 2 + eps)), 5)

