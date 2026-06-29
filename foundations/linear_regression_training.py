import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64]) -> float:
        return -2 * np.dot(X.T, ground_truth - model_prediction) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        # pass 
        W = initial_weights
        N = len(Y)
        for i in range(num_iterations):
            logits = self.get_model_prediction(X, W)
            print(logits)
            gradient = self.get_derivative(logits, Y, N, X)
            W = W - self.learning_rate * gradient

        return np.round(W, 5)
