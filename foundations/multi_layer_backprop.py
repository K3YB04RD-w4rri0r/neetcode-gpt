import numpy as np
from typing import List


class Solution:
    @staticmethod
    def ReLU(x):
        return np.maximum(0,x)
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        # forward
        x = np.array(x).reshape(1, -1)
        y_true = np.array(y_true).reshape(1, -1)
        W1 = np.array(W1)
        W2 = np.array(W2)
        b1 = np.array(b1).reshape(1, -1)
        b2 = np.array(b2).reshape(1, -1)

        a0 = x
        a1 = self.ReLU(a0 @ W1.T + b1)
        a2 = a1 @ W2.T + b2 # self.linear 
        yhat = a2

        # loss
        loss = np.mean((yhat - y_true) ** 2)

        d2 = 2 * (yhat - y_true) / y_true.size
        dW2 = np.outer(d2, a1)
        db2 = d2

        d1 = (d2 @ W2) * self.relu_derivative(a1)
        dW1 = np.outer(d1, a0)
        db1 = d1

        return {
            'loss': np.round(loss, 4),
            'dW1': np.round(dW1, 4),
            'db1': np.round(np.squeeze(db1).tolist(), 4),
            'dW2': np.round(dW2, 4),
            'db2': np.round(db2.flatten().tolist(), 4)
        }


        
        
