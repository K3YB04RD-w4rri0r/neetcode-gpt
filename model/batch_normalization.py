import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        x = np.array(x)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

        if training == False:
            # mean, var  = running_mean[-1], running_var[-1]
            return np.round(beta + gamma *(x - running_mean) / np.sqrt(running_var + eps),4), np.round(running_mean, 4), np.round(running_var, 4)


        else: 
            batch_mean = np.mean(x, axis=0)          # shape: (D,)
            batch_var  = np.var(x, axis=0)           # shape: (D,)

            running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            running_var = (1 - momentum) * running_var + momentum * batch_var

            return np.round(beta + gamma * (x - batch_mean) / np.sqrt(batch_var + eps),4), np.round(running_mean, 4), np.round(running_var, 4)



