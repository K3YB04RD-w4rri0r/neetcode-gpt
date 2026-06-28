import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        maxi = np.max(z)
        expod = np.exp(z - maxi)
        ssum = np.sum(expod)
        return np.round(expod / ssum, 4)
