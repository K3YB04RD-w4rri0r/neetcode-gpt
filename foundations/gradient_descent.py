class Solution:
    def df(self,x):
        return 2*x
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x = init
        for i in range(iterations):
            x = x - learning_rate * self.df(x)

        return round(x, 5)
            