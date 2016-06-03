import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def function(self, vector):
        return 1.0 / (1.0 + np.exp(-vector))

    def derivative(self, actualOutput):
        return self.function(actualOutput) * (1 - self.function(actualOutput))
