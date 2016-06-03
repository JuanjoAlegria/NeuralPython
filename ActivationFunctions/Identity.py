import numpy as np
class Identity:
    def __init__(self):
        pass

    def function(self, vector):
        return vector

    def derivative(self, actualOutput, desiredOutput):
        return np.ones(actualOutput.shape)
