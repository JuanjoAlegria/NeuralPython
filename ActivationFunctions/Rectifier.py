import numpy as np

class Rectifier:
    def __init__(self):
        pass

    def function(self, vector):
        return np.fmax(vector, np.zeros(len(vector)))

    def derivative(self, vector):
        def singleDerivative(x):
            if x > 0: return 1
            else: return 0
        f = np.vectorize(singleDerivative)
        return f(vector)
