import numpy as np

class Adam:
    def __init__(self, eta, beta1 = 0.8, beta2 = 0.9, epsilon = 1e-8):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, weightsDict, nSamples):
        self.t += 1
        for key in weightsDict:
            if key not in self.m:
                self.m[key] = (1 - self.beta1) * weightsDict[key]
                self.v[key] = (1 - self.beta2) * weightsDict[key]**2
            else:
                self.m[key] = self.beta1 * self.m[key] + \
                                (1 - self.beta1) * weightsDict[key]
                self.v[key] = self.beta2 * self.v[key] + \
                                (1 - self.beta2) * weightsDict[key]**2

            m = self.m[key] / (1 - self.beta1 ** self.t)
            v = self.v[key] / (1 - self.beta2 ** self.t)
            weightsDict[key] = self.eta * (m /(np.sqrt(v) + self.epsilon))
