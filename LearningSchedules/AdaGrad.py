import numpy as np

class AdaGrad:
    def __init__(self, eta, forget = 0):
        self.eta = eta
        self.historyDict = {}
        self.forget = forget
        self.count = 0

    def update(self, weightsDict, nSamples):
        self.count += 1
        for key in weightsDict:
            if key not in self.historyDict:
                self.historyDict[key] = weightsDict[key]**2
            else:
                self.historyDict[key] += weightsDict[key]**2

            weightsDict[key] = self.eta * (weightsDict[key] \
                                           / (np.sqrt(self.historyDict[key] + 1e-8)))
        if self.forget != 0 and (self.count % self.forget == 0):
            self.historyDict = {}
