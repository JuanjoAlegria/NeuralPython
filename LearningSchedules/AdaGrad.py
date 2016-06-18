import numpy as np

class AdaGrad:
    def __init__(self, eta, forget = 0, etaDecay = False, \
                 etaDecayFactor = 0, etaDecayEpoch = 0):
        self.eta = eta
        self.historyDict = {}
        self.forget = forget
        self.epoch = 1
        self.etaDecay = etaDecay
        self.etaDecayFactor = etaDecayFactor
        self.etaDecayEpoch = etaDecayEpoch

    def update(self, weightsDict, nSamples):
        for key in weightsDict:
            if key not in self.historyDict:
                self.historyDict[key] = weightsDict[key]**2
            else:
                self.historyDict[key] += weightsDict[key]**2

            weightsDict[key] = self.eta * (weightsDict[key] \
                                           / (np.sqrt(self.historyDict[key] + 1e-8)))
        if self.forget != 0 and (self.epoch % self.forget == 0):
            self.historyDict = {}

    def updateEpoch(self):
        self.epoch += 1
        if self.etaDecay and (self.epoch % self.etaDecayEpoch == 0):
            self.epoch *= self.etaDecayFactor
