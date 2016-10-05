import numpy as np
from AbstractLearningSchedule import AbstractLearningSchedule
class Adam(AbstractLearningSchedule):
    def __init__(self, eta, beta1 = 0.8, beta2 = 0.9, epsilon = 1e-8, etaDecay = False, etaDecayFactor = 0, etaDecayEpoch = 0):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.epoch = 1
        self.etaDecay = etaDecay
        self.etaDecayFactor = etaDecayFactor
        self.etaDecayEpoch = etaDecayEpoch

    def update(self, weightsDict, nSamples):
        for key in weightsDict:
            if key not in self.m:
                self.m[key] = (1 - self.beta1) * weightsDict[key]
                self.v[key] = (1 - self.beta2) * weightsDict[key]**2
            else:
                self.m[key] = self.beta1 * self.m[key] + \
                                (1 - self.beta1) * weightsDict[key]
                self.v[key] = self.beta2 * self.v[key] + \
                                (1 - self.beta2) * weightsDict[key]**2


            m = self.m[key] / (1 - self.beta1 ** self.epoch)
            v = self.v[key] / (1 - self.beta2 ** self.epoch)
            weightsDict[key] = self.eta * (m /(np.sqrt(v) + self.epsilon))

    def updateEpoch(self):
        self.epoch += 1
        if self.etaDecay and (self.epoch % self.etaDecayEpoch == 0):
            self.eta *= self.etaDecayFactor
