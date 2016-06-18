class SimpleEta:
    def __init__(self, eta, etaDecay = False, etaDecayFactor = 0, etaDecayEpoch = 0):
        self.eta = 1.0 * eta
        self.epoch = 1
        self.etaDecay = etaDecay
        self.etaDecayFactor = etaDecayFactor
        self.etaDecayEpoch = etaDecayEpoch

    def update(self, weightsDict, nSamples):
        for key in weightsDict:
            weightsDict[key] *= (self.eta / nSamples)

    def updateEpoch(self):
        self.epoch += 1
        if self.etaDecay and (self.epoch % self.etaDecayEpoch == 0):
            self.epoch *= self.etaDecayFactor
