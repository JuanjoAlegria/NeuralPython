class SimpleEta:
    def __init__(self, eta):
        self.eta = 1.0 * eta

    def update(self, weightsDict, nSamples):
        for key in weightsDict:
            weightsDict[key] *= (self.eta / nSamples)


