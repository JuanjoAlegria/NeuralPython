class SimpleEta:
    def __init__(self, eta):
        self.eta = eta

    def update(self, weightsDict, nSamples):
        for key in weightsDict:
            weightsDict[key] *= (self.eta / nSamples)


