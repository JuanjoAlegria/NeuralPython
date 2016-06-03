class L2Reg:
    def __init__(self, lambdaReg):
        self.lambdaReg = lambdaReg
        self.eta = 0
        self.nSamples = 0

    def setEta(self, eta):
        self.eta = eta

    def setNSamples(self, nSamples):
        self.nSamples = nSamples

    def weightsDerivation(self, weights):
        return (self.eta* self.lambdaReg / self.nSamples) * weights
