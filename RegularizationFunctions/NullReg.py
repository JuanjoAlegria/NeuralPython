from AbstractRegularizationFunction import AbstractRegularizationFunction
class NullReg(AbstractRegularizationFunction):
    def __init__(self):
        pass

    def setEta(self, eta):
        pass

    def setNSamples(self, nSamples):
        pass

    def weightsDerivation(self, weights):
        return 0
