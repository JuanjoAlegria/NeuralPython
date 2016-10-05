from abc import ABCMeta, abstractmethod

class AbstractRegularizationFunction:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def setEta(self, eta):
        pass

    @abstractmethod
    def setNSamples(self, nSamples):
        pass

    @abstractmethod
    def weightsDerivation(self, weights):
        pass
