from NeuralPython.Layers.AbstractLayer import AbstractLayer
from NeuralPython.Layers.FilterLayer import FilterLayer
from NeuralPython.PoolingFunctions.MaxPooling import MaxPooling
from NeuralPython.PoolingFunctions.MaxPooling2D import MaxPooling2D
import numpy as np
from scipy import signal


class PoolLayer(AbstractLayer):
    def __init__(self, nInputs, inputSize, inputDimension, layerId, channelId, fString, step):
        super(PoolLayer, self).__init__(nInputs, inputSize, layerId, channelId)
        self.step = step
        self.dimension = inputDimension
        self.poolFunction = self.buildPoolFunction(fString)

    def buildPoolFunction(self, fString):
        if fString.lower() == "max":
            if self.dimension == 1:
                return MaxPooling(self.inputSize, self.step)
            elif self.dimension == 2:
                return MaxPooling2D(self.inputSize, self.step)
        else:
            raise Exception()

    def forward(self, x):
        result = []
        for i in range(self.nInputs):
            v = x[i]
            result.append(self.poolFunction.down(v, i))
        if self.nextLayer is None:
            return result
        else:
            return self.nextLayer.forward(result)


    def backward(self, dNext):
        deltas = []
        if self.nextLayer is None:
            singleOutputSize = self.inputSize.prod() / (self.step ** self.dimension)
            for k in range(self.nInputs):
                a = k*singleOutputSize
                d = dNext[a : a + singleOutputSize]
                if self.dimension == 2:
                    d = np.reshape(d, self.inputSize / self.step)

                up_d = self.poolFunction.up(d, k)
                deltas.append(up_d)


        elif type(self.nextLayer) == FilterLayer:
            fNext = self.nextLayer.getFilters()
            for i in range(self.nInputs):
                delta = np.zeros(self.inputSize/self.step)
                for j in range(len(dNext)):
                    delta += signal.fftconvolve(dNext[j], fNext[i][j], 'full')
                deltas.append(self.poolFunction.up(delta, i))

        self.previousLayer.backward(deltas)

    def getStepSize(self):
        return self.step

    def getOutputSize(self):
        return self.inputSize / self.step

    def getNOutputs(self):
        return self.nInputs

    def calculateParameters(self):
        return None, None

    def updateParameters(self, biases, weights):
        pass

    def save(self, directory):
        pass

    def load(self, directory):
        pass

