import numpy as np
from NeuralPython.Layers.PoolLayer import PoolLayer

class ConvNetIterator:
    def __init__(self, network):
        self.network = network
        self.channelId = 0
        self.currentLayer = 0

    def next(self):
        channels = self.network.channels
        if self.channelId < len(channels):
            layers = channels[self.channelId].getLayers()
            if self.currentLayer < len(layers):
                self.currentLayer += 1
                return layers[self.currentLayer - 1]
            else:
                self.channelId += 1
                self.currentLayer = 0
                return self.next()
        else:
            raise StopIteration()

class ConvNet:
    def __init__(self, channels, regularizationFunction):
        self.channels = channels
        self.regularizationFunction = regularizationFunction

    def getNLayers(self):
        s = 0
        for c in self.channels:
            s += c.getNLayers()
        return s

    def getOutputSize(self):
        s = 0
        for c in self.channels:
            s += c.getOutputSize()
        return s

    def forward(self, x, test = False):
        results = {}
        for i in range(len(self.channels)):
            c = self.channels[i]
            result = c.forward([x[i]], test)
            results[c.id] = result
        return results

    def updateParameters(self, weightsDict, nSamples):
        self.regularizationFunction.setNSamples(nSamples)
        for layer in self:
            if isinstance(layer, PoolLayer):
                continue
            biases = weightsDict[str(layer.layerId) + "biases"]
            weights = weightsDict[str(layer.layerId) + "weights"]
            layer.updateParameters(biases, weights, self.regularizationFunction)


    def backward(self, deltas):
        for i in range(len(self.channels)):
            c = self.channels[i]
            c.backward(deltas[i])

    def getInputShape(self):
        inputSize = self.channels[0].layers[0].inputSize
        nChannels = len(self.channels)
        result = np.zerps(len(inputSize) + 1)
        result[0] = nChannels
        result[1:] = inputSize
        return result

    def save(self, directory):
        for layer in self:
            layer.save(directory)

    def load(self, directory):
        for layer in self:
            layer.load(directory)

    def __iter__(self):
        return ConvNetIterator(self)


