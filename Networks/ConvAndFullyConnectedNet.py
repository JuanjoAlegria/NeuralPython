import numpy as np
from NeuralPython.Layers.PoolLayer import PoolLayer
from AbstractNetwork import AbstractNetwork

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


    def forward(self, x, test = False):
        results = {}
        for i in range(len(self.channels)):
            c = self.channels[i]
            result = c.forward([x[i]], test)
            results[c.id] = result
        return results

    def backward(self, deltas):
        for i in range(len(self.channels)):
            c = self.channels[i]
            c.backward(deltas[i])

    def updateParameters(self, weightsDict, nSamples):
        self.regularizationFunction.setNSamples(nSamples)
        for layer in self:
            if isinstance(layer, PoolLayer):
                continue
            biases = weightsDict[str(layer.layerId) + "biases"]
            weights = weightsDict[str(layer.layerId) + "weights"]
            layer.updateParameters(biases, weights, self.regularizationFunction)

    def save(self, directory):
        for layer in self:
            layer.save(directory)

    def load(self, directory):
        for layer in self:
            layer.load(directory)

    def __iter__(self):
        return ConvNetIterator(self)

    def getInputShape(self):
        inputSize = self.channels[0].layers[0].inputSize
        nChannels = len(self.channels)
        result = np.zeros(len(inputSize) + 1)
        result[0] = nChannels
        result[1:] = inputSize
        return result

    def getOutputSize(self):
        s = 0
        for c in self.channels:
            s += c.getOutputSize()
        return s

    def getNLayers(self):
        s = 0
        for c in self.channels:
            s += c.getNLayers()
        return s



class ConvFFNetIterator:
    def __init__(self, net):
        self.network = net
        self.currentIterator = self.network.convNet.__iter__()
        self.inChannels = True

    def next(self):
        if self.inChannels:
            try:
                return self.currentIterator.next()
            except StopIteration:
                self.currentIterator = self.network.ffNet.__iter__()
                self.inChannels = False
                return self.currentIterator.next()

        else: return self.currentIterator.next()


class ConvAndFullyConnectedNet(AbstractNetwork):
    def __init__(self, convNet, ffNet, costFunction, \
                 regularizationFunction, regression):
        self.convNet = convNet
        self.ffNet = ffNet
        self.costFunction = costFunction
        self.regularizationFunction = regularizationFunction
        self.regression = regression
        self.outputSize = self.ffNet.outputSize

    def forward(self, x, test = False):
        convResults = self.convNet.forward(x, test)
        inputFF = self.buildInputFeedForward(convResults)
        finalResult = self.ffNet.forward(inputFF, test)
        return finalResult

    def backward(self, y):
        delta = self.ffNet.backward(y)
        deltaConv = self.buildDeltaConv(delta)
        self.convNet.backward(deltaConv)

    def updateParameters(self, weightsDict, nSamples):
        self.convNet.updateParameters(weightsDict, nSamples)
        self.ffNet.updateParameters(weightsDict, nSamples)


    def save(self, directory):
        self.convNet.save(directory)
        self.ffNet.save(directory)

    def load(self, directory):
        self.convNet.load(directory)
        self.ffNet.load(directory)

    def __iter__(self):
        return ConvFFNetIterator(self)


    def buildInputFeedForward(self, convResults):
        result = []
        channels = self.convNet.channels
        for i in range(len(channels)):
            currentChannel = convResults[i]
            chRs = []
            for l in currentChannel:
                chRs = np.concatenate((chRs, l.flatten()))
            result = np.concatenate((result, chRs))
        return result

    def buildDeltaConv(self, dNext):
        firstFFLayer = self.ffNet.layers[0]
        wNext = firstFFLayer.getWeights()
        reshapedDNext = np.reshape(dNext, (len(dNext), 1))
        delta = np.dot(wNext.transpose(), reshapedDNext)
        result = {}
        startIndex = 0
        for c in self.convNet.channels:
            outputSize = c.getOutputSize()
            result[c.id] = delta[startIndex: startIndex + outputSize]
            startIndex += outputSize
        return result



