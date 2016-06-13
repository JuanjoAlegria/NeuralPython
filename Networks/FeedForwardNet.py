import numpy as np

class FFNetIterator:
    def __init__(self, network):
        self.network = network
        self.currentLayer = 0

    def next(self):
        layers = self.network.layers
        if self.currentLayer < len(layers):
            self.currentLayer += 1
            return layers[self.currentLayer - 1]
        else:
            raise StopIteration()


class FeedForwardNet:
    def __init__(self, layers, costFunction, regularizationFunction, regression):
        self.layers = layers
        self.costFunction = costFunction
        self.regularizationFunction = regularizationFunction
        self.regression = regression
        self.tieLayers()

    def getNLayers(self):
        return len(self.layers)

    def tieLayers(self):
        for i in range(len(self.layers) - 1):
            self.layers[i].setNextLayer(self.layers[i + 1])

        for i in range(1, len(self.layers)):
            self.layers[i].setPreviousLayer(self.layers[i - 1])

    def forward(self, x):
        return self.layers[0].forward(x)

    def backward(self, y):
        return self.layers[-1].backward(None, y, self.costFunction)

    def error(self, data):
        totalError = 0
        samples = zip(*data)
        for sample in samples:
            result = self.forward(*sample[:-1])
            totalError += self.costFunction.function(result, sample[-1])
        return 1.0 * totalError / len(data[0])

    def train(self, data):
        samples = zip(*data)
        for sample in samples:
            self.forward(*sample[:-1])
            self.backward(sample[-1])

    def test(self, data, epsilon):
        s = 0
        samples = zip(*data)
        for sample in samples:
            if self.evaluateOneSample(sample, epsilon) == 1:
                s += 1
        return s

    def evaluateOneSample(self, sample, epsilon):
        actualOutput = self.forward(*sample[:-1])
        desiredOutput = sample[-1]
        if self.regression:
            return int(np.abs(actualOutput - desiredOutput) <= epsilon)
        else:
            return int(np.argmax(actualOutput) == np.argmax(desiredOutput))


    def updateParameters(self, weightsDict, nSamples):
        self.regularizationFunction.setNSamples(nSamples)
        for layer in self:
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
        return FFNetIterator(self)
