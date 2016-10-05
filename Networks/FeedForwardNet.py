from AbstractNetwork import AbstractNetwork


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


class FeedForwardNet(AbstractNetwork):
    def __init__(self, layers, costFunction, regularizationFunction, regression):
        self.layers = layers
        self.costFunction = costFunction
        self.regularizationFunction = regularizationFunction
        self.regression = regression
        self.outputSize = self.layers[-1].nUnits
        self.tieLayers()

    def forward(self, x, test = False):
        return self.layers[0].forward(x, test)

    def backward(self, y):
        return self.layers[-1].backward(None, y, self.costFunction)

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

    def tieLayers(self):
        for i in range(len(self.layers) - 1):
            self.layers[i].setNextLayer(self.layers[i + 1])

        for i in range(1, len(self.layers)):
            self.layers[i].setPreviousLayer(self.layers[i - 1])

