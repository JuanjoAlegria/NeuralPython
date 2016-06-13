class Channel:
    def __init__(self, channelId, layers):
        self.id = channelId
        self.layers = layers
        self.tieLayers()

    def tieLayers(self):
        for i in range(len(self.layers) - 1):
            self.layers[i].setNextLayer(self.layers[i+1])

        for i in range(1, len(self.layers)):
            self.layers[i].setPreviousLayer(self.layers[i-1])


    def getOutputSize(self):
        lastLayer = self.layers[-1]
        outputSize = lastLayer.getOutputSize()
        nOutputs = lastLayer.getNOutputs()
        return outputSize.prod() * nOutputs

    def getLayers(self):
        return self.layers

    def getNLayers(self):
        return len(self.layers)

    def forward(self, xTrain):
        return self.layers[0].forward(xTrain)

    def backward(self, deltas):
        return self.layers[-1].backward(deltas)

    def updateParameters(self, eta, nSamples):
        for l in self.layers:
            l.updateParameters(eta, nSamples)

    def save(self, directory):
        for l in self.layers:
            l.save(directory)

    def load(self, directory):
        for l in self.layers:
            l.load(directory)


