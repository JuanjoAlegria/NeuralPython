from FilterLayer import FilterLayer
from PoolLayer import PoolLayer

class Channel:
    def __init__(self, channelId, inputSize, stringRep, activationFunction, \
                 inputDimension, idFirstLayer = 0):
        self.id = channelId
        self.inputSize = inputSize
        self.outputSize = 0
        self.nOutputs = 0
        self.activationFunction = activationFunction
        self.dimension = inputDimension
        self.layers = self.buildLayers(stringRep, idFirstLayer)


    def buildLayers(self, stringRep, idFirstLayer):
        layers = []
        nInputs = 1
        inputSize = self.inputSize.copy()
        layerId = idFirstLayer
        for s in stringRep:
            a = s.split("-")
            if a[0] == "Conv":
                nFilters = int(a[1])
                filtersSize = int(a[2])
                l = FilterLayer(nInputs, inputSize.copy(), self.dimension, layerId, self.id, \
                                nFilters, filtersSize, self.activationFunction)
                nInputs = nFilters
                inputSize = inputSize - filtersSize + 1
                layerId += 1
                layers.append(l)

            if a[0] == "Pool":
                fPool = a[1]
                stepPool = int(a[2])
                l = PoolLayer(nInputs, inputSize.copy(), self.dimension, layerId, self.id, fPool, stepPool)
                inputSize /= stepPool
                layerId += 1
                layers.append(l)

        self.outputSize = inputSize
        self.nOutputs = nInputs


        for i in range(len(layers) - 1):
            layers[i].setNextLayer(layers[i+1])

        for i in range(1, len(layers)):
            layers[i].setPreviousLayer(layers[i-1])

        return layers

    def getNFlattenedOutputs(self):
        x = self.outputSize.prod() * self.nOutputs
        print x
        return x

    def getLayers(self):
        return self.layers

    def getNLayers(self):
        return len(self.layers)

    def forward(self, xTrain):
        return self.layers[0].forward(xTrain)

    def updateParameters(self, eta, nSamples):
        for l in self.layers:
            l.updateParameters(eta, nSamples)

    def save(self, directory):
        for l in self.layers:
            l.save(directory)

    def load(self, directory):
        for l in self.layers:
            l.load(directory)


