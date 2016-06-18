import numpy as np
from NeuralPython.Layers.AbstractLayer import AbstractLayer
from NeuralPython.CostFunctions.LogLikelihood import LogLikelihood
from NeuralPython.ActivationFunctions.Softmax import Softmax

class FullyConnectedLayer(AbstractLayer):
    def __init__(self, inputSize, nUnits, layerId, activationFunction, finalLayer = False):
        super(FullyConnectedLayer, self).__init__(1, inputSize, layerId)
        self.nUnits = nUnits
        self.activationFunction = activationFunction
        self.finalLayer = finalLayer
        self.weights = np.random.normal(0, 1.0 / inputSize, (nUnits, inputSize))
        self.biases = np.random.randn(nUnits)
        self.deltaWeights = np.zeros(self.weights.shape)
        self.deltaBiases = np.zeros(self.biases.shape)
        self.nSamples = 0.0
        self.output = None

    def forward(self, x, test = False):
        z = np.dot(self.weights, x) + self.biases
        result = self.activationFunction.function(z)
        if not test:
            self.input = x
            self.z = z

        if self.finalLayer:
            if not test:
                self.output = result
            return result

        return self.nextLayer.forward(result)

    def backward(self, dNext = None, desiredOutput = None, costFunction = None):
        if self.finalLayer:
            deltas = self.calculateDeltaOutputLayer(costFunction, desiredOutput)
        else:
            wNext = self.nextLayer.getWeights()
            # calcular deltas
            deltas = np.dot(wNext.transpose(), dNext) * \
                    self.activationFunction.derivative(self.z)
        # calcular delta biases
        deltaBiases = np.copy(deltas)

        # calcular delta pesos
        reshapedDelta = np.reshape(deltas, (self.nUnits, 1))
        reshapedInput = np.reshape(self.input, (self.inputSize, 1))
        deltaWeights =  np.dot(reshapedDelta, reshapedInput.transpose())

        # guardar deltas
        self.deltaWeights += deltaWeights
        self.deltaBiases += deltaBiases
        self.nSamples += 1

        if self.previousLayer is None:
            return deltas
        else:
            return self.previousLayer.backward(deltas)
    def getWeights(self):
        return self.weights

    def getParameters(self):
        return self.biases, self.weights

    def calculateParameters(self):
        return self.deltaBiases / self.nSamples, self.deltaWeights / self.nSamples

    def updateParameters(self, biasesDelta, weightsDelta, regularization):
        self.biases -= biasesDelta
        self.weights -= weightsDelta + regularization.weightsDerivation(self.weights)
        self.deltaWeights = np.zeros(self.weights.shape)
        self.deltaBiases = np.zeros(self.biases.shape)
        self.nSamples = 0.0

    def calculateDeltaOutputLayer(self, costFunction, desiredOutput):
        if isinstance(costFunction, LogLikelihood) \
        and isinstance(self.activationFunction, Softmax):
            return self.output - desiredOutput
        else:
            return costFunction.derivative(self.output, desiredOutput) * \
                            self.activationFunction.derivative(self.z, desiredOutput)

    def save(self, directory):
        baseFilename = directory + "fullyConnectedLayer" + str(self.layerId)
        np.save(open(baseFilename + "_biases.npy", "w"), self.biases)
        np.save(open(baseFilename + "_weights.npy", "w"), self.weights)

    def load(self, directory):
        biasesFile = directory + "fullyConnectedLayer" + str(self.layerId) + "_biases.npy"
        weightsFile = directory + "fullyConnectedLayer" + str(self.layerId) + "_weights.npy"

        self.biases = np.load(open(biasesFile))
        self.weights = np.load(open(weightsFile))









