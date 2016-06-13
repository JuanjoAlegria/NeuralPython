from ActivationFunctions.Rectifier import Rectifier
from ActivationFunctions.Softmax import Softmax
from ActivationFunctions.Sigmoid import Sigmoid
from ActivationFunctions.Identity import Identity
from CostFunctions.LogLikelihood import LogLikelihood
from CostFunctions.Quadratic import Quadratic
from Layers.Channel import Channel
from Layers.FullyConnectedLayer import FullyConnectedLayer
from Layers.PoolLayer import PoolLayer
from RegularizationFunctions.L2Reg import L2Reg
from RegularizationFunctions.NullReg import NullReg
import numpy as np


class NetworkIterator:
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
            layers = self.network.fullyConnectedLayers
            if self.currentLayer < len(layers):
                self.currentLayer += 1
                return layers[self.currentLayer - 1]
            else:
                raise StopIteration()


class Network:
    @staticmethod
    def buildFromDict(d):
        return Network(d['channelReps'], d['fullLayersRep'], d['inputDimension'], \
                d['inputSizeChannels'], d['nInputsExtra'], \
                d['activationString'], d['costString'], d['outputActivationString'],)

    def __init__(self, channelReps, fullLayersRep, inputDimension, \
                 inputSizeChannels, nInputsExtra, \
                 activationString, costString, outputActivationString, \
                 regularizationString):
        self.inputSizeChannels = np.array(inputSizeChannels)
        self.inputFullyConnected = nInputsExtra
        self.inputDimension = inputDimension
        self.outputSize = int(fullLayersRep[-1].split("-")[1])
        self.activationFunction = self.buildActivationFunction(
            activationString)
        self.costFunction = self.buildCostFunction(costString)
        self.outputActivationFunction = self.buildActivationFunction(
            outputActivationString)
        self.regularizationFunction = self.buildRegularizationFunction(regularizationString)
        self.nLayers = 0
        self.channels = self.buildChannels(channelReps)
        self.fullyConnectedLayers = self.buildFullyConnected(fullLayersRep)
        self.tieLayers()

    def buildActivationFunction(self, activationString):
        if activationString.lower() == "rectifier":
            return Rectifier()
        elif activationString.lower() == "softmax":
            return Softmax()
        elif activationString.lower() == "sigmoid":
            return Sigmoid()
        elif activationString.lower() == "identity":
            return Identity()

        raise Exception()

    def buildCostFunction(self, costString):
        if costString.lower() == "loglikelihood":
            return LogLikelihood()
        elif costString.lower() == "quadratic":
            return Quadratic()
        raise Exception()

    def buildRegularizationFunction(self, regularizationString):
        regString = regularizationString.split("-")
        if len(regString) != 2: return NullReg()

        if regString[0].lower() == "l2reg":
            return L2Reg(float(regString[1]))
        else:
            return NullReg()

    def buildChannels(self, channelReps):
        if channelReps == "":
            return None
        channels = []
        idFirstLayer = self.nLayers
        for i in range(len(channelReps)):
            c = Channel(i, self.inputSizeChannels, channelReps[i],
                        self.activationFunction, self.inputDimension, idFirstLayer)
            idFirstLayer += c.getNLayers()
            channels.append(c)

        s = 0
        for c in channels:
            s += c.getNFlattenedOutputs()

        self.inputFullyConnected += s
        self.nLayers = idFirstLayer
        return channels

    def buildFullyConnected(self, fullLayersRep):
        layers = []
        inputSize = self.inputFullyConnected
        idFirstLayer = self.nLayers
        for s in fullLayersRep:
            a = s.split("-")
            if a[0].lower() == "hidden":
                nUnits = int(a[1])
                l = FullyConnectedLayer(inputSize, nUnits,
                                        idFirstLayer, self.activationFunction)
                layers.append(l)
                inputSize = nUnits
                idFirstLayer += 1
            elif a[0].lower() == "output":
                nUnits = int(a[1])
                l = FullyConnectedLayer(inputSize, nUnits, idFirstLayer,
                                        self.outputActivationFunction, True)
                idFirstLayer += 1
                layers.append(l)

        for i in range(len(layers) - 1):
            layers[i].setNextLayer(layers[i + 1])

        for i in range(1, len(layers)):
            layers[i].setPreviousLayer(layers[i - 1])

        self.nLayers = idFirstLayer
        return layers

    def buildInputFullyConnected(self, channelsResults, extraInputs):
        result = []
        for i in range(len(self.channels)):
            currentChannel = channelsResults[i]
            chRs = []
            for l in currentChannel:
                chRs = np.concatenate((chRs, l.flatten()))
            result = np.concatenate((result, chRs))

        result = np.concatenate((result, extraInputs))

        return result

    def tieLayers(self):
        if self.channels is None:
            return
        firstFullyConnectedLayer = self.fullyConnectedLayers[0]
        lastLayers = []
        for c in self.channels:
            lastLayer = c.getLayers()[-1]
            lastLayer.setNextLayer(firstFullyConnectedLayer)
            lastLayers.append(lastLayer)

        firstFullyConnectedLayer.setPreviousLayer(lastLayers)

    def forwardOneSample(self, xs, xs_extra):
        import pdb; pdb.set_trace()  # breakpoint f9fbadce //

        if self.channels is not None:
            results = {}
            for i in range(len(self.channels)):
                c = self.channels[i]
                result = c.forward([xs[i]])
                results[c.id] = result

            inputFullyConnected = self.buildInputFullyConnected(results, xs_extra)
        else:
            inputFullyConnected = xs

        firstFCLayer = self.fullyConnectedLayers[0]
        finalResult = firstFCLayer.forward(inputFullyConnected)
        return finalResult

    def evaluateOneSample(self, xs, xs_extra, desiredOutput, epsilon):
        # actualOutput = self.forwardOneSample(xs, xs_extra)
        # result = int(np.argmax(actualOutput) == desiredOutput)
        # return result
        actualOutput = self.forwardOneSample(xs, xs_extra)
        if self.outputSize == 1:
            return int(np.abs(actualOutput - desiredOutput) <= epsilon)
        else:
            return int(np.argmax(actualOutput) == np.argmax(desiredOutput))



    def backwardOneSample(self, desiredOutput):
        outputLayer = self.fullyConnectedLayers[-1]
        outputLayer.backward(None, desiredOutput, self.costFunction)

    def trainOneSample(self, xs, xs_extra, desiredOutput):
        self.forwardOneSample(xs, xs_extra)
        self.backwardOneSample(desiredOutput)

    def train(self, xTrainChannels, xTrainExtra, yTrain):
        assert len(xTrainChannels) == len(yTrain)
        for i in range(len(xTrainChannels)):
            xTE = [] if xTrainExtra == [] else xTrainExtra[i]
            self.trainOneSample(xTrainChannels[i], xTE, yTrain[i])

    def test(self, xTestChannels, xTestExtra, yTest, epsilon):
        v = []
        for i in range(len(xTestChannels)):
            xTE = [] if xTestExtra == [] else xTestExtra[i]
            if self.evaluateOneSample(xTestChannels[i], xTE, yTest[i], epsilon) == 1:
                v.append(i)
        return len(v)

    def save(self, directory):
        for layer in self:
            layer.save(directory)

    def load(self, directory):
        for layer in self:
            layer.load(directory)

    def error(self, xs, xsExtra, ys):
        N = len(ys)
        totalError = 0

        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            xE = [] if xsExtra == [] else xsExtra[i]
            result = self.forwardOneSample(x, xE)
            miniError = self.costFunction.function(result, y)
            totalError += miniError
        return 1.0*totalError/N

    def updateParameters(self, weightsDict, nSamples):
        self.regularizationFunction.setNSamples(nSamples)
        for layer in self:
            if isinstance(layer, PoolLayer):
                continue
            biases = weightsDict[str(layer.layerId) + "biases"]
            weights = weightsDict[str(layer.layerId) + "weights"]
            layer.updateParameters(biases, weights, self.regularizationFunction)



    def __iter__(self):
        return NetworkIterator(self)
