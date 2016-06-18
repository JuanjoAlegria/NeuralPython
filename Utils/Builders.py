import numpy as np
import os
from NeuralPython.ActivationFunctions.Identity import Identity
from NeuralPython.ActivationFunctions.Rectifier import Rectifier
from NeuralPython.ActivationFunctions.Sigmoid import Sigmoid
from NeuralPython.ActivationFunctions.Softmax import Softmax
from NeuralPython.CostFunctions.LogLikelihood import LogLikelihood
from NeuralPython.CostFunctions.Quadratic import Quadratic
from NeuralPython.Layers.FilterLayer import FilterLayer
from NeuralPython.Layers.FullyConnectedLayer import FullyConnectedLayer
from NeuralPython.Layers.PoolLayer import PoolLayer
from NeuralPython.Layers.Channel import Channel
from NeuralPython.LearningSchedules.Adam import Adam
from NeuralPython.LearningSchedules.AdaGrad import AdaGrad
from NeuralPython.LearningSchedules.SimpleEta import SimpleEta
from NeuralPython.Networks.ConvAndFullyConnectedNet import ConvAndFullyConnectedNet
from NeuralPython.Networks.ConvNet import ConvNet
from NeuralPython.Networks.FeedForwardNet import FeedForwardNet
from NeuralPython.RegularizationFunctions.L2Reg import L2Reg
from NeuralPython.RegularizationFunctions.NullReg import NullReg
from NeuralPython.Training.MPITraining import MPITraining
from NeuralPython.Training.SimpleTraining import SimpleTraining
from NeuralPython.Utils import Hyperparameters


def buildTraining(d = None):
    if d is None:
        d = Hyperparameters.buildFromArgs()

    # Mandatory Parameters
    trainType = d['trainingType']
    if trainType.lower() == "mpi":
        train = MPITraining()
    elif trainType.lower() == "simple":
        train = SimpleTraining()

    miniBatchSize = d['miniBatchSize']
    eta = d['eta']
    learningString = d['learningScheduleString']
    epochSave = d['epochSave']

    learningSchedule = buildLearningSchedule(learningString, eta)

    train.setTrainingHyperParameters(miniBatchSize, eta)
    train.setLearningSchedule(learningSchedule)
    train.setLoadSaveParameters(epochSave)

    stopCriteria = d['stopCriteria']

    if stopCriteria.lower() == "maxepochs":
        maxEpochs = d['maxEpochs']
        train.setStopCriteria(stopCriteria, maxEpochs = maxEpochs)
    elif stopCriteria.lower() in ["thresholderrortrain", "thresholderrorvalidation"]:
        thresholdError = d['thresholdError']
        train.setStopCriteria(stopCriteria, thresholdError = thresholdError)

    if 'regression' in d:
        epsilonError = d['epsilonError']
        train.setEpsilonError(epsilonError)

    return train




def buildNetwork(d):
    networkType = d['networkType'].lower()
    activationString = d['activationString']
    outputActivationString = d['outputActivationString']
    regularizationString = d['regularizationString']
    costString = d['costString']

    if 'regression' in d:
        regression = d['regression']
    else:
        regression = False

    activationFunction = buildActivationFunction(activationString)
    regularizationFunction = buildRegularizationFunction(regularizationString)
    costFunction = buildCostFunction(costString)
    if outputActivationString == activationString:
        outputActivationFunction = activationFunction
    else:
        outputActivationFunction = buildActivationFunction(outputActivationString)


    if networkType == "convnet":
        channelsRep = d['channelsRep']
        inputSizeChannels = np.array(d['inputSizeChannels'])
        net = buildConvNetwork(channelsRep, inputSizeChannels,\
                               activationFunction, regularizationFunction, 0)

    elif networkType == "feedforwardnet":
        ffRep = d['feedForwardRep']
        inputSize = d['inputSize']
        net = buildFFNetwork(ffRep, inputSize, activationFunction, \
                             outputActivationFunction, costFunction, \
                             regularizationFunction, regression, 0)

    elif networkType == "convandffnet":
        channelsRep = d['channelsRep']
        ffRep = d['feedForwardRep']
        inputSizeChannels = np.array(d['inputSizeChannels'])
        nInputsExtra = d['nInputsExtra'] if 'nInputsExtra' in d \
                                            else 0
        net = buildConvFFNet(channelsRep, ffRep, inputSizeChannels, nInputsExtra, \
                        activationFunction, costFunction, outputActivationFunction, \
                        regularizationFunction, regression)

    if "networkLoadDir" in d and d["networkLoadDir"] != "":
        basePath = os.path.dirname(os.path.realpath(__file__))
        networksModelsDir = "../NetworksModels/"
        path = os.path.join(basePath, networksModelsDir)
        path = os.path.join(path, d["networkLoadDir"])
        net.load(path)
        print "Cargando red desde ", path

    return net



def buildConvNetwork(channelsRep, inputSizeChannels, \
                     activationFunction, regularizationFunction, idFirstLayer):

    channels = buildChannels(channelsRep, inputSizeChannels, \
                                          activationFunction, idFirstLayer)
    convNet = ConvNet(channels, regularizationFunction)
    return convNet

def buildFFNetwork(feedForwardRep, inputSize, activationFunction, \
                   outputActivationFunction, costFunction,\
                   regularizationFunction, regression, idFirstLayer):
    ffLayers = buildLayersFeedForward(feedForwardRep, \
                            inputSize, activationFunction, \
                            outputActivationFunction, idFirstLayer)
    ffNet = FeedForwardNet(ffLayers, costFunction, \
                            regularizationFunction, regression)
    return ffNet

def buildConvFFNet(channelsRep, feedForwardRep, inputSizeChannels, nInputsExtra,
                 activationFunction, costFunction, outputActivationFunction, \
                 regularizationFunction, regression):

    convNet = buildConvNetwork(channelsRep, inputSizeChannels, activationFunction, regularizationFunction, 0)

    nLayers = convNet.getNLayers()
    inputFeedForward = convNet.getOutputSize() + nInputsExtra
    ffNet = buildFFNetwork(feedForwardRep, inputFeedForward, activationFunction, outputActivationFunction, costFunction, regularizationFunction, regression, nLayers)

    return ConvAndFullyConnectedNet(convNet, ffNet, costFunction, \
                                    regularizationFunction, regression)

def buildChannels(channelReps, inputSize, activationFunction, \
                  idFirstLayer = 0):
    channels = []
    idFirstLayer = idFirstLayer
    for i in range(len(channelReps)):
        layers = buildLayersChannel(channelReps[i], inputSize.copy(), \
                                    activationFunction, i, idFirstLayer)
        c = Channel(i, layers)
        idFirstLayer += c.getNLayers()
        channels.append(c)

    return channels

def buildLayersChannel(stringRep, inputSize, activationFunction, \
                       channelId, idFirstLayer = 0):
        layers = []
        nInputs = 1
        inputDimension = len(inputSize)
        layerId = idFirstLayer
        for s in stringRep:
            a = s.split("-")
            if a[0] == "Conv":
                nFilters = int(a[1])
                filtersSize = int(a[2])
                l = FilterLayer(nInputs, inputSize.copy(), inputDimension, \
                                 layerId, channelId, nFilters, filtersSize, \
                                 activationFunction)
                nInputs = nFilters
                inputSize = inputSize - filtersSize + 1
                layerId += 1
                layers.append(l)

            if a[0] == "Pool":
                fPool = a[1]
                stepPool = int(a[2])
                l = PoolLayer(nInputs, inputSize.copy(), inputDimension, \
                              layerId, channelId, fPool, stepPool)
                inputSize /= stepPool
                layerId += 1
                layers.append(l)
        return layers

def buildLayersFeedForward(layersRep, inputSize, activationFunction, \
                            outputActivationFunction, idFirstLayer):
    layers = []
    inputSize = inputSize
    idFirstLayer = idFirstLayer
    for s in layersRep:
        a = s.split("-")
        if a[0].lower() == "hidden":
            nUnits = int(a[1])
            l = FullyConnectedLayer(inputSize, nUnits,
                                    idFirstLayer, activationFunction)
            layers.append(l)
            inputSize = nUnits
            idFirstLayer += 1
        elif a[0].lower() == "output":
            nUnits = int(a[1])
            l = FullyConnectedLayer(inputSize, nUnits, idFirstLayer,
                                    outputActivationFunction, True)
            idFirstLayer += 1
            layers.append(l)
    return layers


def buildActivationFunction(activationString):
    if activationString.lower() == "rectifier":
        return Rectifier()
    elif activationString.lower() == "softmax":
        return Softmax()
    elif activationString.lower() == "sigmoid":
        return Sigmoid()
    elif activationString.lower() == "identity":
        return Identity()

    raise Exception()

def buildCostFunction(costString):
    if costString.lower() == "loglikelihood":
        return LogLikelihood()
    elif costString.lower() == "quadratic":
        return Quadratic()
    raise Exception()

def buildRegularizationFunction(regularizationString):
    regString = regularizationString.split("-")
    if len(regString) != 2: return NullReg()

    if regString[0].lower() == "l2reg":
        return L2Reg(float(regString[1]))
    else:
        return NullReg()

def buildLearningSchedule(learningString, eta):
    if learningString.lower() == "simpleeta":
        return SimpleEta(eta)
    elif learningString.lower() == "adagrad":
        return AdaGrad(eta)
    elif learningString.lower().startswith("adagradforget"):
        forgetRate = int(learningString.split("-")[1])
        return AdaGrad(eta, forgetRate)
    elif learningString.lower() == "adam":
        return Adam(eta)

