# -*- coding: utf-8 -*-
import time, os
import numpy as np
from mpi4py import MPI
from Network import Network
from Layers.PoolLayer import PoolLayer
from LearningSchedules.AdaGrad import AdaGrad
from LearningSchedules.Adam import Adam
from LearningSchedules.SimpleEta import SimpleEta
from Utils import Hyperparameters, DataManipulation

class MPINetwork:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nProcesses = self.comm.Get_size()
        self.network = None
        self.initTime = int(time.time())

        self.channelsRep = ""
        self.fullyConnectedRep = ""
        self.inputDimension = 0
        self.inputSizeChannels = 0
        self.nInputsExtra = 0
        self.activationString = ""
        self.costString = ""
        self.outputActivationString = ""
        self.nChannels = 0
        self.outputSize = 0

        self.miniBatchSize = 0
        self.miniBatchValidationSize = 0
        self.eta = 0
        self.maxError = 0
        self.epsilonError = 0
        self.errorIndex = 0
        self.epochSave = None
        self.learningSchedule = None

        self.errors = np.zeros(2)
        self.nTotalSamples = np.zeros(1)
        self.errors[0] = float('inf') # entrenamiento
        self.errors[1] = float('inf') # validación

        self.trainData = [[], [], []]
        self.validationData = [[], [], []]
        self.testData = [[], [], []]
        self.miniValidationData = None

        self.loadNetwordDir = ""
        saveDir = "NetworksModels/" + str(self.initTime) + "/"
        self.saveDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), saveDir)

    def setNetworkParameters(self, channelsRep, fullyConnectedRep, \
                             inputDimension, inputSizeChannels, nInputsExtra, \
                             activationString, costString, outputActivationString, \
                             regularizationString):
        self.channelsRep = channelsRep
        self.fullyConnectedRep = fullyConnectedRep
        self.inputDimension = inputDimension
        self.inputSizeChannels = inputSizeChannels
        self.nInputsExtra = nInputsExtra
        self.activationString = activationString
        self.costString = costString
        self.outputActivationString = outputActivationString
        self.regularizationString = regularizationString
        self.nChannels = len(channelsRep)
        self.outputSize = int(self.fullyConnectedRep[-1].split("-")[1])


    def setTrainingHyperParameters(self, miniBatchSize, miniBatchValidationSize, \
                                   eta, learningString, maxError, epsilonError, \
                                   monitorValidationError = False):
        self.miniBatchSize = miniBatchSize
        self.miniBatchValidationSize = miniBatchValidationSize
        self.eta = eta
        self.maxError = maxError
        self.epsilonError = epsilonError
        self.errorIndex = 1 if monitorValidationError else 0
        self.learningSchedule = self.buildLearningSchedule(learningString)

    def setLoadSaveParameters(self, epochSave, networkLoadDir):
        self.epochSave = epochSave
        self.networkLoadDir = networkLoadDir

    def setData(self, trainData, validationData, testData):
        if self.rank == 0:
            self.trainData = trainData
            self.totalTestData = testData
            self.nTotalSamples[0] = len(self.trainData[0])

            vData = []
            for i in range(len(validationData)):
                d = validationData[i]
                if d == []:
                    vData.append([])
                else:
                    vData.append(d[:self.miniBatchValidationSize * self.nProcesses])
            self.validationData = vData

        self.comm.Barrier()
        self.comm.Bcast(self.nTotalSamples)

        miniDatas = []
        for d in [(self.trainData, int(self.nTotalSamples[0] / self.nProcesses)), \
                  (self.validationData, self.miniBatchValidationSize)]:
            dataset = d[0]
            batchSize = d[1]
            if self.inputDimension == 1:
                x_shape = (batchSize, self.nChannels, \
                            self.inputSizeChannels[0])
            elif self.inputDimension == 2:
                x_shape = (batchSize, self.nChannels, \
                            self.inputSizeChannels[0], self.inputSizeChannels[1])

            x = np.zeros(x_shape)
            xe = np.zeros((batchSize, self.nInputsExtra))
            y = np.zeros((batchSize, self.outputSize, 1))

            self.comm.Scatter(dataset[0], x, 0)
            if dataset[1] != []:
                self.comm.Scatter(dataset, xe, 0)
            else:
                xe = []
            self.comm.Scatter(dataset[2], y, 0)


            miniDatas.append([x, xe, y])

        self.miniTrainData = miniDatas[0]
        self.miniValidationData = miniDatas[1]

        if self.rank == 0:
            print "Datos cargados"
            print "Número datos de entrenamiento = ", self.nTotalSamples[0]
            print "Número datos de validación: ", len(self.validationData[0])

    def loadData(self, loadFunction):
        if self.rank == 0:
            trainData, validationData, testData = loadFunction()
        else:
            trainData = validationData = testData = None
        self.setData(trainData, validationData, testData)

    def buildNetwork(self):
        if self.rank == 0:
            self.network = Network(self.channelsRep, self.fullyConnectedRep, \
                                   self.inputDimension, self.inputSizeChannels, \
                                   self.nInputsExtra, self.activationString, \
                                   self.costString, self.outputActivationString, \
                                   self.regularizationString)
            if self.loadNetwordDir != "":
                print "Cargando red desde" + self.networkDirectory
                self.network.load(self.loadNetwordDir)

        self.network = self.comm.bcast(self.network, 0)

    def buildLearningSchedule(self, learningString):
        if learningString.lower() == "simpleeta":
            return SimpleEta(self.eta)
        elif learningString.lower() == "adagrad":
            return AdaGrad(self.eta)
        elif learningString.lower().startswith("adagradforget"):
            forgetRate = int(learningString.split("-")[1])
            return AdaGrad(self.eta, forgetRate)
        elif learningString.lower() == "adam":
            return Adam(self.eta)

    def makeSaveDir(self):
        if self.rank == 0:
            os.makedirs(self.saveDir)

    def saveNetwork(self):
        if self.rank == 0:
            print "Guardando red en directorio ", self.saveDir
            self.network.save(self.saveDir)

    def fullEpochError(self, epoch = -1):
        partialErrorsResult = np.zeros(2)
        partialErrorsResult[0] = self.network.error(self.miniTrainData[0], \
                                        self.miniTrainData[1], self.miniTrainData[2])
        partialErrorsResult[1] = self.network.error(self.miniValidationData[0], \
                                self.miniValidationData[1], self.miniValidationData[2])
        self.comm.Reduce(partialErrorsResult, self.errors, op=MPI.SUM)
        self.errors /= self.nProcesses

        if self.rank == 0:
            print "###########################################"
            if epoch == -1:
                print "Calculando errores iniciales, antes de entrenar"
                print "Previous Error IN: ", self.errors[0]
                print "Previous Error OUT: ", self.errors[1]
            else:
                print "Epoch ", epoch, "finalizada"
                print "Current Error IN: ", self.errors[0]
                print "Current Error OUT: ", self.errors[1]


    def miniBatchError(self, miniBatchEpoch):
        partialTestResult = np.zeros(1)
        partialTestResult[0] = self.network.test(self.miniValidationData[0], \
                    self.miniValidationData[1], self.miniValidationData[2], \
                    self.epsilonError)

        totalTestResult = np.zeros(1)
        self.comm.Reduce(partialTestResult, totalTestResult, op=MPI.SUM)

        if self.rank == 0:
            print "Terminada pasada", miniBatchEpoch
            print "Resultados: ", totalTestResult[0], "/", \
                  self.miniBatchValidationSize * self.nProcesses


    def scatterData(self, k = -1):
        batchSize = self.miniBatchSize if k == -1 \
                                        else int(self.nTotalSamples[0] / self.nProcesses)

        if self.inputDimension == 1:
            x_shape = (batchSize, self.nChannels, \
                        self.inputSizeChannels[0])
        elif self.inputDimension == 2:
            x_shape = (batchSize, self.nChannels, \
                        self.inputSizeChannels[0], self.inputSizeChannels[1])
        x_train = np.zeros(x_shape)
        x_extra_train = np.zeros((batchSize, self.nInputsExtra))
        y_train = np.zeros((batchSize, self.outputSize))

        if self.rank == 0:
            trainData = DataManipulation.permute(self.trainData[0], \
                                    self.trainData[1], self.trainData[2])
            if k == -1:
                scatteredX, scatteredXExtra, scatteredY = trainData
            else:
                scatteredX = trainData[0][k: k + \
                                self.miniBatchSize * self.nProcesses]
                scatteredXExtra = trainData[1][k: k + \
                                self.miniBatchSize * self.nProcesses]
                scatteredY = trainData[2][k: k + \
                                self.miniBatchSize * self.nProcesses]
        else: # rank != 0
            scatteredX = scatteredY = None
            scatteredXExtra = []
        self.comm.Scatter(scatteredX, x_train, 0)
        self.comm.Scatter(scatteredY, y_train, 0)
        if scatteredXExtra != []:
            self.comm.Scatter(scatteredXExtra, x_extra_train, 0)
        else:
            x_extra_train = []

        return x_train, x_extra_train, y_train


    def calcDeltas(self):
        # Calcular deltas en cada proceso
        localDict = {}
        for layer in self.network:
            if isinstance(layer, PoolLayer):
                continue
            biases, weights = layer.calculateParameters()
            localDict[str(layer.layerId) + "weights"] = weights
            localDict[str(layer.layerId) + "biases"] = biases

        return localDict


    def updateWeights(self, localDict):
        totalDict = {}
        for key in localDict:
            totalDict[key] = np.zeros(np.shape(localDict[key]))
            self.comm.Allreduce(localDict[key], totalDict[key], MPI.SUM)

        nSamples = self.miniBatchSize * self.nProcesses
        self.learningSchedule.update(totalDict, nSamples)
        self.network.updateParameters(totalDict, nSamples)


    def train(self):
        epoch = 0
        self.network.regularizationFunction.setEta(self.eta)
        while self.errors[self.errorIndex] >= self.maxError:
            k = 0
            i = 0
            while k + self.miniBatchSize * self.nProcesses <= self.nTotalSamples[0]:
                x_current, x_extra_current, y_current = self.scatterData(k)
                # Entrenamiento
                self.network.train(x_current, x_extra_current, y_current)
                localDict = self.calcDeltas()
                self.updateWeights(localDict)
                self.miniBatchError(i)

                k += self.miniBatchSize * self.nProcesses
                i += 1

            self.fullEpochError(epoch)
            epoch += 1
            if self.epochSave != 0  and epoch % self.epochSave == 0:
                self.saveNetwork()

    def printFinalTime(self):
        if self.rank == 0:
            currentTime = int(time.time())
            deltaTime = currentTime - self.initTime
            print "Han transcurrido ", deltaTime, " segundos"

    def run(self):
        self.buildNetwork()
        self.makeSaveDir()
        self.fullEpochError(epoch = -1)
        self.train()
        self.saveNetwork()
        self.printFinalTime()

def buildFromDict(hpDict = None):
    if hpDict is None:
        hpDict = Hyperparameters.buildFromArgs()
    # Network parameters
    channelReps = hpDict['channelReps']
    fullLayersRep = hpDict['fullLayersRep']
    inputDimension = hpDict['inputDimension']
    inputSizeChannels = hpDict['inputSizeChannels']
    nInputsExtra = hpDict['nInputsExtra']
    activationString = hpDict['activationString']
    costString = hpDict['costString']
    outputActivationString = hpDict['outputActivationString']
    regularizationString = hpDict['regularizationString']

    # Training Parameters
    miniBatchSize = hpDict['miniBatchSize']
    miniBatchValidationSize = hpDict['miniBatchValidationSize']
    eta = hpDict['eta']
    learningString = hpDict['learningScheduleString']
    maxError = hpDict['maxError']
    epsilonError = hpDict['epsilonError']
    monitorValidationError = hpDict['monitorValidationError']

    # Load/Save Parameters
    epochSave = hpDict['epochSave']
    loadNetwordDir = hpDict['networkLoadDir']


    mpiNetwork = MPINetwork()
    mpiNetwork.setNetworkParameters(channelReps, fullLayersRep, inputDimension, inputSizeChannels, nInputsExtra, activationString, costString, outputActivationString, regularizationString)
    mpiNetwork.setTrainingHyperParameters(miniBatchSize, miniBatchValidationSize, eta, learningString, maxError, epsilonError, monitorValidationError)
    mpiNetwork.setLoadSaveParameters(epochSave, loadNetwordDir)

    return mpiNetwork
