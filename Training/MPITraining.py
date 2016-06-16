# -*- coding: utf-8 -*-
import time, os
import numpy as np
from mpi4py import MPI
from NeuralPython.Layers.PoolLayer import PoolLayer
from NeuralPython.Utils import DataManipulation

class MPITraining:
    def __init__(self):
        # MPI values
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nProcesses = self.comm.Get_size()
        self.initTime = int(time.time())
        # Training Mandatory Parameters
        self.miniBatchSize = 0
        self.eta = 0
        # Objects
        self.network = None
        self.learningSchedule = None
        #Stop Criteria Parameters
        self.stopCriteria = None
        self.thresholdError = None
        self.maxEpochs = None
        # Training init values
        self.currentEpoch = -1
        self.errors = np.zeros(2)
        self.nTotalSamples = np.zeros(2)
        self.errors[0] = float('inf') # entrenamiento
        self.errors[1] = float('inf') # validación
        # Datasets
        self.trainData = [[], [], []]
        self.validationData = [[], [], []]
        self.testData = [[], [], []]
        self.miniTrainData = None
        self.miniValidationData = None
        self.shapes = None
        self.dtypes = None
        # Save Parameters
        self.epochSave = None
        saveDir = "../NetworksModels/" + str(self.initTime) + "/"
        self.saveDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                                    saveDir)
        # Epsilon Error only in regression
        self.epsilonError = 0


    def setTrainingHyperParameters(self, miniBatchSize, eta):
        self.miniBatchSize = miniBatchSize
        self.eta = eta


    def setLoadSaveParameters(self, epochSave):
        self.epochSave = epochSave

    def setLearningSchedule(self, learningSchedule):
        self.learningSchedule = learningSchedule

    def setEpsilonError(self, epsilonError):
        self.epsilonError = epsilonError

    def setStopCriteria(self, stopCriteria, thresholdError = None, maxEpochs = None):
        self.stopCriteria = stopCriteria
        self.thresholdError = thresholdError
        self.maxEpochs = maxEpochs

    def setNetwork(self, network):
        if self.rank == 0:
            self.network = network
        self.network = self.comm.bcast(self.network, 0)

    def setData(self, trainData, validationData, testData):
        if self.rank == 0:
            self.trainData = trainData
            self.testData = testData
            self.validationData = validationData

            self.shapes, self.dtypes = self.getShapesAndDTypes(trainData)

            self.nTotalSamples[0] = len(trainData[0])
            self.nTotalSamples[1] = len(validationData[0])

        self.comm.Barrier()
        self.shapes = self.comm.bcast(self.shapes)
        self.dtypes = self.comm.bcast(self.dtypes)
        self.comm.Bcast(self.nTotalSamples)

        miniDatas = []
        for index, dataset in enumerate([self.trainData, self.validationData]):
            batchSize = self.nTotalSamples[index] / self.nProcesses
            miniData = self.scatter(dataset, batchSize)
            miniDatas.append(miniData)

        self.miniTrainData = miniDatas[0]
        self.miniValidationData = miniDatas[1]

        if self.rank == 0:
            print "Datos cargados"
            print "Número datos de entrenamiento = ", self.nTotalSamples[0]
            print "Número datos de validación: ", len(self.validationData[0])
            print "Minibatch total", self.miniBatchSize * self.nProcesses


    def scatter(self, dataset, batchSize):
        shapes = [np.zeros(len(shape) + 1).astype(int) for shape in self.shapes]
        for index, shape in enumerate(shapes):
            shape[0] = batchSize
            shape[1:] = self.shapes[index]

        recipients = [np.zeros(shapes[i]).astype(self.dtypes[i]) for i in range(len(shapes))]
        for i in range(len(recipients)):
            self.comm.Scatter(dataset[i], recipients[i], 0)
        return recipients

    def permuteTrainData(self):
        if self.rank == 0:
            self.trainData = DataManipulation.permute(*self.trainData)

    def scatterTrainData(self, epoch):
        batchSize = self.miniBatchSize
        dataset = []
        if self.rank == 0:
            dataset = [d[epoch: epoch + self.miniBatchSize * self.nProcesses]
                        for d in self.trainData]
        else:
            dataset = [None] * len(self.shapes)

        result = self.scatter(dataset, batchSize)

        return result

    def loadData(self, loadFunction):
        if self.rank == 0:
            trainData, validationData, testData = loadFunction()
        else:
            trainData = validationData = testData = None
        self.setData(trainData, validationData, testData)


    def makeSaveDir(self):
        if self.rank == 0:
            os.makedirs(self.saveDir)

    def saveNetwork(self):
        if self.rank == 0:
            print "Guardando red en directorio ", self.saveDir
            self.network.save(self.saveDir)

    def calcError(self):
        # if self.currentEpoch % 5 != 0: return

        partialErrorsResult = np.zeros(2)
        partialErrorsResult[0] = self.network.error(self.miniTrainData)
        partialErrorsResult[1] = self.network.error(self.miniValidationData)
        self.comm.Allreduce(partialErrorsResult, self.errors, op=MPI.SUM)
        self.errors /= self.nProcesses

        if self.rank == 0:
            print "###########################################"
            if self.currentEpoch == -1:
                print "Calculando errores iniciales, antes de entrenar"
                print "Previous Error IN: ", self.errors[0]
                print "Previous Error OUT: ", self.errors[1]
            else:
                print "Epoch ", self.currentEpoch, "finalizada"
                print "Current Error IN: ", self.errors[0]
                print "Current Error OUT: ", self.errors[1]


    def calcAccuracy(self, miniBatchEpoch):
        if self.currentEpoch % 10 != 0: return

        partialTestResult = np.zeros(1)
        partialTestResult[0] = self.network.test(self.miniValidationData, \
                                                   self.epsilonError)

        totalTestResult = np.zeros(1)
        self.comm.Reduce(partialTestResult, totalTestResult, op=MPI.SUM)

        if self.rank == 0:
            print "Terminada pasada", miniBatchEpoch
            print "Resultados: ", totalTestResult[0], "/", \
                  len(self.validationData[0])



    def calcDeltas(self):
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
            totalDict[key] /= self.nProcesses

        nSamples = self.miniBatchSize * self.nProcesses
        self.learningSchedule.update(totalDict, nSamples)
        self.network.updateParameters(totalDict, nSamples)

    def stopTraining(self):
        if self.stopCriteria == "maxEpochs":
            return self.currentEpoch > self.maxEpochs
        elif self.stopCriteria == "thresholdErrorTrain":
            return self.errors[0] <= self.thresholdError
        elif self.stopCriteria == "thresholdErrorValidation":
            return self.errors[1] <= self.thresholdError


    def train(self):
        self.network.regularizationFunction.setEta(self.eta)
        while not self.stopTraining():
            k = 0
            i = 0
            self.permuteTrainData()
            while k + self.miniBatchSize * self.nProcesses <= self.nTotalSamples[0]:
                currentTrainData = self.scatterTrainData(k)
                # Entrenamiento
                self.network.train(currentTrainData)
                localDict = self.calcDeltas()
                self.updateWeights(localDict)

                k += self.miniBatchSize * self.nProcesses
                i += 1

            self.calcAccuracy(i)
            self.calcError()
            self.currentEpoch += 1
            if self.epochSave != 0  and self.currentEpoch % self.epochSave == 0:
                self.saveNetwork()

    def printFinalTime(self):
        if self.rank == 0:
            currentTime = int(time.time())
            deltaTime = currentTime - self.initTime
            print "Han transcurrido ", deltaTime, " segundos"

    def getShapesAndDTypes(self, dataset):
        shapes = []
        dtypes = []
        for d in dataset:
            totalShape = d.shape
            if len(totalShape) < 2:
                shapeOneSample = np.array([1])
            else:
                shapeOneSample = totalShape[1:]
            shapes.append(shapeOneSample)
            dtypes.append(d.dtype)
        return shapes, dtypes

    def run(self):
        self.makeSaveDir()
        self.calcError()
        self.train()
        self.saveNetwork()
        self.printFinalTime()


