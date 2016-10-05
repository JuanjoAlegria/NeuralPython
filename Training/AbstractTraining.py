# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np
import time, os
from NeuralPython.Layers.PoolLayer import PoolLayer
from NeuralPython.Utils import DataManipulation

class InvalidMiniBatchSizeException(Exception):
    def __init__(self):
        pass

class AbstractTraining:
    __metaclass__ = ABCMeta


    def __init__(self, projectName):
        self.projectName = projectName
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
        self.accuracy = 0
        # Datasets
        self.trainData = [[], [], []]
        self.validationData = [[], [], []]
        self.testData = [[], [], []]
        # Save Parameters
        self.epochSave = None
        saveDir = "../NetworksModels/" + self.projectName + "/" + str(self.initTime) + "/"
        self.saveDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                                    saveDir)
        self.bestResultDir = os.path.join(self.saveDir, "BestResult/")
        self.bestResultValidation = float("inf")
        # Epsilon Error only in regression
        self.epsilonError = 0


    def setTrainingHyperParameters(self, miniBatchSize, eta):
        self.miniBatchSize = miniBatchSize
        self.eta = eta

    def setLoadSaveParameters(self, epochSave, bestResultOn):
        self.epochSave = epochSave
        self.bestResultOn = bestResultOn
        if bestResultOn == "accuracy":
            self.bestResultValidation = 0
        elif bestResultOn == "error":
            self.bestResultValidation = float("inf")

    def setLearningSchedule(self, learningSchedule):
        self.learningSchedule = learningSchedule

    def setEpsilonError(self, epsilonError):
        self.epsilonError = epsilonError

    def setStopCriteria(self, stopCriteria, thresholdError = None, maxEpochs = None):
        self.stopCriteria = stopCriteria
        self.thresholdError = thresholdError
        self.maxEpochs = maxEpochs

    def setNetwork(self, network):
        self.network = network

    def stopTraining(self):
        if self.stopCriteria == "maxEpochs":
            return self.currentEpoch >= self.maxEpochs
        elif self.stopCriteria == "thresholdErrorTrain":
            return self.errors[0] <= self.thresholdError
        elif self.stopCriteria == "thresholdErrorValidation":
            return self.errors[1] <= self.thresholdError

    def isBestResult(self):
        if self.bestResultOn == "accuracy":
            return self.accuracy > self.bestResultValidation
        elif self.bestResultOn == "error":
            return self.errors[1] < self.bestResultValidation

    def updateBestResult(self):
        if self.bestResultOn == "accuracy":
            self.bestResultValidation = self.accuracy
        elif self.bestResultOn == "error":
            self.bestResultValidation = self.errors[1]

    def calcDeltas(self):
        weightsDict = {}
        for layer in self.network:
            if isinstance(layer, PoolLayer):
                continue
            biases, weights = layer.calculateParameters()
            weightsDict[str(layer.layerId) + "weights"] = weights
            weightsDict[str(layer.layerId) + "biases"] = biases
        return weightsDict

    def confussionMatrix(self, bestResults = True):
        if bestResults:
            self.network.load(self.bestResultDir)
        return self.network.confussionMatrix(self.testData)

    def run(self):
        self.makeSaveDir()
        self.calcError()
        self.train()
        self.saveNetwork()
        self.printFinalTime()


    def makeSaveDir(self):
        os.makedirs(self.saveDir)
        os.makedirs(self.bestResultDir)

    def saveNetwork(self):
        if (self.epochSave != 0  and self.currentEpoch % self.epochSave == 0):
            print "Guardando red en directorio ", self.saveDir
            self.network.save(self.saveDir)
        if self.isBestResult():
            print "Mejor resultado actualizado, guardando red en directorio ", \
                                self.bestResultDir
            self.network.save(self.bestResultDir)
            self.updateBestResult()

    def printDataLoaded(self):
        print "Datos cargados"
        print "Número datos de entrenamiento: ", self.nTotalSamples[0]
        print "Número datos de validación: ", self.nTotalSamples[1]

    def printFinalTime(self):
        currentTime = int(time.time())
        deltaTime = currentTime - self.initTime
        print "Han transcurrido ", deltaTime, " segundos"

    def printCurrentError(self):
        if self.currentEpoch == -1:
            print "Calculando errores iniciales, antes de entrenar"
        else:
            print "Epoch ", self.currentEpoch, "finalizada"
        print "Error actual datos de entrenamiento ", self.errors[0]
        print "Error actual datos de validación: ", self.errors[1]

    def printCurrentStep(self, step):
        print "Terminada pasada ", step

    def printCurrentAccuracy(self):
        print "Resultados: ", self.accuracy, "/", \
                  len(self.validationData[0])

    def permuteTrainData(self):
        self.trainData = DataManipulation.permute(*self.trainData)

    def train(self):
        if self.calcMiniBatchStep() > self.nTotalSamples[0]:
            raise InvalidMiniBatchSizeException()
        self.network.regularizationFunction.setEta(self.eta)
        while not self.stopTraining():
            self.currentEpoch += 1
            k = 0
            i = 0
            self.permuteTrainData()
            while k + self.calcMiniBatchStep() <= self.nTotalSamples[0]:
                currentTrainData = self.getCurrentTrainData(k)
                # Entrenamiento
                self.network.train(currentTrainData)
                weightsDict = self.calcDeltas()
                self.updateWeights(weightsDict)

                k += self.calcMiniBatchStep()
                i += 1
                self.learningSchedule.updateEpoch()
                self.printCurrentStep(i)
            self.calcAccuracy()
            self.calcError()
            self.saveNetwork()

    @abstractmethod
    def setData(self, trainData, validationData, testData):
        pass

    @abstractmethod
    def loadData(self, loadFunction):
        pass

    @abstractmethod
    def calcError(self):
        pass

    @abstractmethod
    def calcAccuracy(self):
        pass

    @abstractmethod
    def updateWeights(self, weightsDict):
        pass

    @abstractmethod
    def calcMiniBatchStep(self):
        pass

    @abstractmethod
    def getCurrentTrainData(self, k):
        pass


