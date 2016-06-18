# -*- coding: utf-8 -*-
import time, os
import numpy as np
from NeuralPython.Layers.PoolLayer import PoolLayer
from NeuralPython.Utils import DataManipulation

class SimpleTraining:
    def __init__(self):
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
        # Save Parameters
        self.epochSave = None
        saveDir = "../NetworksModels/" + str(self.initTime) + "/"
        self.saveDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                                    saveDir)
        self.bestResultDir = os.path.join(self.saveDir, "BestResult/")
        self.bestResultValidation = float("inf")
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
        self.network = network

    def setData(self, trainData, validationData, testData):
        self.trainData = trainData
        self.testData = testData
        self.validationData = validationData

        self.nTotalSamples[0] = len(trainData[0])
        self.nTotalSamples[1] = len(validationData[0])

        print "Datos cargados"
        print "Número datos de entrenamiento: ", self.nTotalSamples[0]
        print "Número datos de validación: ", len(self.validationData[0])


    def loadData(self, loadFunction):
        trainData, validationData, testData = loadFunction()
        self.setData(trainData, validationData, testData)

    def makeSaveDir(self):
        os.makedirs(self.saveDir)
        os.makedirs(self.bestResultDir)

    def saveNetwork(self):
        if (self.epochSave != 0  and self.currentEpoch % self.epochSave == 0):
            print "Guardando red en directorio ", self.saveDir
            self.network.save(self.saveDir)
        if self.errors[1] < self.bestResultValidation:
            print "Mejor resultado actualizado, guardando red en directorio ", \
                                self.bestResultDir
            self.network.save(self.bestResultDir)
            self.bestResultValidation = self.errors[1]

    def calcError(self):
        self.errors[0] = self.network.error(self.trainData)
        self.errors[1] = self.network.error(self.validationData)

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
        accuracy = self.network.test(self.validationData, self.epsilonError)
        print "Terminada pasada", miniBatchEpoch
        print "Resultados: ", accuracy, "/", len(self.validationData[0])


    def calcDeltas(self):
        weightsDict = {}
        for layer in self.network:
            if isinstance(layer, PoolLayer):
                continue
            biases, weights = layer.calculateParameters()
            weightsDict[str(layer.layerId) + "weights"] = weights
            weightsDict[str(layer.layerId) + "biases"] = biases
        return weightsDict


    def updateWeights(self, weightsDict):
        self.learningSchedule.update(weightsDict, self.miniBatchSize)
        self.network.updateParameters(weightsDict, self.miniBatchSize)

    def stopTraining(self):
        if self.stopCriteria == "maxEpochs":
            return self.currentEpoch >= self.maxEpochs
        elif self.stopCriteria == "thresholdErrorTrain":
            return self.errors[0] <= self.thresholdError
        elif self.stopCriteria == "thresholdErrorValidation":
            return self.errors[1] <= self.thresholdError


    def train(self):
        self.network.regularizationFunction.setEta(self.eta)
        while not self.stopTraining():
            self.currentEpoch += 1
            k = 0
            i = 0
            self.trainData = DataManipulation.permute(*self.trainData)
            while k + self.miniBatchSize <= self.nTotalSamples[0]:
                currentTrainData = [d[k: k + self.miniBatchSize] for d in self.trainData]
                # Entrenamiento
                self.network.train(currentTrainData)
                weightsDict = self.calcDeltas()
                self.updateWeights(weightsDict)

                k += self.miniBatchSize
                i += 1
                self.learningSchedule.updateEpoch()

            self.calcAccuracy(i)
            self.calcError()
            self.saveNetwork()

    def printFinalTime(self):
        currentTime = int(time.time())
        deltaTime = currentTime - self.initTime
        print "Han transcurrido ", deltaTime, " segundos"

    def run(self):
        self.makeSaveDir()
        # self.calcError()
        self.train()
        self.saveNetwork()
        self.printFinalTime()


