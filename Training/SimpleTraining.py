# -*- coding: utf-8 -*-
from AbstractTraining import AbstractTraining

class SimpleTraining(AbstractTraining):

    def setData(self, trainData, validationData, testData):
        self.trainData = trainData
        self.testData = testData
        self.validationData = validationData

        self.nTotalSamples[0] = len(trainData[0])
        self.nTotalSamples[1] = len(validationData[0])
        self.printDataLoaded()

    def loadData(self, loadFunction):
        trainData, validationData, testData = loadFunction()
        self.setData(trainData, validationData, testData)


    def calcError(self):
        self.errors[0] = self.network.getError(self.trainData)
        if len(self.validationData[0]) > 0:
            self.errors[1] = self.network.getError(self.validationData)
        self.printCurrentError()


    def calcAccuracy(self):
        accuracy = self.network.getAccuracy(self.validationData, self.epsilonError)
        self.accuracy = accuracy
        self.printCurrentAccuracy()


    def updateWeights(self, weightsDict):
        self.learningSchedule.update(weightsDict, self.miniBatchSize)
        self.network.updateParameters(weightsDict, self.miniBatchSize)


    def calcMiniBatchStep(self):
        return self.miniBatchSize


    def getCurrentTrainData(self, index):
        currentTrainData = [d[index: index + self.miniBatchSize] for d in self.trainData]
        return currentTrainData


