# -*- coding: utf-8 -*-
import numpy as np
from mpi4py import MPI
from AbstractTraining import AbstractTraining

class MPITraining(AbstractTraining):
    def __init__(self, projectName):
        super(MPITraining, self).__init__(projectName)
        # MPI values
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nProcesses = self.comm.Get_size()
        self.miniTrainData = None
        self.miniValidationData = None
        self.shapes = None
        self.dtypes = None


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
        self.printDataLoaded()

    def loadData(self, loadFunction):
        if self.rank == 0:
            trainData, validationData, testData = loadFunction()
        else:
            trainData = validationData = testData = None
        self.setData(trainData, validationData, testData)


    def calcError(self):
        partialErrorsResult = np.zeros(2)
        partialErrorsResult[0] = self.network.getError(self.miniTrainData)
        partialErrorsResult[1] = self.network.getError(self.miniValidationData)
        self.comm.Allreduce(partialErrorsResult, self.errors, op=MPI.SUM)
        self.errors /= self.nProcesses
        self.printCurrentError()


    def calcAccuracy(self):
        partialTestResult = np.zeros(1)
        partialTestResult[0] = self.network.getAccuracy(self.miniValidationData, \
                                                        self.epsilonError)

        totalTestResult = np.zeros(1)
        self.comm.Allreduce(partialTestResult, totalTestResult, op=MPI.SUM)
        self.accuracy = totalTestResult[0]
        self.printCurrentAccuracy()


    def updateWeights(self, localDict):
        totalDict = {}
        for key in localDict:
            totalDict[key] = np.zeros(np.shape(localDict[key]))
            self.comm.Allreduce(localDict[key], totalDict[key], MPI.SUM)

        nSamples = self.miniBatchSize * self.nProcesses
        self.learningSchedule.update(totalDict, nSamples)
        self.network.updateParameters(totalDict, nSamples)

    def getCurrentTrainData(self, index):
        batchSize = self.miniBatchSize
        dataset = []
        if self.rank == 0:
            dataset = [d[index: index + self.miniBatchSize * self.nProcesses]
                        for d in self.trainData]
        else:
            dataset = [None] * len(self.shapes)

        result = self.scatter(dataset, batchSize)
        return result

    def calcMiniBatchStep(self):
        return self.miniBatchSize * self.nProcesses

    def scatter(self, dataset, batchSize):
        shapes = [np.zeros(len(shape) + 1).astype(int) for shape in self.shapes]
        for index, shape in enumerate(shapes):
            shape[0] = batchSize
            shape[1:] = self.shapes[index]

        recipients = [np.zeros(shapes[i]).astype(self.dtypes[i]) for i in range(len(shapes))]
        for i in range(len(recipients)):
            self.comm.Scatter(dataset[i], recipients[i], 0)
        return recipients


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


    def makeSaveDir(self):
        if self.rank == 0:
            super(MPITraining, self).makeSaveDir()

    def saveNetwork(self):
        if self.rank == 0:
            super(MPITraining, self).saveNetwork()

    def confussionMatrix(self, bestResults = True):
        if self.rank == 0:
            return super(MPITraining, self).confussionMatrix(bestResults)

    def printFinalTime(self):
        if self.rank == 0:
            super(MPITraining, self).printFinalTime()

    def printDataLoaded(self):
        if self.rank == 0:
            super(MPITraining, self).printDataLoaded()

    def printCurrentError(self):
        if self.rank == 0:
            super(MPITraining, self).printCurrentError()

    def printCurrentAccuracy(self):
        if self.rank == 0:
            super(MPITraining, self).printCurrentAccuracy()

    def printCurrentStep(self, step):
        if self.rank == 0:
            super(MPITraining, self).printCurrentStep(step)

    def permuteTrainData(self):
        if self.rank == 0:
            super(MPITraining, self).permuteTrainData()
