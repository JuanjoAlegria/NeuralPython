# -*- coding: utf-8 -*-
import time, os, sys
import numpy as np
from mpi4py import MPI
from Network import Network
from Layers.PoolLayer import PoolLayer
from Utils import Hyperparameters, ProcessData
# import plot


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nProcesses = comm.Get_size()
nTotalSamples = np.zeros(1)
errorIn = np.zeros(1)
errorIn[0] = float('inf')


hpDict = Hyperparameters.buildFromArgs()

#hpDict['regHorizon'] /= hpDict['dataInterval']
hpDict['predHorizon'] /= hpDict['dataInterval']
hpDict['yRanges'] = ProcessData.fixIntervals(hpDict['yRanges'])
# hpDict['fullLayersRep'][-1] = \
#             hpDict['fullLayersRep'][-1].format(len(hpDict['yRanges']))
hpDict['nProcesses'] = nProcesses

nChannels = len(hpDict['channelReps'])


# hiperparámetros necesitados en este archivo
miniBatchSize = hpDict['miniBatchSize']
miniBatchTestSize = hpDict['miniBatchTestSize']
nInputsExtra = hpDict['nInputsExtra']
regHorizon = hpDict['regHorizon']
predHorizon = hpDict['predHorizon']
eta = hpDict['eta']
maxError = hpDict['maxError']

if rank == 0:
    network = Network.buildFromDict(hpDict)

    if hpDict['networkDirectory'] != "":
        print "Cargando red desde" + hpDict['networkDirectory']
        network.load(hpDict['networkDirectory'])

    if hpDict["xTrain"] != "" and hpDict["yTrain"] != "":
        xTrainTotal = np.load(hpDict["xTrain"])
        yTrainTotal = np.load(hpDict["yTrain"])
    else:
        _, xTrainTotal, yTrainTotal = ProcessData.loadFullWeeks(hpDict['year1Train'], hpDict['month1Train'], hpDict['year2Train'], hpDict['month2Train'], hpDict['dataInterval'])

    if hpDict["xTest"] != "" and hpDict["yTest"] != "":
        xTestTotal = np.load(hpDict["xTest"])
        yTestTotal = np.load(hpDict["yTest"])
    else:
        _, xTestTotal, yTestTotal = ProcessData.loadFullWeeks(hpDict['year1Test'], hpDict['month1Test'], hpDict['year2Test'], hpDict['month2Test'], hpDict['dataInterval'])

    xTestTotal = xTestTotal[:hpDict['miniBatchTestSize'] * nProcesses]
    yTestTotal = yTestTotal[:hpDict['miniBatchTestSize'] * nProcesses]
    xExtraTrainTotal = []
    xExtraTestTotal = []
    nTotalSamples[0] = len(xTrainTotal)

    print "Datos cargados, n datos = ", nTotalSamples
    print "# datos de prueba: ", len(xTestTotal)


else:
    network = None
    scatteredX = scatteredY = None
    xTestTotal = yTestTotal = None
    xExtraTestTotal = []
    scatteredXExtra = []


x_test = np.zeros((miniBatchTestSize, nChannels, regHorizon))
if xExtraTestTotal != []:
    x_extra_test = np.zeros((miniBatchTestSize, nInputsExtra))
else:
    x_extra_test = []
y_test = np.zeros((miniBatchTestSize))

comm.Barrier()
comm.Bcast(nTotalSamples)
network = comm.bcast(network, 0)
comm.Scatter(xTestTotal, x_test, 0)
if xExtraTestTotal != []:
    comm.Scatter(xExtraTestTotal, x_extra_test, 0)
comm.Scatter(yTestTotal, y_test, 0)


if rank == 0:
    print "###########################################"
    print "Calculando errores iniciales, antes de entrenar"
    prevErrorIn = network.error(xTrainTotal, xExtraTrainTotal, yTrainTotal)
    print "Previous Error IN: ", prevErrorIn
    prevErrorOut = network.error(xTestTotal, xExtraTestTotal, yTestTotal)
    print "Previous Error OUT: ", prevErrorOut

historyDict = {}
epoch = 0
while errorIn[0] >= maxError:
    if rank == 0:
        xTrainTotal, xExtraTrainTotal, yTrainTotal = ProcessData.permute(
            xTrainTotal, xExtraTrainTotal, yTrainTotal)
    k = 0
    i = 0

    while k + miniBatchSize * nProcesses <= nTotalSamples[0]:
        x_train = np.zeros((miniBatchSize, nChannels, regHorizon))
        x_extra_train = np.zeros((miniBatchSize, nInputsExtra))
        y_train = np.zeros((miniBatchSize))

        if rank == 0:
            scatteredX = xTrainTotal[k: k + miniBatchSize * nProcesses]
            scatteredXExtra = xExtraTrainTotal[
                k: k + miniBatchSize * nProcesses]
            scatteredY = yTrainTotal[k: k + miniBatchSize * nProcesses]
        comm.Scatter(scatteredX, x_train, 0)
        if scatteredXExtra != []:
            comm.Scatter(scatteredXExtra, x_extra_train, 0)
        comm.Scatter(scatteredY, y_train, 0)


        # Entrenamiento
        network.train(x_train, x_extra_train, y_train)
        localDict = {}
        # Calcular deltas en cada proceso
        for layer in network:
            if isinstance(layer, PoolLayer):
                continue
            biases, weights = layer.calculateParameters()
            localDict[str(layer.layerId) + "weights"] = weights
            localDict[str(layer.layerId) + "biases"] = biases

        # Sumar deltas de cada proceso y promediar
        totalDict = {}
        for key in localDict:
            totalDict[key] = np.zeros(np.shape(localDict[key]))
            comm.Allreduce(localDict[key], totalDict[key], MPI.SUM)
            if key not in historyDict:
                historyDict[key] = totalDict[key]**2
            else:
                historyDict[key] += totalDict[key]**2

            totalDict[key] = eta * (totalDict[key] / (np.sqrt(historyDict[key] + 1e-8)))

        # Actualizar en cada proceso los pesos y biases, usando los valores
        # promediados
        for layer in network:
            if isinstance(layer, PoolLayer):
                continue
            biases = totalDict[str(layer.layerId) + "biases"]
            weights = totalDict[str(layer.layerId) + "weights"]
            layer.updateParameters(biases, weights)

        # Test
        partialTestResult = np.zeros(1)
        partialTestResult[0] = network.test(x_test, x_extra_test, y_test, 0.25)
        totalTestResult = np.zeros(1)
        comm.Reduce(partialTestResult, totalTestResult, op=MPI.SUM)
        if rank == 0:
            print "Terminada pasada", i
            print "Resultados: ", totalTestResult[0], "/", miniBatchTestSize * nProcesses

        k += miniBatchSize * nProcesses
        i += 1

    if rank == 0:
        print "###########################################"
        print "Epoch ", epoch, "finalizada"
        errorIn[0] = network.error(xTrainTotal, xExtraTrainTotal, yTrainTotal)
        print "Current Error IN: ", errorIn
        errorOut = network.error(xTestTotal, xExtraTestTotal, yTestTotal)
        print "Current Error OUT: ", errorOut

    comm.barrier()
    errorIn = comm.bcast(errorIn, 0)
    epoch += 1


if rank == 0:
    saveDir = "NetworksModels/" + str(int(time.time())) + "/"
    saveDir = os.path.join(sys.path[0], saveDir)
    os.makedirs(saveDir)

    print "Guardando red en directorio ", saveDir
    network.save(saveDir)

