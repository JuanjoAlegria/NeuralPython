# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
from NeuralPython.Layers.AbstractLayer import AbstractLayer

class FilterLayer(AbstractLayer):
    def __init__(self, nInputs, inputSize, inputDimension, layerId, channelId, \
                 nFilters, filtersSize, activationFunction):
        # arriba: todas las capas, abajo: específicos para esta capa
        super(FilterLayer, self).__init__(nInputs, inputSize, layerId, channelId)
        self.nFilters = nFilters
        self.filtersSize = filtersSize
        self.activationFunction = activationFunction
        self.dimension = inputDimension
        if self.dimension == 1:
            filterShape = (self.nInputs, self.nFilters, self.filtersSize)
        elif self.dimension == 2:
            filterShape = (self.nInputs, self.nFilters, self.filtersSize, self.filtersSize)
        self.filters = np.random.normal(0, 1.0 / self.inputSize.prod(), filterShape)
        self.biases = np.random.randn(nFilters)
        self.deltaBiases = np.zeros(self.biases.shape)
        self.deltaFilters = np.zeros(self.filters.shape)

    def forward(self, x):
        zetas = []
        result = []
        for j in range(self.nFilters):
            s = None
            for i in range(self.nInputs):
                f = self.filters[i][j]
                v = x[i]
                conv = signal.fftconvolve(v, f, 'valid')
                if s is None:
                    s = np.zeros(conv.shape)
                s += conv
            z = self.biases[j] + s
            a = self.activationFunction.function(z)
            zetas.append(z)
            result.append(a)

        self.input = x
        self.z = zetas
        return self.nextLayer.forward(result)

    def backward(self, dNext):
        deltas = []
        # calcular deltas
        for i in range(self.nFilters):
            delta_i = self.activationFunction.derivative(self.z[i]) * dNext[i]
            deltas.append(delta_i)

        # calcular delta biases
        deltaBiases = []
        for delta in deltas:
            deltaBiases.append(np.sum(delta))

        # calcular delta filtros
        deltaFilters = np.zeros(self.filters.shape)
        for j in range(self.nFilters):
            rDelta = self.flipArray(deltas[j])
            for i in range(self.nInputs):
                deltaFilters[i][j] += self.flipArray(signal.fftconvolve(self.input[i], rDelta, 'valid'))

        self.deltaFilters += deltaFilters
        self.deltaBiases += deltaBiases

        if self.previousLayer != None:
            self.previousLayer.backward(deltas)

    def getFilters(self):
        return self.filters

    def getParameters(self):
        return self.biases, self.filters

    def getOutputSize(self):
        return self.inputSize - self.filtersSize + 1

    def getNOutputs(self):
        return self.nFilters

    def calculateParameters(self):
        return self.deltaBiases, self.deltaFilters

    def updateParameters(self, biasesDelta, filtersDelta, regularization):
        self.biases -= biasesDelta
        self.filters -= filtersDelta + regularization.weightsDerivation(self.filters)
        self.deltaFilters = np.zeros(np.shape(self.filters))
        self.deltaBiases = np.zeros(np.shape(self.biases))

    def save(self, directory):
        baseFilename = directory + "filterLayer" + str(self.layerId)
        np.save(open(baseFilename + "_biases.npy", "w"), self.biases)
        np.save(open(baseFilename + "_filters.npy", "w"), self.filters)

    def load(self, directory):
        biasesFile = directory + "filterLayer" + str(self.layerId) + "_biases.npy"
        filtersFile = directory + "filterLayer" + str(self.layerId) + "_filters.npy"

        self.biases = np.load(open(biasesFile))
        self.filters = np.load(open(filtersFile))

    def flipArray(self, array):
        if self.dimension == 1:
            return np.flipud(array)
        elif self.dimension == 2:
            return np.fliplr(np.flipud(array))






