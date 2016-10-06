# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

class AbstractLayer:
    """Clase base para todas las capas de la red
    """
    __metaclass__ = ABCMeta


    def __init__(self, nInputs, inputSize, layerId, channelId = None):
        """Inicializa valores necesarios para la red

        :param nInputs: número de entradas que tendrá la red
        :type nInputs: int
        :param inputSize: tamaño de cada entrada de la red, puede tener más de una dimensión
        :type inputSize: numpy.ndarray
        :param layerId: id de la capa
        :type layerId: int
        :param channelId: opcional, id del canal donde se encuentra la capa (si es que es multicanal)
        :type channelId: int
        """
        self.nInputs = nInputs
        self.inputSize = inputSize
        self.layerId = layerId
        self.channelId = channelId
        self.z = None
        self.input = None
        self.nextLayer = None
        self.previousLayer = None

    def setNextLayer(self, nextLayer):
        """Actualiza la siguiente capa de la red

        :param nextLayer: capa siguiente
        :type nextLayer: Layer
        """
        self.nextLayer = nextLayer


    def setPreviousLayer(self, previousLayer):
        """Actualiza la capa anterior de la red

        :param previousLayer: capa anterior
        :type previousLayer: Layer
        """
        self.previousLayer = previousLayer

    @abstractmethod
    def forward(self, x, test):
        """Ejecuta una pasada hacia adelante en la capa, tomando como input el arreglo x. Además, en caso de estar utilizando la red en etapa de predicción, y no entrenamiento, se utiliza el boolean test para indicar que no es necesario realizar ciertos cálculos necesarios a la hora de entrenar

        :param x: Input para la capa
        :type x: numpy.ndarray
        :param test: Indica si se está en regimen de entrenamiento (false), o de predicción (true)
        :type test: boolean
        """
        pass

    @abstractmethod
    def backward(self, dNext, desiredOutput = None, costFunction = None):
        """Calcula una pasada hacia atrás en el entrenamiento de la red, utilizando el algoritmo de backpropagation.

        :param dNext: deltas calculado por la red siguiente
        :type dNext: numpy.ndarray
        :param desiredOutput: Sólo en el caso de estar en la última capa de la red, ya que en esa situación se está en el caso baso de backpropagation, y para calcular el delta correspondiente es necesario conocer el resultado real que se esperaba la red predijera
        :type desiredOutput: numpy.ndarray
        :param costFunction: Se utiliza sólo en la última capa de la red, corresponde a la función de costo a utilizar para calcular el delta correspondiente
        :type costFunction: CostFunction
        """
        pass

    def save(self, directory):
        """Guarda los parámetros aprendidos por la capa en el directorio señalado, en formato numpy

        :param directory: directorio donde se deben guardar los parámetros
        :type directory: string
        """
        pass

    def load(self, directory):
        """Carga parámetros de red desde directorio

        :param directory: directorio desde donde se deben cargar los parámetros
        :type directory: string
        """
        pass

    def updateParameters(self, biases, weights):
        """Actualiza parámetros

        :param biases: nuevos biases
        :type biases: numpy.ndarray
        :param weights: nuevos pesos
        :type weights: numpy.ndarray
        """
        pass
