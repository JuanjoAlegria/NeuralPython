# -*- coding: utf-8 -*-
from abc import ABCMeta

class AbstractLayer:
    """
        Clase base para todas las capas de la red
    """
    __metaclass__ = ABCMeta


    def __init__(self, nInputs, inputSize, layerId, channelId = None):
        """
            Inicializa valores necesarios para la red
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
        """
            Actualiza la siguiente capa de la red
            :param nextLayer: capa siguiente
            :type nextLayer: Layer
        """
        self.nextLayer = nextLayer


    def setPreviousLayer(self, previousLayer):
        """
            Actualiza la capa anterior de la red
            :param previousLayer: capa anterior
            :type previousLayer: Layer
        """
        self.previousLayer = previousLayer

