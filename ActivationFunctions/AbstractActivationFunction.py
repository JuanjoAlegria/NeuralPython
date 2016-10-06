# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

class AbstractActivationFunction:
    """Clase base para todas las funciones de activación
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def function(self, vector):
        """Calcula la función de activación sobre el vector entregado

        :param vector: vector sobre el cual se quiere calcular la función
        :type vector: numpy.ndarray
        :returns: numpy.ndarray - resultado de calcular función de activación
        """
        pass

    @abstractmethod
    def derivative(self, layerOutput, desiredOutput = None):
        """Calcula la derivada de la función de activación; se usa en backpropagation

        :param layerOutput: vector sobre el cual se quiere calcular la derivada, generalmente es el vector calculado por la red neuronal en la capa correspondiente
        :type layerOutput: numpy.ndarray
        :param desiredOutput: opcional, se utiliza sólo en la últipa capa y sólo para la función de activación softmax (el cual necesita de este valor para calcular correctamente la derivada). Corresponde al output esperado de la red.
        :type desiredOutput: numpy.ndarray
        :returns: numpy.ndarray - resultado de calcular la derivada de la función de activación en layerOutput
        """
        pass
