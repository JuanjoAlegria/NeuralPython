# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

class AbstractCostFunction:
    """Clase base para todas las funciones de costo
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def function(self, vector):
        """Calcula la función de costo sobre el vector entregado

        :param vector: vector sobre el cual se quiere calcular la función, suele ser el output de la última capa de la red
        :type vector: numpy.ndarray
        :returns: numpy.ndarray - resultado de calcular función de costo
        """
        pass

    @abstractmethod
    def derivative(self, layerOutput, desiredOutput):
        """Calcula la derivada de la función de costo; se usa en backpropagation

        :param layerOutput: vector sobre el cual se quiere calcular la derivada, es el vector calculado por la red neuronal en la última capa
        :type layerOutput: numpy.ndarray
        :param desiredOutput: output deseado.
        :returns: numpy.ndarray - resultado de calcular la derivada de la función de activación en layerOutput y desiredOutput
        """
        pass
