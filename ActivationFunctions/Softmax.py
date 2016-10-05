# -*- coding: utf-8 -*-
import numpy as np
from AbstractActivationFunction import AbstractActivationFunction

class Softmax(AbstractActivationFunction):
    """
        Función de activación softmax, permite interpretar el vector de salida como una distribución de probabilidad
    """
    def function(self, vector):
        """
            Calcula f(x_i) = e**x_i / sum(vector), para todo x_i en el vector
        """
        v = vector - np.max(vector)
        return np.exp(v) / np.sum(np.exp(v))

    def derivative(self, layerOutput, desiredOutput):
        """
            Calcula la derivada de la función softmax
        """
        index_y = np.argmax(desiredOutput)
        a_y = layerOutput[index_y]
        return -a_y * (layerOutput - desiredOutput)
