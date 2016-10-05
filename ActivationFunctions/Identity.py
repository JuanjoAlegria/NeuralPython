# -*- coding: utf-8 -*-
import numpy as np
from AbstractActivationFunction import AbstractActivationFunction
class Identity(AbstractActivationFunction):
    """
        Función de activación identidad, útil para modelar regresión
    """

    def function(self, vector):
        """
            Ejecuta la función Identidad, es decir, f(x) = x
        """
        return vector

    def derivative(self, layerOutput, desiredOutput = None):
        """
            Calcula la derivada de la función identidad,
            es decir, df(x) / dx = [1, 1, ..., 1]
        """
        return np.ones(layerOutput.shape)
