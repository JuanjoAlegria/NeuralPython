# -*- coding: utf-8 -*-
import numpy as np
from AbstractActivationFunction import AbstractActivationFunction
class Identity(AbstractActivationFunction):
    """
        Función de activación identidad, útil para modelar regresión. Esta función se define como

        .. math::
            f(x) = x

        y su derivada se calcula como

        .. math::

            f'(x) = 1
    """

    def function(self, vector):
        """Calcula la función identidad sobre vector
        """
        return vector

    def derivative(self, layerOutput, desiredOutput = None):
        """Calcula la derivada de la función identidad sobre layerOutput
        """

        return np.ones(layerOutput.shape)
