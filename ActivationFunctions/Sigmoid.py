# -*- coding: utf-8 -*-
import numpy as np
from AbstractActivationFunction import AbstractActivationFunction

class Sigmoid(AbstractActivationFunction):
    """Función de activación sigmoide, definida como

    .. math::

        f(x) = \\frac{1}{1 + e^{-x}}

    y con derivada igual a

    .. math::

        f'(x) = f(x) \\times (1 - f(x))
    """
    def function(self, vector):
        """Calcula la versión vectorizada de la función sigmoide
        """
        return 1.0 / (1.0 + np.exp(-vector))

    def derivative(self, layerOutput, desiredOutput = None):
        """Calcula la versión vectorizada de la derivada de la función sigmoide
        """
        return self.function(layerOutput) * (1 - self.function(layerOutput))
