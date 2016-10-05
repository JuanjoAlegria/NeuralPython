# -*- coding: utf-8 -*-
import numpy as np
from AbstractActivationFunction import AbstractActivationFunction

class Sigmoid(AbstractActivationFunction):
    """
        Función de activación sigmoide, definida como 1/(1 + exp(-x))
    """
    def function(self, vector):
        """
            Calcula la versión vectorizada de f(x) = 1/(1 + exp(-x))
        """
        return 1.0 / (1.0 + np.exp(-vector))

    def derivative(self, layerOutput, desiredOutput = None):
        """
            Calcula la versión vectorizada de f'(x) = f(x)*(1 - f(x))
        """
        return self.function(layerOutput) * (1 - self.function(layerOutput))
