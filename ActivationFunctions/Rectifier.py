# -*- coding: utf-8 -*-
import numpy as np
from AbstractActivationFunction import AbstractActivationFunction

class Rectifier(AbstractActivationFunction):
    """
        Función de activación rectificadora, definida como f(x) = max(0,x)
    """
    def function(self, vector):
        """
            Calcula la versión vectorizada de f(x) = max(0,x)
        """
        return np.fmax(vector, np.zeros(len(vector)))

    def derivative(self, layerOutput, desiredOutput = None):
        """
            Calcula la versión vectorizada de f'(x), tal que f'(x) = 0, si x <= 0, y f'(x) = 1, en caso contrario
        """
        def singleDerivative(x):
            if x >= 0: return 1
            else: return 0
        f = np.vectorize(singleDerivative)
        return f(layerOutput)
