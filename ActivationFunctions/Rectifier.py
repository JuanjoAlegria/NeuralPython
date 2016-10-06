# -*- coding: utf-8 -*-
import numpy as np
from AbstractActivationFunction import AbstractActivationFunction

class Rectifier(AbstractActivationFunction):
    """Función de activación rectificadora, definida como

    .. math::
        f(x) = max(0,x)

    y con derivada igual a

    .. math::

        f'(x) =
             \\begin{cases}
               1 &\\quad\\text{if } x > 0\\\\
               0 &\\quad\\text{if } x \\leq 0 \\
             \\end{cases}

    """
    def function(self, vector):
        """Calcula la versión vectorizada de la función rectificadora
        """
        return np.fmax(vector, np.zeros(len(vector)))

    def derivative(self, layerOutput, desiredOutput = None):
        """Calcula la versión vectorizada de la derivada de la función rectificadora
        """
        def singleDerivative(x):
            if x >= 0: return 1
            else: return 0
        f = np.vectorize(singleDerivative)
        return f(layerOutput)
