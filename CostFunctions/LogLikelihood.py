# -*- coding: utf-8 -*-
import numpy as np
from AbstractCostFunction import AbstractCostFunction
class LogLikelihood(AbstractCostFunction):
    """Función de costo log-likelihood, definida como

    .. math::

        C_x(a,y) = -\\text{ln}a_y

    Donde :math:`x` es un input de la red, :math:`a` es el vector de salida calculado por la red e :math:`y` es la clase real de :math:`x`. Es decir, LogLikelihood sólo se aplica en casos de clasificación

    La derivada se calcula como:

    .. math::

        \\frac{dC}{da_y} = -\\frac{1}{a_y}
    """
    def function(self, layerOutput, desiredOutput):
        """Calcula la función de costo log-likelihood
        """
        epsilon = 1e-8
        if isinstance(desiredOutput, np.ndarray):
            desiredOutput = np.argmax(desiredOutput)
        return -np.log(layerOutput[desiredOutput] + epsilon)
    def derivative(self,layerOutput, desiredOutput):
        """Calcula la derivada de la función de costo log-likelihood
        """
        a_y = layerOutput[desiredOutput]
        return -1.0/a_y
