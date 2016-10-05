# -*- coding: utf-8 -*-
import numpy as np
from AbstractCostFunction import AbstractCostFunction
class LogLikelihood(AbstractCostFunction):
    """
        Función de costo log-likelihood
    """
    def function(self, layerOutput, desiredOutput):
        """
            Calcula la función de costo log-likelihood
        """
        epsilon = 1e-8
        if isinstance(desiredOutput, np.ndarray):
            desiredOutput = np.argmax(desiredOutput)
        return -np.log(layerOutput[desiredOutput] + epsilon)
    def derivative(self,layerOutput, desiredOutput):
        """
            Calcula la derivada de la función de costo log-likelihood
        """
        a_y = layerOutput[desiredOutput]
        return -1.0/a_y
