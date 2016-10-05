# -*- coding: utf-8 -*-
import numpy as np
from AbstractCostFunction import AbstractCostFunction
class Quadratic(AbstractCostFunction):
    """
        Función de costo cuadrática (también conocida como Mean Squared Error)
    """
    def function(self, layerOutput, desiredOutput):
        """
            Calcula la función de costo cuadrática
        """
        return 0.5 * np.linalg.norm(desiredOutput - layerOutput)**2

    def derivative(self,layerOutput, desiredOutput):
        """
            Calcula la derivada de la función de costo cuadrática
        """
        return layerOutput - desiredOutput
