# -*- coding: utf-8 -*-
import numpy as np
from AbstractCostFunction import AbstractCostFunction
class Quadratic(AbstractCostFunction):
    """Función de costo cuadrática (también conocida como Mean Squared Error), se calcula como

    .. math::

        C(a, y) = \\frac{1}{2n} \sum_{x} ||y - a|| ^2

    donde :math:`n` es la cantidad de vectores, :math:`x` son todos los vectores de entrada para la red, :math:`y` es el valor observado que acompaña a :math:`x`, y :math:`a` es el valor calculado por la red.

    A pesar de que esta función trabaja sobre todos los vectores de entrada, es claro que es posible descomponerla en una función que se concentre en un sólo valor :math:`x`, y luego promediar. Esto se calcula como

    .. math::

        C_x(a, y) = \\frac{1}{2} ||y - a|| ^2

    donde :math:`x` es un valor fijo para :math:`a` e :math:`y`

    Además, la derivada se calcula como

    .. math::

        \\frac{dC}{da_j} = a_j - y_j

    """
    def function(self, layerOutput, desiredOutput):
        """Calcula la función de costo cuadrática
        """
        return 0.5 * np.linalg.norm(desiredOutput - layerOutput)**2

    def derivative(self,layerOutput, desiredOutput):
        """Calcula la derivada de la función de costo cuadrática
        """
        return layerOutput - desiredOutput
