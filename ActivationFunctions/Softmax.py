# -*- coding: utf-8 -*-
import numpy as np
from AbstractActivationFunction import AbstractActivationFunction

class Softmax(AbstractActivationFunction):
    """Función de activación softmax, permite interpretar el vector de salida como una distribución de probabilidad. Se calcula como

    .. math::

        f(x_i) = \\frac{e^{x_i}}{\sum_{j = 0}^{n - 1} e^{x_j}}, \\forall i \\in \\{0,1,...,n-1\\}

    y con derivada igual a

    .. math::

        \\frac{df(x_i)}{x_j} =
             \\begin{cases}
               f(x_i)(1 - f(x_i)) &\\quad\\text{if } i = j\\\\
               -f(x_i)f(x_j) &\\quad\\text{if } i \\neq j \\
             \\end{cases}

    Notamos que en este caso la función de activación depende de todos los valores del vector, a diferencia de otras funciones como la sigmoide, donde cada salida :math:`f(x_i)` depende sólo del punto :math:`x_i`
    """
    def function(self, vector):
        """
            Calcula la función de activación softmax sobre vector
        """
        v = vector - np.max(vector)
        return np.exp(v) / np.sum(np.exp(v))

    def derivative(self, layerOutput, desiredOutput):
        """
            Calcula la derivada de la función de activación softmax
        """
        index_y = np.argmax(desiredOutput)
        a_y = layerOutput[index_y]
        return -a_y * (layerOutput - desiredOutput)
