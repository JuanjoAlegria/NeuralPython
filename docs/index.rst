.. NeuralPython documentation master file, created by
   sphinx-quickstart on Tue Oct  4 02:57:00 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentación de NeuralPython
========================================

Contents:

.. toctree::
   :maxdepth: 2

   usersmanual
   modules

Introducción
=============
**NeuralPython** es una librería de redes neuronales implementada en Python 2.7, como parte de la práctica profesional realizada en el Centro de Modelamiento Matemático *(CMM)*, de la Facultad de Ciencias Físicas y Matemáticas de la Universidad de Chile. Esta librería fue desarrollada en base a los siguientes objetivos:

* **Compatible con computadores de alto rendimiento con computación paralelizada**: dado que el *CMM* dispone de un cluster donde se realizaron la mayor parte de los experimentos, fue necesario desarrollar esta implementación adhoc de redes neuronales, para así aprovechar al máximo los recursos disponibles.
* **Facilidad de uso**: debía ser fácil de utilizar y configurar; sin necesidad de *hardcodear* parámetros y con una interfaz simple
* **Extensibilidad**: dado que continuamente se requería probar nuevos algoritmos, se hizo necesario construir la libería de tal forma que añadir nuevas componentes fuera lo más simple posible.

Si bien no está optimizada para uso en GPU's, se intentó que su rendimiento fuera óptimo de todas maneras en un computador personal.

