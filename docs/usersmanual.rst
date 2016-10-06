Manual de Usuario
==================

Contents:

.. toctree::
   :maxdepth: 1


NeuralPython fue construido con la idea de que el usuario necesitase la menor interacción posibe con el código subyacente. De esta forma, se llegó a un diseño que consiste de **constructores**, que toman un archivo de configuración en formato JSON y retornan los objetos correspondientes, listos para ser utilizados. En un uso típico, los únicos tres objetos que el usuario debiese ver son:

* Diccionario de configuración
* Régimen de entrenamiento (para computador normal o para computación distribuida)
* Red Neuronal

Un ejemplo real de uso es el siguiente::

	from NeuralPython.Utils import Builders
	from Utils import LoadData
	import json

	config = json.load(open("Config.json"))
	net = Builders.buildNetwork(config)
	mpiTrain = Builders.buildTraining(config)
	mpiTrain.setNetwork(net)
	mpiTrain.loadData(LoadData.loadFinancialData)
	mpiTrain.run()

En este caso, el usuario debe definir sólo un archivo de configuración y una función de carga de datos (también existe la posibilidad de entregar directamente los datos, pero en el caso de un entrenamiento distribuido se desaconseja esta opción).


Archivo de configuración
========================

Es un diccionario de tipo JSON donde se define la arquitectura de la red, junto a otros hiperparámetros necesarios. A continuación, se entrega una lista exhaustiva de los valores posibles en el archivo de configuración.

* ``projectName``: Nombre del proyecto, cualquier string es válido. En la carpeta NetworksModels/$projectName$/ se guardarán los distintos modelos obtenidos en distintos entrenamientos.

* ``trainingType``: especifica el tipo de entrenamiento que deseamos para la red. Esto generalmente viene definido por los recursos de la máquina donde se hará el entrenamiento (computador simple o cluster). Los valores posibles son:

	* ``mpi``: entrenamiento paralelo/distribuido, siguiendo las convenciones MPI.
	* ``simple``: entrenamiento en un sólo núcleo, para computadores personales.

* ``networkType``: especifica el tipo de red que se desea. Las opciones posibles son:

	* ``feedforwardnet``: red neuronal clásica, también conocida como Multilayer Perceptron (MLP)
	* ``convandffnet``: red neuronal convolucional, compuesta de una primera parte convolucional, seguida por un MLP.


* ``channelsRep``: Representación de los distintos canales que se utilizarán en la parte convolucional de la red. Por ejemplo, si se está procesando una imagen RGB se deben definir tres canales. Cada canal es una lista de strings, donde cada string corresponde a una capa. No es necesario agregar ``channelsRep`` si se trabaja con una ``feedForwardNet``. Las opciones de capa en esta parte son:


	* ``Conv-nFilters-filtersSize``: Capa convolucional, con``nFilters`` el número de filtros a entrenar, y ``filtersSize`` el tamaño de los filtros.


	* ``Pool-poolingFunction-poolingStep``: Capa de pooling, utilizando poolingFunction como función de pooling, y con poolingStep como paso de pooling. Por ahora, la única función de pooling implementada es:


		* Max: MaxPooling, válido tanto para entradas de 1D (vectores), como para entradas 2D (matrices)


* ``feedForwardRep`` : Representación de la parte completamente conexa de la red. Las opciones de capas disponibles son:

	
	* ``Hidden-sizeLayer``: Capa oculta, con ``sizeLayer`` unidades.

	* ``Output-sizeLayer``: Capa de salida, con ``sizeLayer`` unidades.

* ``inputSizeChannels``: Requerido cuando se usa una ``convandffnet``. Se especifica como una lista donde se entrega el tamaño del input de los canales (se asume que todos los canales tendrán un input del mismo tamaño). El largo de la lista puede ser 1 (cuando se entrega un vector como input) o 2 (cuando se entrega una matriz como input)

* ``inputSize``: Requerido cuando se usa una ``feedforwardnet``. Se especifica como un entero, que indica el largo del vector de entrada. 

* ``activationString``: Función de activación a utilizar. Las opciones posibles son las siguientes:
	
	* ``rectifier``: Función de activación rectifier

	* ``sigmoid``: Función de activación sigmoide

	* ``identity``: Función de activación identidad

* ``outputActivationString``: En muchos casos, la función de activación a utilizar en la capa de salida es distinta la función de activación usada en el resto de la red. Las opciones son:

	* ``rectifier``: Función de activación rectifier
	
	* ``sigmoid``: Función de activación sigmoide
	
	* ``identity``: Función de activación identidad
	
	* ``softmax``: Función de activación softmax

* ``costString``: Función de costo a utilizar. Las opciones son:

	* ``quadratic``: Función de costo cuadrática, también conocida como Mean Squared Error (MSE)
	
	* ``loglikelihood``: Función de costo LogLikelihood

* ``regularizationString``: Regularización (se utiliza para reducir la probabilidad de overfitting). Las opciones posibles son:
	
	* ``nullReg``: Sin regularización.
	
	* ``L2Reg-regFactor``: Regularización L2, con factor de regularización ``regFactor`` (debe ser un número)

* ``learningScheduleString``: Algoritmo de optimización a utilizar, es decir, de qué forma se actualizan los pesos durante los entrenamientos. Las opciones posibles son:

	* ``simpleEta``: Descenso del gradiente estocástico (SGD), algoritmo clásico
	
	* ``adagrad``: Algoritmo AdaGrad
	
	* ``adam``: Algoritmo Adam

* ``miniBatchSize``: tamaño de los minibatches a utilizar durante el entrenamiento. Debe ser un entero.

* ``eta``: :math:`\eta` a utilizar en el descenso del gradiente

* ``regression``: Booleano que indica si estamos trabajando con regresión o clasificación

* ``epsilonError``: Sólo necesario si es que ``regression`` es True; indica el radio dentro del cual señalaremos que un resultado predicho es "correcto". Puede ser cualquier número, aunque obviamente, se aconseja que sea un número menor a 1 y mayor a 0.

* ``stopCriteria``: Indica cuál será el criterio para detener el entrenamiento. Las opciones son:

	* ``maxEpochs``: El entrenamiento termina al alcanzar una cierta cantidad de iteraciones. En tal caso, se debe agregar otra entrada al diccionario, llamada nuevamente ``maxEpochs``, y cuyo valor asociado sea un entero.

	* ``thresholdErrorTrain``: El entrenamiento se detiene cuando el error alcanzado dentro de los datos de entrenamiento es menor a cierto umbral. En tal caso, se debe añadir una entrada llamada ``thresholdError``, cuyo valor asociado sea un número (entero o decimal) que indical el umbral de error.

	* ``thresholdErrorValidation``: El entrenamiento se detiene cuando el error alcanzado dentro de los datos de validación es menor a cierto umbral. En tal caso, se debe añadir una entrada llamada ``thresholdError``, cuyo valor asociado sea un número (entero o decimal) que indical el umbral de error.


* ``epochSave``: cada cuántas iteraciones de desea guardar el resultado de la red; debe ser un entero.

* ``bestResultOn``: Indica cuál será el criterio para determinar que cierto modelo es el que obtiene mejor resultado. Las opciones posibles son:

	* ``accuracy``: Cuando la cantidad de resultados clasificados correctamente es mayor (útil en clasificación). Se evalúa con los datos de validación.

	* ``error``: Cuando el valor producido por la función de error sobre todos los inputs es menor (útil en clasificación y regresión). También se evalúa con los datos de validación.

* ``networkLoadDir``: Directorio desde donde se cargará la red. Opcional.


Carga de datos
==============

La carga de datos se puede hacer de dos formas.

* Creando una función que cargue los datos, y luego entregarle esta función como parámetro a un ``traning.loadData(function)``, donde training es un régimen de entrenamiento. Se recomienda esta opción para computación paralelizada, pues así es posible que sólo un nodo se encargue de los datos y los reparta a todo el resto de los nodos a conveniencia.

* Cargando manualmente los datos y entregándoselos a ``traning.setData(training, test, validation)``.


Ejemplo de uso en cluster **Leftraru**
======================================

A continuación se da un ejemplo de uso real de la librería en el cluster Leftraru, administrado por el National Laboratory for High Performance Computing.

Primero se debe crear un archivo .sh, el cual aquí llamaremos ``script.sh``::

	#!/bin/bash
	#SBATCH --job-name=financialTraining
	#SBATCH --partition=slims
	#SBATCH -n 20 # Debe de ser un número múltiplo de 20
	#SBATCH --output=financialTraining_%j.out
	#SBATCH --error=financialTraining_%j.err
	#SBATCH --mail-user=jotaj.8a@gmail.com
	#SBATCH --mail-type=ALL
	module load intel impi
	module load python/2.7.10

srun python ./FinancialMain.py

Por otro lado, el archivo FinancialMain.py es el siguiente::

	from NeuralPython.Utils import Builders
	import json, os
	import numpy as np

	def loadFinancialData():
	    basePath = os.path.dirname(os.path.realpath(__file__))

	    xTrainDir = "../Datos/MondayToMonday_for_Tuesday/MondayToMonday2014_s1.npy"
	    yTrainDir = "../Datos/MondayToMonday_for_Tuesday/Tuesdays2014_s1.npy"
	    xValidationDir = "../Datos/MondayToMonday_for_Tuesday/MondayToMonday2015_s1.npy"
	    yValidationDir = "../Datos/MondayToMonday_for_Tuesday/Tuesdays2015_s1.npy"

	    xTrain = np.load(os.path.join(basePath, xTrainDir))
	    yTrain = np.load(os.path.join(basePath, yTrainDir))

	    xValidation = np.load(os.path.join(basePath, xValidationDir))
	    yValidation = np.load(os.path.join(basePath, yValidationDir))

	    trainData = [xTrain, yTrain]
	    validationData = [xValidation, yValidation]
	    testData = [[], []]

	    return trainData, validationData, testData

	config = json.load(open("Config.json"))
	net = Builders.buildNetwork(config)
	mpiTrain = Builders.buildTraining(config)
	mpiTrain.setNetwork(net)
	mpiTrain.loadData(loadFinancialData)
	mpiTrain.run()

Y el archivo ``Config.json`` es el siguiente::

	{
		"projectName": "financialTraining",
		"regression": true,

		"trainingType": "mpi",
		"networkType": "ConvAndFFNet",
		"channelsRep":
		[
			["Conv-12-24", "Pool-Max-2"]
		],
		"feedForwardRep": ["Hidden-100", "Output-1"], 
		"inputSizeChannels": [141],
		"activationString": "Rectifier",
		"costString": "Quadratic",
		"outputActivationString":"Identity",
		"regularizationString": "L2Reg-10",

		"miniBatchSize": 8,
		"eta": 0.0005,

		"learningScheduleString": "Adam",
		"epsilonError": 0.01,

		"stopCriteria": "thresholdErrorValidation",
		"thresholdError": 0.01,

		"epochSave": 100,
		"bestResultOn": "error",
	}




