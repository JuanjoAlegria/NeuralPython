# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np

class AbstractNetwork:
	__metaclass__ = ABCMeta

	def train(self, data):
		samples = zip(*data)
		for sample in samples:
			self.forward(*sample[:-1])
			self.backward(sample[-1])

	def getAccuracy(self, data, epsilon):
		s = 0
		samples = zip(*data)
		for sample in samples:
			if self.evaluateOneSample(sample, epsilon) == 1:
				s += 1
		return s

	def evaluateOneSample(self, sample, epsilon):
		actualOutput = self.forward(*sample[:-1], test = True)
		desiredOutput = sample[-1]
		if self.regression:
			return int(np.abs(actualOutput - desiredOutput) <= epsilon)
		else:
			return int(np.argmax(actualOutput) == np.argmax(desiredOutput))

	def confussionMatrix(self, data):
		if self.regression:
		    raise TypeError
		outputs = np.zeros((self.outputSize, self.outputSize))
		samples = zip(*data)
		for sample in samples:
		    actualOutput = np.argmax(self.forward(*sample[:-1], test = True))
		    desiredOutput = np.argmax(sample[-1])
		    outputs[desiredOutput, actualOutput] += 1
		return outputs

	def getError(self, data):
		N = len(data[0])
		totalError = 0
		samples = zip(*data)
		for sample in samples:
		    result = self.forward(*sample[:-1], test = True)
		    miniError = self.costFunction.function(result, sample[-1])
		    totalError += miniError
		return 1.0 * totalError / N

	@abstractmethod
	def forward(self, x, test):
		pass

	@abstractmethod
	def backward(self, y):
		pass

	@abstractmethod
	def updateParameters(self, weightsDict, nSamples):
		pass

	@abstractmethod
	def save(self, directory):
		pass

	@abstractmethod
	def load(self, directory):
		pass

	@abstractmethod
	def __iter__(self):
		pass
