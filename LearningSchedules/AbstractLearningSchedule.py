# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

class AbstractLearningSchedule:
	__metaclass__ = ABCMeta

	@abstractmethod
	def update(self, weightsDict, nSamples):
		pass

	@abstractmethod
	def updateEpoch(self):
		pass
