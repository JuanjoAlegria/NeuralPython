# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

class AbstractPoolingFunction:
    __metaclass__ = ABCMeta

    @abstractmethod
    def down(self, x, nX, test):
        pass

    @abstractmethod
    def up(self, x, nX):
        pass
