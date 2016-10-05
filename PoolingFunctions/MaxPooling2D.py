import numpy as np
from AbstractPoolingFunction import AbstractPoolingFunction
class MaxPooling2D(AbstractPoolingFunction):
    def __init__(self, inputSize, step):
        self.inputSize = inputSize
        self.step = step
        self.outputSize = inputSize / step
        self.max = {}
        self.__INDEXES__ = np.arange(inputSize.prod()).reshape(inputSize)


    def down(self, x, nX, test):
        sz = x.itemsize
        h,w = x.shape
        shape = (h/self.step, w/self.step, self.step, self.step)
        strides = sz*np.array([w*self.step,self.step,w,1])

        blocks = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        blocks = blocks.reshape(h * w / (self.step * self.step), self.step*self.step)

        rel_argmax = blocks.argmax(axis = 1)
        abs_argmax = np.zeros(h*w / (self.step*self.step), dtype = int)

        for index in range(len(rel_argmax)):
            relative_pos = rel_argmax[index]
            rel_i = relative_pos / self.step
            rel_j = relative_pos % self.step
            i = (index / (w / self.step)) * self.step + rel_i
            j = (index % (w / self.step)) * self.step + rel_j
            abs_argmax[index] = i*w + j

        result = x.take(abs_argmax).reshape(h/self.step, w/self.step)

        if not test:
            abs_argmax = abs_argmax.reshape(h/self.step,w/self.step)  \
                            .repeat(self.step, axis = 0).repeat(self.step, axis = 1)

            self.max[nX] = (abs_argmax == self.__INDEXES__).astype(int)
        return result

    def up(self, v, nX):
        maxIndexes = self.max[nX]
        v_extended = v.repeat(self.step, axis = 0).repeat(self.step, axis = 1)
        return v_extended*maxIndexes


