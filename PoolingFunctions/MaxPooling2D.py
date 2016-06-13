import numpy as np

class MaxPooling2D:
    def __init__(self, inputSize, step):
        self.inputSize = inputSize
        self.step = step
        self.outputSize = inputSize / step
        self.max = {}

    def down(self, x, nX):
        maxIndexes = np.zeros(self.inputSize)
        result = np.zeros(self.outputSize)
        for i in range(0, self.inputSize[0], self.step):
            for j in range(0, self.inputSize[1], self.step):
                max_ij = np.unravel_index(np.argmax(x[i:i + self.step, j:j + self.step]),
                                            (self.step, self.step))
                max_i = max_ij[0] + i
                max_j = max_ij[1] + j
                maxIndexes[max_i, max_j] = 1
                result[i/self.step,j/self.step] = x[max_i, max_j]
        self.max[nX] = maxIndexes
        return np.array(result).reshape(self.outputSize)


    def up(self, v, nX):
        maxIndexes = self.max[nX]
        v_extended = v.repeat(self.step, axis = 0).repeat(self.step, axis = 1)
        return v_extended*maxIndexes


