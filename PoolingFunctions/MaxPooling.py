import numpy as np

class MaxPooling:
    def __init__(self, inputSize, step):
        self.inputSize = inputSize
        self.step = step
        self.max = {}

    def down(self, x, nX, test = False):
        maxIndexes = []
        result = []
        for i in range(0, self.inputSize[0], self.step):
            max_i = np.argmax(x[i:i + self.step])
            max_i = max_i + i
            maxIndexes.append(max_i)
            result.append(x[max_i])
        if not test:
            self.max[nX] = maxIndexes
        return np.array(result)

    def up(self, v, nX):
        maxIndexes = self.max[nX]
        r = np.zeros(self.inputSize)
        for i in range(len(v)):
            r[maxIndexes[i]] = v[i]
        return r

