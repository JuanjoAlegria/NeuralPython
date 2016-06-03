import numpy as np
from scipy.misc import logsumexp

class Softmax:
    def __init__(self):
        pass

    def function2(self, vector):
        s = logsumexp(vector)
        return np.exp(vector - s)

    def function(self, vector):
        v2 = vector - np.max(vector)
        return np.exp(v2) / np.sum(np.exp(v2))

    def derivative(self, actualOutput, desiredOutput):
        index_y = np.argmax(desiredOutput)
        a_y = actualOutput[index_y]
        return -a_y * (actualOutput - desiredOutput)


if __name__ == '__main__':
    s = Softmax()
    vectors = []
    for i in range(10):
        vector = np.random.randint(0, 100, 20)
        vectors.append(vector)

    for v in vectors:
        print v
        r = s.function(v)
        print r
        print "Suma = ", np.sum(r)
        print "Max = ", np.max(r)
        print "Max index = ", np.argmax(r)
        print "#####################"
        r = s.function2(v)
        print r
        print "Suma = ", np.sum(r)
        print "Max = ", np.max(r)
        print "Max index = ", np.argmax(r)
        print "#####################"
