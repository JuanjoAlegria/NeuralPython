import numpy as np

class ConvFFNetIterator:
    def __init__(self, net):
        self.network = net
        self.currentIterator = self.network.convNet.__iter__()
        self.inChannels = True

    def next(self):
        if self.inChannels:
            try:
                return self.currentIterator.next()
            except StopIteration:
                self.currentIterator = self.network.ffNet.__iter__()
                self.inChannels = False
                return self.currentIterator.next()

        else: return self.currentIterator.next()


class ConvAndFullyConnectedNet:
    def __init__(self, convNet, ffNet, costFunction, \
                 regularizationFunction, regression):
        self.convNet = convNet
        self.ffNet = ffNet
        self.costFunction = costFunction
        self.regularizationFunction = regularizationFunction
        self.regression = regression


    def train(self, data):
        samples = zip(*data)
        for sample in samples:
            self.forward(*sample[:-1])
            self.backward(sample[-1])


    def forward(self, x, xe = [], test = False):
        convResults = self.convNet.forward(x, test)
        inputFF = self.buildInputFeedForward(convResults, xe)
        finalResult = self.ffNet.forward(inputFF, test)
        return finalResult

    def backward(self, y):
        delta = self.ffNet.backward(y)
        deltaConv = self.buildDeltaConv(delta)
        self.convNet.backward(deltaConv)

    def buildInputFeedForward(self, convResults, extraInputs):
        result = []
        channels = self.convNet.channels
        for i in range(len(channels)):
            currentChannel = convResults[i]
            chRs = []
            for l in currentChannel:
                chRs = np.concatenate((chRs, l.flatten()))
            result = np.concatenate((result, chRs))

        if extraInputs != []:
            result = np.concatenate((result, extraInputs))
        return result

    def buildDeltaConv(self, dNext):
        firstFFLayer = self.ffNet.layers[0]
        wNext = firstFFLayer.getWeights()
        reshapedDNext = np.reshape(dNext, (len(dNext), 1))
        delta = np.dot(wNext.transpose(), reshapedDNext)
        result = {}
        startIndex = 0
        for c in self.convNet.channels:
            outputSize = c.getOutputSize()
            result[c.id] = delta[startIndex: startIndex + outputSize]
            startIndex += outputSize
        return result


    def evaluateOneSample(self, sample, epsilon):
        actualOutput = self.forward(*sample[:-1], test = True)
        desiredOutput = sample[-1]
        if self.regression:
            return int(np.abs(actualOutput - desiredOutput) <= epsilon)
        else:
            return int(np.argmax(actualOutput) == np.argmax(desiredOutput))



    def test(self, data, epsilon):
        s = 0
        samples = zip(*data)
        for sample in samples:
            if self.evaluateOneSample(sample, epsilon) == 1:
                s += 1
        return s

    def save(self, directory):
        self.convNet.save(directory)
        self.ffNet.save(directory)

    def load(self, directory):
        self.convNet.load(directory)
        self.ffNet.load(directory)

    def error(self, data):
        N = len(data[0])
        totalError = 0
        samples = zip(*data)
        for sample in samples:
            result = self.forward(*sample[:-1], test = True)
            miniError = self.costFunction.function(result, sample[-1])
            totalError += miniError
        return 1.0 * totalError / N


    def updateParameters(self, weightsDict, nSamples):
        self.convNet.updateParameters(weightsDict, nSamples)
        self.ffNet.updateParameters(weightsDict, nSamples)

    def getInputShape(self):
        self.convNet.getInputShape()

    def getOutputShape(self):
        self.ffNet.getOutputShape()

    def __iter__(self):
        return ConvFFNetIterator(self)


