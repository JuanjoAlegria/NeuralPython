class AbstractLayer(object):
    def __init__(self, nInputs, inputSize, layerId, channelId = None):
        self.nInputs = nInputs
        self.inputSize = inputSize
        self.layerId = layerId
        self.channelId = channelId
        self.z = None
        self.input = None
        self.nextLayer = None
        self.previousLayer = None

    def setNextLayer(self, nextLayer):
        self.nextLayer = nextLayer

    def setPreviousLayer(self, previousLayer):
        self.previousLayer = previousLayer

