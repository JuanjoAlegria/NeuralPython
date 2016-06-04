import numpy as np

class LogLikelihood:
    def function(self, actualOutput, desiredOutput):
        if isinstance(desiredOutput, np.ndarray):
            desiredOutput = np.argmax(desiredOutput)
        return -np.log(actualOutput[desiredOutput])
    def derivative(self,actualOutput, desiredOutput):
        a_y = actualOutput[desiredOutput]
        return -1.0/a_y

