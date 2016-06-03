import numpy as np

class LogLikelihood:
    def function(self, actualOutput, desiredOutput):
        return -desiredOutput * np.log(actualOutput)

    def derivative(self,actualOutput, desiredOutput):
        a_y = actualOutput[desiredOutput]
        return -1.0/a_y

