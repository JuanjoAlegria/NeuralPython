import numpy as np

class LogLikelihood:
    def function(self, actualOutput, desiredOutput):
        epsilon = 1e-8
        if isinstance(desiredOutput, np.ndarray):
            desiredOutput = np.argmax(desiredOutput)
        return -np.log(actualOutput[desiredOutput]) + epsilon
    def derivative(self,actualOutput, desiredOutput):
        # TODO : fix
        a_y = actualOutput[desiredOutput]
        return -1.0/a_y
