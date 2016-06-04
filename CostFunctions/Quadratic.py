import numpy as np

class Quadratic:
    def function(self, actualOutput, desiredOutput):
        # if isinstance(desiredOutput, int):
        #     d = desiredOutput
        #     desiredOutput = np.zeros(actualOutput.shape)
        #     desiredOutput[d] = 1
        return 0.5 * np.linalg.norm(desiredOutput - actualOutput)**2

    def derivative(self,actualOutput, desiredOutput):
        return actualOutput - desiredOutput
