import numpy as np

class Quadratic:
    def function(self, actualOutput, desiredOutput):
        return 0.5 * (desiredOutput - actualOutput)**2

    def derivative(self,actualOutput, desiredOutput):
        return actualOutput - desiredOutput
