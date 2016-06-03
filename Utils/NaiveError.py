# -*- coding: utf-8 -*-
import numpy as np

def naiveAlgorithm(data):
    xs, xes, ys = data

    eC1 = 0
    eC2 = 0

    outputs = np.empty(0)
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]

        output = x[0,-1]
        outputs = np.append(outputs, output)

        eC1 += (y - output)**2
        eC2 += (y - output)**2 / (y**2)

    eC1 /= (2.0 * len(xs))
    eC2 /= (2.0 * len(xs))

    return outputs, eC1, eC2
