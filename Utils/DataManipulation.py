import numpy as np

def permute(*arrays):
    result = []
    perm = np.random.permutation(len(arrays[0]))
    for a in arrays:
        if a == []:
            result.append(a)
        else:
            result.append(a[perm])
    return tuple(result)
