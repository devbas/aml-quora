import numpy as np
def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))