import numpy as np


def euclidean(x, y):
    ''' Euclidean Distance '''
    return np.sqrt(np.sum((x - y) ** 2))

def manhattan(x, y):
    ''' Manhattan Distance '''
    return np.sum(abs(x - y))

def cosine(x, y):
    ''' Cosine Distance '''
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def chebyshev(x, y):
    ''' Chebyshev Distance '''
    return np.max(abs(x - y))
