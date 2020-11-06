import math


def euclidean(x, y):
    return math.sqrt(sum([(i - j) ** 2 for i, j in zip(x, y)]))


def manhattan(x, y):
    return sum([abs(i - j) for i, j in zip(x, y)])


def chebyshev(x, y):
    return max([abs(i - j) for i, j in zip(x, y)])


def get_distance_function_from_string(name):
    mapping = {
        "euclidean": euclidean,
        "manhattan": manhattan,
        "chebyshev": chebyshev,
    }
    
    if name not in mapping:
        raise ValueError(
            f"{name} is not supported as a distnace function")
        
    return mapping[name]
