import math


def uniform(dist, window):
    u = dist / window
    return 0.5 if u < 1 else 0


def triangular(dist, window):
    u = dist / window
    return 1 - u if u < 1 else 0


def epanechnikov(dist, window):
    u = dist / window
    return 3 * (1 - u ** 2) / 4 if u < 1 else 0


def quartic(dist, window):
    u = dist / window
    return 15 * (1 - u ** 2) ** 2 / 16 if u < 1 else 0


def triweight(dist, window):
    u = dist / window
    return 35 * (1 - u ** 2) ** 3 / 32 if u < 1 else 0


def tricube(dist, window):
    u = dist / window
    return 70 * (1 - (abs(u)) ** 3) ** 3 / 81 if u < 1 else 0


def gaussian(dist, window):
    u = dist / window
    return (1 / (math.sqrt(2 * math.pi))) * math.exp((-1 / 2) * (u ** 2))


def cosine(dist, window):
    u = dist / window
    return (math.pi / 4) * math.cos(math.pi * u / 2) if u < 1 else 0
        

def logistic(dist, window):
    u = dist / window
    return 1 / (math.exp(u) + 2 + math.exp(-u))
    
    
def sigmoid(dist, window):
    u = dist / window
    return (2 / math.pi) / (math.exp(u) + math.exp(-u))


def get_kernel_function_from_string(name):
    mapping = {
        "uniform": uniform,
        "triangular": triangular,
        "epanechnikov": epanechnikov,
        "quartic": quartic,
        "triweight": triweight,
        "tricube": tricube,
        "gaussian": gaussian,
        "cosine": cosine,
        "logistic": logistic,
        "sigmoid": sigmoid,
    }
    
    if name not in mapping:
        raise ValueError(
            f"{name} is not supported as a kernel function")
        
    return mapping[name]
