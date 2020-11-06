import math


def euclidean(x, y):
    return math.sqrt(sum([(i - j) ** 2 for i, j in zip(x, y)]))


def manhattan(x, y):
    return sum([abs(i - j) for i, j in zip(x, y)])


def chebyshev(x, y):
    return max([abs(i - j) for i, j in zip(x, y)])


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


class KNearestNeighbors:
    def __init__(
        self, 
        distance_fn, 
        kernel_fn
    ):
        self.distance_fn = distance_fn
        self.kernel_fn = kernel_fn
    
    def fit(self, X, y):
        if len(X) != len(y):
            raise AssertionError("X and y sizes mismatch")
            
        self.X = X
        self.y = y
        self.size = len(X)
    
    def _sort_by_distance(self, target):
        sorted_tuples = sorted([
            (x, y, self.distance_fn(x, target)) 
            for x, y in zip(self.X, self.y)
        ], key=lambda tup: tup[2])
        
        return sorted_tuples
    
    def _exact_match(self, sorted_tuples):
        labels = []
        for (x, y, dist) in sorted_tuples:
            if dist > 0:
                break
            labels.append(y)

        return sum(labels) / len(labels)
                
    
    def _weighted_prediction(self, sorted_tuples, max_distance):
        if max_distance == 0:
            """No points inside the window"""
            return sum(self.y) / self.size

        numenator = sum([
            y * self.kernel_fn(dist, max_distance) 
            for (x, y, dist) in sorted_tuples
        ])
        denominator = sum([
            self.kernel_fn(dist, max_distance) 
            for (x, y, dist) in sorted_tuples
        ])
        
        if denominator == 0:
            """No points inside the window"""
            return sum(self.y) / self.size
        
        return numenator / denominator
        
        
class KNNFixedWindow(KNearestNeighbors):
    
    def __init__(
        self, 
        window, 
        distance_fn, 
        kernel_fn
    ):
        super(KNNFixedWindow, self).__init__(distance_fn, kernel_fn)
        self.window = window
        
    def predict(self, target):
        sorted_tuples = self._sort_by_distance(target)

        if self.window == 0 and sorted_tuples[0][0] == target:
            """Exact match between points"""
            return self._exact_match(sorted_tuples)

        y_hat = self._weighted_prediction(sorted_tuples, self.window)
        
        return y_hat

    
class KNNVariableWindow(KNearestNeighbors):
    
    def __init__(
        self, 
        k, 
        distance_fn, 
        kernel_fn
    ):
        super(KNNVariableWindow, self).__init__(distance_fn, kernel_fn)
        self.k = k
        
    def predict(self, target):
        sorted_tuples = self._sort_by_distance(target)
        max_distance = sorted_tuples[self.k][2]
        
        if max_distance == 0:
            """Exact match between points"""
            return self._exact_match(sorted_tuples)
        
        y_hat = self._weighted_prediction(sorted_tuples, max_distance)
        
        return y_hat


def get_knn_model(
    distance_name: str,
    kernel_name: str,
    window_type: str,
    window_param: int
):
    if 'fixed' == window_type:
        return KNNFixedWindow(
            window_param, 
            get_distance_function_from_string(distance_name),
            get_kernel_function_from_string(kernel_name),
        )
    
    elif 'variable' == window_type:
        return KNNVariableWindow(
            window_param, 
            get_distance_function_from_string(distance_name),
            get_kernel_function_from_string(kernel_name),
        )
    
    else:
        raise ValueError(f"unknown {window_type} window type")



if __name__ == '__main__':
    n, m = map(int, input().split())
    X, y = [], []

    for _ in range(n):
        values = list(map(int, input().split()))
        X.append(values[:-1])
        y.append(values[-1])

    target = list(map(int, input().split()))
    distance_name = input()
    kernel_name = input()
    window_type = input()
    window_param = int(input())

    model = get_knn_model(
        distance_name, kernel_name, 
        window_type, window_param
    )

    model.fit(X, y)

    print(model.predict(target))

