from distance import get_distance_function_from_string
from kernel import get_kernel_function_from_string


def weighted_one_hot_sum(vectors, weights=None):
    if weights is None:
        weights = [1. for _ in range(len(vectors))]
    
    if len(vectors) != len(weights):
        raise AssertionError(
            "Weights and Vectors lengths mismatch")
    
    result = [0. for _ in range(len(vectors[0]))]
    for v_idx, v in enumerate(vectors):
        for idx, el in enumerate(v):
            result[idx] += weights[v_idx] * el
            
    return result


def argmax(vector):
    index, max_val = -1, -1
    for i in range(len(vector)):
        if vector[i] > max_val:
            index, max_val = i, vector[i]
            
    return index


class KNearestNeighbors:
    """ K-Nearest Nighbors for classification 
    targets are one-hot vectors """
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
            
        result_vector = weighted_one_hot_sum(labels)

        return argmax(result_vector)
                
    
    def _weighted_prediction(self, sorted_tuples, max_distance):
        weights = [
            self.kernel_fn(dist, max_distance) 
            for (x, y, dist) in sorted_tuples
        ]
        
        if sum(weights) == 0 or max_distance == 0:
            """No points inside the window"""
            return argmax(weighted_one_hot_sum(self.y))
        
        result_vector = weighted_one_hot_sum(
            [y for (x, y, dist) in sorted_tuples], 
            weights
        )
        
        return argmax(result_vector)
        
        
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