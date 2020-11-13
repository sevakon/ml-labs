from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple

import numpy as np


def read_dataset(path: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    with open(path, 'r') as file:
        k = int(file.readline())
        n = int(file.readline())
        
        samples = list()
        
        for _ in range(n):
            sample = \
                np.array(file.readline().split(" "), dtype=np.float64)
            
            samples.append(sample)
            
        train = np.array(samples)
        
        m = int(file.readline())
        
        samples = list()
        
        for _ in range(m):
            sample = \
                np.array(file.readline().split(" "), dtype=np.float64)
            
            samples.append(sample)
            
        test = np.array(samples)
        
        if train.shape[0] != n or test.shape[0] != m:
            raise AssertionError("Sizes mismatch")
            
        if train.shape[1] != k + 1 or test.shape[1] != k + 1:
            raise AssertionError("Feature sizes mismatch")
            
        X_train, y_train = train[:, :k], train[:, k]
        X_test, y_test = test[:, :k], test[:, k]

        return X_train, y_train, X_test, y_test

    
def preprocess_features(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(X_train)
    
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    X_train_full = np.ones(
        (X_train_norm.shape[0], X_train_norm.shape[1] + 1), dtype=np.float64)
    X_test_full = np.ones(
        (X_test_norm.shape[0], X_test_norm.shape[1] + 1), dtype=np.float64)
    
    X_train_full[:, :-1] = X_train_norm
    X_test_full[:, :-1] = X_test_norm
    
    return X_train_full, X_test_full
