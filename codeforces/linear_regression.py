import numpy as np
import time
import copy

#np.random.seed(134213)

def smape(X, y, w):
    y_pred = X @ w
    n = X.shape[0]
    res = .0

    for y_true, y_hat in zip(y, y_pred):
        res += abs(y_hat - y_true) / (abs(y_hat) + abs(y_true))

    res /= n
    return res


def normalize(X):
    n = X.shape[0]
    k = X.shape[1]

    coeffs = np.zeros(k)

    for j in range(k):
        coeffs[j] = np.max(np.absolute(X[:, j]))

        if coeffs[j] == 0.:
            coeffs[j] = 1.

        X[:, j] /= coeffs[j]

    return coeffs


def stochastic_gradient(X_i, y_i, weights, lambda_reg):
    y_hat = X_i @ weights
    grad = 2 * ((y_hat - y_i) * X_i + lambda_reg * weights)

    return grad


def gradient_smape(X, y, weights):
    avg_grad = np.zeros(X.shape[1])

    for X_i, y_i in zip(X, y):

        y_hat = X_i @ weights

        t = y_hat * y_i
        num = X_i * (abs(t) + t)
        denom = abs(y_hat) * (abs(y_hat) + abs(y_i)) ** 2

        g = np.sign(y_hat - y_i) * num / denom

        avg_grad += g

    avg_grad /= X.shape[0]
    return avg_grad


def sgd(X_train, y_train, lambd=0.0, learning_rate=0.01, t=1.1, w=None):
    """ Stochastic Gradient Descent of Linear Regression """
    n = X_train.shape[0]
    k = X_train.shape[1]

    # Uniform initilization
    weights = np.random.uniform(low=-1/(2 * n), high=1/(2 * n), size=k) if w is None else w

    start_time = time.process_time()

    while time.process_time() - start_time < t:
        sample_idx = np.random.randint(n)

        y_hat = X_train[sample_idx] @ weights

        weights -= learning_rate * stochastic_gradient(
            X_train[sample_idx], y_train[sample_idx], weights, lambd)

    return weights


def gd(X_train, y_train, learning_rate=0.01, t=1.1, w=None):
    n = X_train.shape[0]
    k = X_train.shape[1]

    # Uniform initilization or cont training
    weights = np.random.uniform(low=-1/(2 * n), high=1/(2 * n), size=k) if w is None else w

    start_time = time.process_time()

    while time.process_time() - start_time < t:
        g = gradient_smape(X_train, y_train, weights)

        weights -= learning_rate * g

    return weights


def fit_least_squares(X, y, lambd=0.0):
    inv = np.linalg.inv(X.T @ X + lambd * np.eye(X.shape[1]))
    pinv = inv @ X.T
    weights = pinv @ y
    return weights


if __name__ == '__main__':
    n, m = map(int, input().split())

    X = np.zeros((n, m + 1))
    y = np.zeros(n)

    for i in range(n):
        s = list(map(int, input().split()))

        X[i, :] = s[:-1] + [1.]
        y[i] = s[-1]

    X_old = copy.deepcopy(X)

    coeffs = normalize(X)

    try:
        w = fit_least_squares(X, y, 0.0) / coeffs
    except:
        w1 = gd(X, y, 1.5e7, 1.1) / coeffs
        w2 = gd(X, y, 1e7, 2.0) / coeffs

        if smape(X_old, y, w1) <= smape(X_old, y, w2):
            w = w1
        else:
            w = w2

    print(*w)

