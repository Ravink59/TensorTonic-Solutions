import numpy as np

def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr, steps):

    X = np.array(X)
    y = np.array(y)

    N, D = X.shape

    # initialize parameters
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):

        # Linear model
        z = np.dot(X, w) + b

        # Prediction
        p = _sigmoid(z)

        # Gradients
        dw = (1/N) * np.dot(X.T, (p - y))
        db = (1/N) * np.sum(p - y)

        # Update parameters
        w -= lr * dw
        b -= lr * db

    return (w, b)