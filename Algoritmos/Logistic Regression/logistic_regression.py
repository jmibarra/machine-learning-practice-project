import numpy as np


class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Inicializo los parámetros
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Descenso del gradiente
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias

            # Aplico la función sigmoide
            y_predicted = self._sigmoid(linear_model)

            # Calculo las derivadas parciales en w y en b
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        # Normalizo el resultado de la función sigmoid que da entre 0 y 1 dejando que todo los menores de 0.5 sean 0 y todos los mayores 1
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
