import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        # X es un numpy nd array con la cantidad de filas es el número de muestras y las columnas son el número de features
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # init mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c == y]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            # Nùmero de muestras con esta etiqueta dividido por el numero total de muestras, que tan seguido ocurre c
            self.priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        # Debo obtener el maximo de la probabilidad de cada uno de los elementos
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(
                np.log(self._probabilityDensity(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        # Con todos las probabiidades posteriores calculadas obtenemos la clase con mayot probabilidad usando el argmax de numpy
        return self._classes[np.argmax(posteriors)]

    # Realizo la formula de la densidad de probabilidad con la distribución de Gauss
    def _probabilityDensity(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
