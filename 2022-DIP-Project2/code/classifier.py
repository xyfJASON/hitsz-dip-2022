import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import RidgeClassifier as RC


class BaseClassifier:
    def __init__(self):
        pass

    def fit(self, X_train: np.array, y_train: np.array) -> None:
        raise NotImplementedError

    def predict(self, X_test: np.array) -> np.array:
        raise NotImplementedError


class RidgeClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.model = RC()

    def fit(self, X_train: np.array, y_train: np.array) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.array) -> np.array:
        return self.model.predict(X_test)


class KNNClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.model = KNN()

    def fit(self, X_train: np.array, y_train: np.array) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.array) -> np.array:
        return self.model.predict(X_test)


class LRClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.model = LR()

    def fit(self, X_train: np.array, y_train: np.array) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.array) -> np.array:
        return self.model.predict(X_test)


class SVMClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.model = SVC()

    def fit(self, X_train: np.array, y_train: np.array) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.array) -> np.array:
        return self.model.predict(X_test)
