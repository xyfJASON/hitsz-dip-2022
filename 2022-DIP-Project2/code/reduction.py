import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class BaseReduction:
    def __init__(self):
        pass

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PCAReduction(BaseReduction):
    def __init__(self):
        super().__init__()
        self.pca = PCA(n_components=0.95)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(X, y)

    def transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)


class LDAReduction(BaseReduction):
    def __init__(self):
        super().__init__()
        self.lda = LDA(solver='eigen', shrinkage='auto')

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.lda.fit_transform(X, y)

    def transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.lda.transform(X)
