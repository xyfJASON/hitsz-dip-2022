import argparse
import os
from typing import Tuple, List
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from extractor import LBPExtractor, GaborExtractor, HOGExtractor
from reduction import PCAReduction, LDAReduction
from classifier import KNNClassifier, LRClassifier, SVMClassifier, RidgeClassifier


def get_dataset(root='../dataset') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(os.path.join(root, 'train_labels.txt'), 'r') as f:
        train_labels = f.readlines()
    with open(os.path.join(root, 'test_labels.txt'), 'r') as f:
        test_labels = f.readlines()
    train_labels = [line.strip().split(' ') for line in train_labels]
    test_labels = [line.strip().split(' ') for line in test_labels]
    train_labels = {a[0]: int(a[1]) for a in train_labels}
    test_labels = {a[0]: int(a[1]) for a in test_labels}

    X_train, y_train, X_test, y_test = [], [], [], []
    for file in os.listdir(os.path.join(root, 'train')):
        img = Image.open(os.path.join(root, 'train', file))
        img = np.array(img)  # noqa
        X_train.append(img)
        y_train.append(train_labels[file])
    for file in os.listdir(os.path.join(root, 'test')):
        img = Image.open(os.path.join(root, 'test', file))
        img = np.array(img)  # noqa
        X_test.append(img)
        y_test.append(test_labels[file])
    X_train = np.stack(X_train, axis=0)
    y_train = np.array(y_train)
    X_test = np.stack(X_test, axis=0)
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test


def feature_extraction(methods: List[str], X: np.ndarray) -> np.ndarray:
    feature = []
    for method in methods:
        if method == 'lbp':
            extractor = LBPExtractor()
        elif method == 'gabor':
            extractor = GaborExtractor()
        elif method == 'hog':
            extractor = HOGExtractor()
        else:
            raise ValueError
        feature.append(extractor.extract(X))
    feature = np.concatenate(feature, axis=1)
    return feature


def feature_reduction(method: str or None,
                      feature_train: np.ndarray, y_train: np.ndarray,
                      feature_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if method is None:
        return feature_train, feature_test
    elif method == 'pca':
        model = PCAReduction()
    elif method == 'lda':
        model = LDAReduction()
    else:
        raise ValueError
    feature_train = model.fit_transform(feature_train, y_train)
    feature_test = model.transform(feature_test, y_test)
    return feature_train, feature_test


def classify(method: str,
             X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    if method == 'knn':
        model = KNNClassifier()
    elif method == 'lr':
        model = LRClassifier()
    elif method == 'svm':
        model = SVMClassifier()
    elif method == 'rc':
        model = RidgeClassifier()
    else:
        raise ValueError
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    return acc, f1


def main():
    # ================== argparse ================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('--extractor', nargs='+', type=str, choices=['lbp', 'gabor', 'hog'],
                        help='choose feature extractors. Options: lbp, gabor, hog')
    parser.add_argument('--reduction', type=str, default=None, choices=['pca', 'lda'],
                        help='choose a feature reduction method. Options: pca, lda')
    parser.add_argument('--classifier', type=str, choices=['knn', 'lr', 'svm', 'rc'],
                        help='choose a classifier. Options: knn, lr, svm, rc')
    args = parser.parse_args()
    print(f'Extractor: {args.extractor}')
    print(f'Reduction: {args.reduction}')
    print(f'Classifier: {args.classifier}')

    # ================== get dataset ================== #
    print('==> Getting dataset...')
    X_train, y_train, X_test, y_test = get_dataset()
    print(f'data sizes: {X_train.shape}, {y_train.shape}, {X_test.shape}, {y_test.shape}')

    # ================== feature extraction ================== #
    print('==> Extracting feature...')
    feature_train = feature_extraction(args.extractor, X_train)
    feature_test = feature_extraction(args.extractor, X_test)
    print(f'feature sizes: {feature_train.shape}, {feature_test.shape}')

    # ================== feature reduction ================== #
    print('==> Reducing feature...')
    feature_train, feature_test = feature_reduction(args.reduction, feature_train, y_train, feature_test, y_test)
    print(f'reduced feature sizes: {feature_train.shape}, {feature_test.shape}')

    # ================== feature reduction ================== #
    print('==> Classifying...')
    acc, f1 = classify(args.classifier, feature_train, y_train, feature_test, y_test)
    print(f'acc: {acc}\nmacro f1: {f1}')


if __name__ == '__main__':
    main()
