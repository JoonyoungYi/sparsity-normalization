import torch
import numpy as np
from sklearn.datasets import (fetch_california_housing, load_diabetes,
                               load_boston, load_wine, load_breast_cancer)
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _load_sklearn_dataset(dataset):
    if dataset == 'boston':
        data = load_boston()
    elif dataset == 'diabetes':
        data = load_diabetes()
    elif dataset == 'california':
        data = fetch_california_housing()
    elif dataset == 'wine':
        data = load_wine()
    elif dataset == 'cancer':
        data = load_breast_cancer()
    else:
        raise NotImplementedError(f'Dataset {dataset} not supported.')
    return data.data.astype(np.float32), data.target.astype(np.float32)


def _inject_sparsity(x, missing_rate):
    """각 샘플에 무작위로 missing_rate 비율의 결측값(NaN)을 삽입."""
    if missing_rate <= 0:
        return x
    mask = np.random.random(x.shape) < missing_rate
    x = x.copy()
    x[mask] = np.nan
    return x


def _normalize_input(train_x, test_x, method):
    if method == 'min-max':
        scaler = MinMaxScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
    elif method == 'standardization':
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
    else:
        raise NotImplementedError(f'Normalization {method} not supported.')
    return train_x.astype(np.float32), test_x.astype(np.float32)


def _normalize_output(train_y, test_y):
    y_mean = np.mean(train_y)
    y_std = np.std(train_y)
    train_y = (train_y - y_mean) / y_std
    test_y = (test_y - y_mean) / y_std
    return train_y, test_y


def _sparsity_normalization(train_x, test_x):
    """Sparsity Normalization: 관찰된 피처 수로 정규화."""
    train_c = np.sum(~np.isnan(train_x), axis=1, keepdims=True).astype(np.float32)
    k_prime = np.mean(train_c)
    train_c = np.maximum(train_c, 1.0)

    test_c = np.sum(~np.isnan(test_x), axis=1, keepdims=True).astype(np.float32)
    test_c = np.maximum(test_c, 1.0)

    train_x = train_x * k_prime / train_c
    test_x = test_x * k_prime / test_c
    return train_x, test_x


def _impute(train_x, test_x, method):
    if method == 'zero':
        train_x = np.nan_to_num(train_x, nan=0.0)
        test_x = np.nan_to_num(test_x, nan=0.0)
    elif method == 'mean':
        col_means = np.nanmean(train_x, axis=0)
        for j in range(train_x.shape[1]):
            train_x[np.isnan(train_x[:, j]), j] = col_means[j]
            test_x[np.isnan(test_x[:, j]), j] = col_means[j]
    else:
        raise NotImplementedError(f'Imputation {method} not supported.')
    return train_x, test_x


def _to_tensor(x, y):
    x = torch.from_numpy(x).float().cuda()
    y = torch.from_numpy(y).float().cuda()
    return x, y


class Dataset:
    def __init__(self,
                 dataset,
                 sparsity_normalization=False,
                 imputation='zero',
                 input_normalization='min-max',
                 missing_rate=0.5,
                 seed=1):
        np.random.seed(seed)

        x, y = _load_sklearn_dataset(dataset)
        y = y.reshape(-1, 1)

        perm = np.random.permutation(x.shape[0])
        x, y = x[perm], y[perm]

        split = int(x.shape[0] * 0.8)
        train_x, test_x = x[:split], x[split:]
        train_y, test_y = y[:split], y[split:]

        train_x, test_x = _normalize_input(train_x, test_x, input_normalization)
        train_y, test_y = _normalize_output(train_y, test_y)

        # 결측값 삽입 (SN 실험용)
        train_x = _inject_sparsity(train_x, missing_rate)
        test_x = _inject_sparsity(test_x, missing_rate)

        if sparsity_normalization:
            train_x, test_x = _sparsity_normalization(train_x, test_x)

        train_x, test_x = _impute(train_x, test_x, imputation)

        self.train_x, self.train_y = _to_tensor(train_x, train_y)
        self.test_x, self.test_y = _to_tensor(test_x, test_y)
