import numpy as np
from sklearn.datasets import load_boston

def _normalize_input(x, input_normalization):
    if input_normalization == 'none':
        pass
    elif input_normalization == 'min-0-max-1':
        # feature-wise normalization min~max = 0~1
        feature_min = np.min(x, axis=0)
        feature_max = np.max(x, axis=0)
        for j in range(x.shape[1]):
            x[:, j] = (x[:, j] - feature_min[j]) / (
                feature_max[j] - feature_min[j])
    elif input_normalization == 'min--1-max-1':
        # feature-wise normalization min~max = -1~1
        feature_min = np.min(x, axis=0)
        feature_max = np.max(x, axis=0)
        for j in range(x.shape[1]):
            x[:, j] = (x[:, j] - feature_min[j]) / (
                feature_max[j] - feature_min[j]) * 2 - 1
    elif input_normalization == 'standardization':
        # feature-wise standardization
        feature_mean = np.mean(x, axis=0)
        feature_std = np.std(x, axis=0)
        for j in range(x.shape[1]):
            x[:, j] = (x[:, j] - feature_mean[j]) / feature_std[j]
    else:
        raise NotImplementedError()
    return x

def init_boston(input_normalization, split_ratio):
    # get boston dataset from sklearn
    boston = load_boston()

    x = boston['data']   # (506, 13)
    y = boston['target'] # (506, )

    # input normalization
    x = _normalize_input(x, input_normalization)

    # random split
    perm = np.random.permutation(x.shape[0])
    x = x[perm]
    y = y[perm]

    train_x = x[:int(x.shape[0] * split_ratio)]
    test_x = x[int(x.shape[0] * split_ratio):]

    train_y = y[:int(x.shape[0] * split_ratio)]
    test_y = y[int(x.shape[0] * split_ratio):]

    return train_x, train_y, test_x, test_y
