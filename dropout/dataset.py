import numpy as np
from sklearn.datasets import load_boston
import torch


def _normalize_input(x, input_normalization):
    if input_normalization == 'none':
        pass
    elif input_normalization == 'min-max':
        # feature-wise normalization min~max = 0~1
        feature_min = np.min(x, axis=0)
        feature_max = np.max(x, axis=0)
        for j in range(x.shape[1]):
            x[:, j] = (x[:, j] - feature_min[j]) / (
                feature_max[j] - feature_min[j])
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

    x = boston['data']  # (506, 13)
    y = boston['target']  # (506, )

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

    # numpy to torch tensor
    train_x = torch.from_numpy(train_x).type(torch.FloatTensor).cuda()
    train_y = torch.from_numpy(train_y).type(
        torch.FloatTensor).unsqueeze(1).cuda()
    test_x = torch.from_numpy(test_x).type(torch.FloatTensor).cuda()
    test_y = torch.from_numpy(test_y).type(
        torch.FloatTensor).unsqueeze(1).cuda()

    return train_x, train_y, test_x, test_y
