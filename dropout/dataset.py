import numpy as np
from sklearn.datasets import load_boston
import torch

from common.utils import set_torch_seed


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


def load_data(input_normalization,
              train_test_ratio=0.9,
              train_valid_ratio=0.8,
              dataset_seed=1):
    set_torch_seed(dataset_seed)

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

    split_boundary = int(x.shape[0] * train_test_ratio)
    train_x = x[:split_boundary]
    train_y = y[:split_boundary]
    test_x = x[split_boundary:]
    test_y = y[split_boundary:]

    split_boundary = int(train_x.shape[0] * train_valid_ratio)
    valid_x = train_x[split_boundary:]
    valid_y = train_y[split_boundary:]
    train_x = train_x[:split_boundary]
    train_y = train_y[:split_boundary]

    # numpy to torch tensor
    train_x = torch.from_numpy(train_x).type(torch.FloatTensor).cuda()
    train_y = torch.from_numpy(train_y).type(
        torch.FloatTensor).unsqueeze(1).cuda()
    valid_x = torch.from_numpy(valid_x).type(torch.FloatTensor).cuda()
    valid_y = torch.from_numpy(valid_y).type(
        torch.FloatTensor).unsqueeze(1).cuda()
    test_x = torch.from_numpy(test_x).type(torch.FloatTensor).cuda()
    test_y = torch.from_numpy(test_y).type(
        torch.FloatTensor).unsqueeze(1).cuda()

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
