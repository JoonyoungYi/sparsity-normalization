import numpy as np
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_california_housing
import torch

from common.utils import set_torch_seed


def _load_data(name, n_col=12, splitter=";"):
    f = open('dropout/datasets/{}.txt'.format(name))
    f.readline()

    data = []
    for line in f:
        cols = line.strip().split(splitter)
        # print(len(cols))
        if len(cols) != n_col:
            print(len(cols))
            assert len(cols) == n_col

        for j, c in enumerate(cols):
            try:
                cols[j] = float(c)
            except:
                cols[j] = float('nan')
        data.append(cols)

    data = np.array(data)
    return data


def _load_datasets(name, n_col=12, splitter=";"):
    data = _load_data(name, n_col=n_col, splitter=splitter)
    target = data[:, -1]
    data = data[:, :-1]
    return {
        'data': data,
        'target': target,
    }


def load_white_wine():
    return _load_datasets('white-wine')


def load_red_wine():
    return _load_datasets('red-wine')


def load_parkinsons():
    return _load_datasets('parkinsons', splitter=",", n_col=22)


def load_coil_2000():
    return _load_datasets('coil-2000', splitter="\t", n_col=86)


def load_power_plant():
    return _load_datasets('power-plant', splitter="\t", n_col=5)


def load_grid_stability():
    data = _load_data('grid-stability', splitter=",", n_col=14)
    target = data[:, -2]
    data = data[:, :-2]
    return {
        'data': data,
        'target': target,
    }


def load_online_news():
    data = _load_data('online-news', splitter=",", n_col=61)
    target = data[:, -1]
    data = data[:, 1:-1]
    return {
        'data': data,
        'target': target,
    }


def load_protein():
    data = _load_data('protein', splitter=",", n_col=10)
    target = data[:, 0]
    data = data[:, 0:]
    return {
        'data': data,
        'target': target,
    }


def load_gpu():
    data = _load_data('gpu', splitter=",", n_col=18)
    target = np.mean(data[:, -4:], axis=1)
    data = data[:, :-4]
    return {
        'data': data,
        'target': target,
    }


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


def _normalize_output(y):
    # if input_normalization == 'none':
    #     pass
    # elif input_normalization == 'min-max':
    #     # feature-wise normalization min~max = 0~1
    #     feature_min = np.min(y, axis=0)
    #     feature_max = np.max(y, axis=0)
    #     for j in range(x.shape[1]):
    #         x[:, j] = (x[:, j] - feature_min[j]) / (
    #             feature_max[j] - feature_min[j])
    # elif input_normalization == 'standardization':
    # feature-wise standardization
    feature_mean = np.mean(y, axis=0)
    feature_std = np.std(y, axis=0)
    print(feature_mean, feature_std)
    y = (y - feature_mean) / feature_std
    # for j in range(x.shape[1]):
    #     x[:, j] = (x[:, j] - feature_mean[j]) / feature_std[j]
    # else:
    #     raise NotImplementedError()
    # return x
    return y


def load_data(dataset,
              input_normalization,
              train_test_ratio=0.9,
              train_valid_ratio=0.8,
              dataset_seed=1,
              regression=True):
    assert dataset
    set_torch_seed(dataset_seed)

    # get boston dataset from sklearn
    if dataset == 'boston':
        loader = load_boston()
    elif dataset == 'diabetes':
        loader = load_diabetes()
    elif dataset == 'iris':
        loader = load_iris()
    elif dataset == 'white-wine':
        loader = load_white_wine()
    elif dataset == 'red-wine':
        loader = load_red_wine()
    elif dataset == 'parkinsons':
        loader = load_parkinsons()
    elif dataset == 'coil-2000':
        loader = load_coil_2000()
    elif dataset == 'power-plant':
        loader = load_power_plant()
    elif dataset == 'grid-stability':
        loader = load_grid_stability()
    elif dataset == 'online-news':
        loader = load_online_news()
    elif dataset == 'protein':
        loader = load_protein()
    elif dataset == 'gpu':
        loader = load_gpu()

    elif dataset == 'wine':
        loader = load_wine()
    elif dataset == 'breast-cancer':
        loader = load_breast_cancer()
    elif dataset == 'california':
        loader = fetch_california_housing()
    else:
        raise NotImplementedError()

    x = loader['data']  # (506, 13)
    y = loader['target']  # (506, )

    # unique, counts = np.unique(y, return_counts=True)
    # print(dict(zip(unique, counts)))
    # input('!')

    # input normalization
    x = _normalize_input(x, input_normalization)
    y = _normalize_output(y)

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
    valid_x = torch.from_numpy(valid_x).type(torch.FloatTensor).cuda()
    test_x = torch.from_numpy(test_x).type(torch.FloatTensor).cuda()

    if regression:
        train_y = torch.from_numpy(train_y).type(
            torch.FloatTensor).unsqueeze(1).cuda()
        valid_y = torch.from_numpy(valid_y).type(
            torch.FloatTensor).unsqueeze(1).cuda()
        test_y = torch.from_numpy(test_y).type(
            torch.FloatTensor).unsqueeze(1).cuda()
    else:
        train_y = torch.from_numpy(train_y).cuda()
        valid_y = torch.from_numpy(valid_y).cuda()
        test_y = torch.from_numpy(test_y).cuda()

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
