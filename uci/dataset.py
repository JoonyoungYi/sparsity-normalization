import math

import torch
import numpy as np


def _get_data(train=True, dataset=None):
    if train:
        xs = np.load(
            'uci/datasets/{dataset}/train_x.npy'.format(dataset=dataset))
        ys = np.load(
            'uci/datasets/{dataset}/train_y.npy'.format(dataset=dataset))
    else:
        xs = np.load(
            'uci/datasets/{dataset}/test_x.npy'.format(dataset=dataset))
        ys = np.load(
            'uci/datasets/{dataset}/test_y.npy'.format(dataset=dataset))

    ys = ys.reshape([-1, 1])
    return xs, ys

    # if imputation == 'zero':
    #     xs[np.isnan(xs)] = 0
    #     if sparsity_normalization:
    #         l0 = np.maximum(np.sum(xs, axis=1, keepdims=True), 1)
    #         xs = xs * n_feature / l0
    # elif imputation == 'mean':
    #     for j in range(xs.shape[1]):
    #         _xs = xs[:, j]
    #         _mean = np.mean(_xs[~np.isnan(_xs)])
    #         _xs[np.isnan(_xs)] = _mean
    # else:
    #     raise NotImplementedError()


# assert imputation in ['zero', 'mean']
# if sparsity_normalization:
#     assert imputation == 'zero'
#
#
# imputation='zero',
# # imputation='mean',
# # sparsity_normalization=True,
# sparsity_normalization=False


def _normalize_output(train_y, test_y):
    y_mean = np.mean(train_y)
    y_std = np.std(train_y)

    train_y = (train_y - y_mean) / y_std
    test_y = (test_y - y_mean) / y_std
    return train_y, test_y

    # y_max = np.max(train_y)
    # y_min = np.min(train_y)
    #
    # train_y = (train_y - y_min) / (y_max - y_min)
    # test_y = (test_y - y_min) / (y_max - y_min)

    # train_y = (train_y - y_min)
    # test_y = (test_y - y_min)
    return train_y, test_y


def _normalize_input(train_x, test_x, input_normalization):
    # for j in range(train_x.shape[1]):
    #     _train_x = train_x[:, j]
    #     _train_x = _train_x[~np.isnan(_train_x)]
    #     mean = np.mean(_train_x)
    #     std = np.std(_train_x)
    #
    #     if math.isnan(mean) or math.isnan(std):
    #         continue
    #     elif std <= 1e-20:
    #         train_x[:, j] = 0
    #         test_x[:, j] = 0
    #     else:
    #         # print(mean, std)
    #         train_x[:, j] = (train_x[:, j] - mean) / std
    #         test_x[:, j] = (test_x[:, j] - mean) / std

    if input_normalization == 'min-max':
        train_x = (train_x + 104) / 104
        test_x = (test_x + 104) / 104
        return train_x, test_x

    elif input_normalization == 'standardization':
        train_x_valid = train_x[~np.isnan(train_x)]
        x_mean = np.mean(train_x_valid)
        x_std = np.std(train_x_valid)

        train_x = (train_x - x_mean) / x_std
        test_x = (test_x - x_mean) / x_std
        return train_x, test_x

    else:
        raise NotImplementedError()


def _sparsity_normalization(train_x, test_x):
    train_c = np.sum(~np.isnan(train_x), axis=1, keepdims=True)
    k_prime = np.mean(train_c)
    print(k_prime)
    train_c = np.maximum(train_c, 1.0)

    test_c = np.sum(~np.isnan(test_x), axis=1, keepdims=True)
    test_c = np.maximum(test_c, 1.0)

    train_x = train_x * k_prime / train_c
    test_x = test_x * k_prime / test_c
    return train_x, test_x


def _impute(train_x, test_x, imputation):

    if imputation == 'mean':
        mean = np.mean(train_x[~np.isnan(train_x)])
        train_x[np.isnan(train_x)] = mean
        test_x[np.isnan(test_x)] = mean
    elif imputation == 'zero':
        train_x[np.isnan(train_x)] = 0
        test_x[np.isnan(test_x)] = 0
    else:
        raise NotImplementedError()

    return train_x, test_x


def _convert(xs, ys):
    xs = torch.from_numpy(xs).type(torch.FloatTensor).cuda()
    ys = torch.from_numpy(ys).type(torch.FloatTensor).cuda()
    return xs, ys


class Dataset:
    def __init__(self,
                 dataset,
                 sparsity_normalization=False,
                 imputation=None,
                 input_normalization=None):
        assert imputation
        assert input_normalization

        train_x, train_y = _get_data(train=True, dataset=dataset)
        test_x, test_y = _get_data(train=False, dataset=dataset)

        train_y, test_y = _normalize_output(train_y, test_y)

        train_x, test_x = _normalize_input(train_x, test_x,
                                           input_normalization)
        if sparsity_normalization:
            train_x, test_x = _sparsity_normalization(train_x, test_x)
        train_x, test_x = _impute(train_x, test_x, imputation=imputation)

        # test_x, test_y = train_x[-2000:], train_y[-2000:]
        # train_x, train_y = train_x[:-2000], train_y[:-2000]

        train_x, train_y = _convert(train_x, train_y)
        test_x, test_y = _convert(test_x, test_y)

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
