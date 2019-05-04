import re
import random

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer


def _normalize_feature(x):
    # feature-wise normalization min-max=0-1
    feature_min = np.min(x, axis=0)
    feature_max = np.max(x, axis=0)
    for j in range(x.shape[1]):
        x[:, j] = (x[:, j] - feature_min[j]) / (
            feature_max[j] - feature_min[j])
    return x


def _inject_sparsity(x, args):
    g = np.zeros(x.shape[0])  # g means group
    if args.inject_sparsity:
        for i in range(x.shape[0]):
            row = x[i]

            m = None
            while m is None or m[0].shape[0] >= x.shape[1]:
                if random.random() < 0.5:
                    m = np.where(np.random.random(x.shape[1]) < 0.2)
                    g[i] = 1
                else:
                    m = np.where(np.random.random(x.shape[1]) < 0.8)
                    g[i] = -1

            row[m] = 0
            # assert np.sum(row) > 1e-8

            if args.normalize_input:
                # print(row)
                # row = np.true_divide(row, )
                divider = x.shape[1] - m[0].shape[0]
                assert divider >=1
                # print(row.shape)
                # print(divider)
                row /= divider
                # print(row)
                # input('')
            x[i] = row

    return x, g


def _split_sparse_and_dense_group(test_x, g):
    return test_x[g < -0.5], test_x[g > 0.5]


def _load_congress():
    f = open('uci/data/congress.csv', 'r')
    x = []
    # m = []
    y = []
    for line in f:
        tokens = line.strip().split(',')
        assert len(tokens) == 17
        y.append(1 if tokens[0] == 'republican' else 0)

        tokens = tokens[1:]
        _x = []
        for t in tokens:
            if t == 'y':
                _x.extend([1.])
            elif t == '?':
                _x.extend([0.])
            elif t == 'n':
                _x.extend([0.])
            else:
                raise NotImplementedError()

            # if t == 'y':
            #     _x.extend([1])
            # elif t == '?':
            #     _x.extend([0])
            # elif t == 'n':
            #     _x.extend([0])
            # else:
            #     raise NotImplementedError()
        x.append(_x)
    f.close()

    data = np.array(x)
    target = np.array(y)

    # # remove unused rows
    # while True:
    #     diff = np.max(data, axis=1) - np.min(data, axis=1)
    #     no_info = np.where(diff < 1e-8)[0]
    #     if no_info.shape[0] == 0:
    #         break
    #
    #     index = no_info[0]
    #     data = np.delete(data, index, 0)
    #     target = np.delete(target, index, 0)

    class Congress:
        pass

    congress = Congress()
    congress.data = data
    congress.target = target
    return congress


class Dataset:
    def __init__(self, args, column_multiplier):
        # https://scikit-learn.org/stable/datasets/index.html
        # https://datascienceschool.net/view-notebook/4b948c4d99c94213bf02bcff2ced88eb/
        # https://datascienceschool.net/view-notebook/577a01e24d4f456bb5060da6e47054e1/
        if args.dataset == 'boston':
            dataset = load_boston()
            n_class = None
        elif args.dataset == 'diabetes':
            dataset = load_diabetes()
            n_class = None
        elif args.dataset == 'california':
            dataset = fetch_california_housing()
            n_class = None
        elif args.dataset == 'iris':
            dataset = load_iris()
            n_class = 3
        elif args.dataset == 'wine':
            dataset = load_wine()
            n_class = 3
        elif args.dataset == 'cancer':
            dataset = load_breast_cancer()
            n_class = 2
        elif args.dataset == 'congress':
            dataset = _load_congress()
            n_class = 2
        else:
            raise NotImplementedError()

        x = dataset.data
        y = dataset.target

        if args.normalize_feature:
            x = _normalize_feature(x)

        # augment features
        print(x.shape)
        x = np.concatenate([x] * column_multiplier, axis=1)

        # split
        perm = np.random.permutation(x.shape[0])
        x = x[perm]
        y = y[perm]

        train_x = x[:int(x.shape[0] * 0.6)]
        valid_x = x[int(x.shape[0] * 0.6):int(x.shape[0] * 0.8)]
        test_x = x[int(x.shape[0] * 0.8):]

        train_y = y[:int(x.shape[0] * 0.6)]
        valid_y = y[int(x.shape[0] * 0.6):int(x.shape[0] * 0.8)]
        test_y = y[int(x.shape[0] * 0.8):]

        # sparsity injection
        train_x, train_g = _inject_sparsity(train_x, args)
        valid_x, valid_g = _inject_sparsity(valid_x, args)
        test_x, test_g = _inject_sparsity(test_x, args)

        # mean = np.mean(train_y)
        # print('>> Baseline (mean predictor) error:',
        #       np.sqrt(np.mean(np.square(mean - test_y))))
        # train_mean = np.mean(train_y)
        # print('>> Train mean:', train_mean)
        # test_mean = np.mean(test_y)
        # print('>> Test mean:', test_mean)
        # input('')

        sparse_test_x, dense_test_x = _split_sparse_and_dense_group(
            test_x, test_g)

        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y = test_y
        self.n_class = n_class

        self.sparse_test_x = sparse_test_x
        self.dense_test_x = dense_test_x
