import os
import random

import numpy as np
from .datasets import mnist as datasets_mnist
from .datasets import fashion as datasets_fashion
from .datasets import cifar10 as datasets_cifar10
from .datasets import cifar100 as datasets_cifar100


def _normalize_input(x):
    for i in range(x.shape[0]):
        row = x[i]
        count_nonzero = max(np.where(row > 1e-8)[0].shape[0], 1)
        row /= count_nonzero
    return x


def inject_sparsity(x, args):
    g = np.zeros(x.shape[0])
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
            if args.normalize_input:
                divider = x.shape[1] - m[0].shape[0]
                assert divider >= 1
                row /= divider
            x[i] = row
    return x, g


class Dataset:
    def __init__(self, args):
        # load data
        if args.dataset == 'mnist':
            dataset = datasets_mnist
        elif args.dataset == 'fashion':
            dataset = datasets_fashion
        elif args.dataset == 'cifar10':
            dataset = datasets_cifar10
        elif args.dataset == 'cifar100':
            dataset = datasets_cifar100
        else:
            raise NotImplementedError()
        assert dataset

        (train_x, train_y), (test_x, test_y) = dataset.load_data()

        # flatten and normalize
        if args.dataset == 'mnist' or args.dataset == 'fashion':
            train_x = train_x.reshape([-1, 28 * 28]).astype('float32') / 255.0
            test_x = test_x.reshape([-1, 28 * 28]).astype('float32') / 255.0
        elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
            if args.grey_scale:
                train_x = np.average(train_x, 3).reshape(
                    [-1, 32 * 32]).astype('float32') / 255.0
                test_x = np.average(test_x, 3).reshape(
                    [-1, 32 * 32]).astype('float32') / 255.0
            else:
                train_x = train_x.reshape([-1, 32 * 32 * 3
                                           ]).astype('float32') / 255.0
                test_x = test_x.reshape([-1, 32 * 32 * 3
                                         ]).astype('float32') / 255.0

            train_y = train_y.reshape([-1])
            test_y = test_y.reshape([-1])
        else:
            raise NotImplementedError()

        if args.dataset == "mnist" or args.dataset == "fashion":
            n_in, n_out = 784, 10
        elif args.dataset == 'cifar10':
            if args.grey_scale:
                n_in, n_out = 32 * 32, 10
            else:
                n_in, n_out = 32 * 32 * 3, 10
        elif args.dataset == 'cifar100':
            if args.grey_scale:
                n_in, n_out = 32 * 32, 100
            else:
                n_in, n_out = 32 * 32 * 3, 100
        else:
            raise NotImplementedError()
        self.n_in = n_in
        self.n_out = n_out

        # sparsity injection
        if args.inject_sparsity:
            train_x, train_g = inject_sparsity(train_x, args)
            test_x, test_g = inject_sparsity(test_x, args)
        else:
            if args.normalize_input:
                train_x = _normalize_input(train_x)
                test_x = _normalize_input(test_x)
            train_g = None
            test_g = None

        # split
        train_perm = np.random.permutation(train_x.shape[0])
        train_x = train_x[train_perm]
        train_y = train_y[train_perm]

        if args.dataset == 'mnist' or args.dataset == 'fashion':
            valid_x = train_x[50000:]
            train_x = train_x[:50000]
            valid_y = train_y[50000:]
            train_y = train_y[:50000]
        elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
            valid_x = train_x[40000:]
            train_x = train_x[:40000]
            valid_y = train_y[40000:]
            train_y = train_y[:40000]
        else:
            raise NotImplementedError()

        if train_g and test_g:
            dense_test_x = test_x[test_g > 0.5]
            dense_test_y = test_y[test_g > 0.5]

            sparse_test_x = test_x[test_g < -0.5]
            sparse_test_y = test_y[test_g < -0.5]
        else:
            dense_test_x = test_x
            dense_test_y = test_y

            sparse_test_x = test_x
            sparse_test_y = test_y

        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y = test_y
        self.sparse_test_x = sparse_test_x
        self.sparse_test_y = sparse_test_y
        self.dense_test_x = dense_test_x
        self.dense_test_y = dense_test_y

        # merge train, valid, test to plot activation histogram
        if args.plot_raw_activations:  # for safety.
            self.total_x = np.concatenate((train_x, valid_x, test_x))
            self.total_y = np.concatenate((train_y, valid_y, test_y))
        else:
            self.total_x = None
            self.total_y = None
