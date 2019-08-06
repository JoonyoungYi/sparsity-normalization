import os
import random

import numpy as np

from common.utils import set_seed
from .datasets import mnist as datasets_mnist
from .datasets import fashion as datasets_fashion
from .datasets import cifar10 as datasets_cifar10
from .datasets import cifar100 as datasets_cifar100


def _sparsity_normalization(x, K_prime_opt, K_prime, n_in):
    if K_prime_opt == 'n0':
        K_prime = n_in
    elif K_prime_opt == 'n1':
        raise NotImplementedError()
    elif K_prime_opt == 'custom':
        K_prime = K_prime
    else:
        assert False, "Invalid K_prime option."

    for i in range(x.shape[0]):
        row = x[i]
        count_nonzero = max(np.where(row > 1e-8)[0].shape[0], 1)
        row *= K_prime
        row /= count_nonzero
    return x


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

        set_seed(args.dataset_seed)
        (train_x, train_y), (test_x, test_y) = dataset.load_data()

        # flatten and normalize
        if args.dataset == 'mnist' or args.dataset == 'fashion':
            train_x = train_x.reshape([-1, 28 * 28]).astype('float32') / 255.0
            test_x = test_x.reshape([-1, 28 * 28]).astype('float32') / 255.0
        elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
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
            n_in, n_out = 32 * 32 * 3, 10
        elif args.dataset == 'cifar100':
            n_in, n_out = 32 * 32 * 3, 100
        else:
            raise NotImplementedError()
        self.n_in = n_in
        self.n_out = n_out

        # sparsity injection
        if args.sparsity_normalization:
            train_x = _sparsity_normalization(train_x, args.K_prime_opt, args.K_prime, self.n_in)
            test_x = _sparsity_normalization(test_x, args.K_prime_opt, args.K_prime, self.n_in)

        # split
        train_perm = np.random.permutation(train_x.shape[0])
        train_x = train_x[train_perm]
        train_y = train_y[train_perm]

        test_perm = np.random.permutation(test_x.shape[0])
        test_x = test_x[test_perm]
        test_y = test_y[test_perm]

        if args.dataset == 'mnist' or args.dataset == 'fashion':
            if args.valid_split == 'from_train':
                valid_x = train_x[50000:]
                train_x = train_x[:50000]
                valid_y = train_y[50000:]
                train_y = train_y[:50000]
            elif args.valid_split == 'from_test':
                valid_x = test_x[5000:]
                test_x = test_x[:5000]
                valid_y = test_y[5000:]
                test_y = test_y[:5000]
            else:
                assert False, "Invalid split."

        elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
            if args.valid_split == 'from_train':
                valid_x = train_x[40000:]
                train_x = train_x[:40000]
                valid_y = train_y[40000:]
                train_y = train_y[:40000]
            elif args.valid_split == 'from_test':
                raise NotImplementedError()
            else:
                assert False, "Invalid split."
        else:
            raise NotImplementedError()

        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y = test_y

        # merge train, valid, test to plot activation histogram
        if args.plot_activations:  # for safety.
            self.total_x = np.concatenate((train_x, valid_x, test_x))
            self.total_y = np.concatenate((train_y, valid_y, test_y))
        else:
            self.total_x = None
            self.total_y = None
