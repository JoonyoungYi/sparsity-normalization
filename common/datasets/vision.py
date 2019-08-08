import os
import random
import math

import numpy as np

from ..utils import set_seed
from .visions import mnist as datasets_mnist
from .visions import fashion as datasets_fashion
from .visions import cifar10 as datasets_cifar10
from .visions import cifar100 as datasets_cifar100

# TODO: ciFAIR https://cvjena.github.io/cifair/?fbclid=IwAR2x5yhfab53CXjECieqPTjdQR4x13p4L8B-6c1yMryLreVpVa8l-7LX4XU


def __get_lp_norm(row, p):
    return np.power(np.sum(np.power(np.abs(row), p)), 1 / p)


def __get_norm(row, norm):
    if norm == 'l0':
        _norm = max(np.where(row > 1e-8)[0].shape[0], 1)
    elif norm == 'l1/4':
        _norm = __get_lp_norm(row, 0.25)
    elif norm == 'l1/2':
        _norm = __get_lp_norm(row, 0.5)
    elif norm == 'l3/4':
        _norm = __get_lp_norm(row, 0.75)
    elif norm == 'l1':
        _norm = np.sum(np.abs(row))
    elif norm == 'l5/4':
        _norm = __get_lp_norm(row, 1.25)
    elif norm == 'l3/2':
        _norm = __get_lp_norm(row, 1.5)
    elif norm == 'l7/4':
        _norm = __get_lp_norm(row, 1.75)
    elif norm == 'l2':
        _norm = np.sqrt(np.sum(np.square(row)))
    else:
        raise NotImplementedError()
    return _norm


def _sparsity_normalization(x, K_prime, norm='l0'):
    if K_prime is None:
        _norms = []
        for i in range(x.shape[0]):
            row = x[i]
            _norm = __get_norm(row, norm)
            _norms.append(_norm)
        K_prime = sum(_norms) / len(_norms)
        print('>> calculated K_prime:', K_prime)
        # input('')

    for i in range(x.shape[0]):
        row = x[i]
        _norm = __get_norm(row, norm)
        row *= (K_prime / _norm)
        # print(row)
        assert not math.isnan(K_prime / _norm)
        # print((K_prime / _norm))
        x[i] = row
    # input('')
    return x, K_prime


def _inject_sparsity(images, max_padding=28):
    assert max_padding >= 0

    if max_padding > 0.5:
        _x = np.zeros(images.shape)
        _g = np.zeros(images.shape[0])
        print('>> image resizing...')
        for i in tqdm(range(images.shape[0])):
            image_array = images[i]
            image = Image.fromarray(image_array)

            # padding = random.randint(0, max_padding)
            # if padding > 0:
            #     image = ImageOps.expand(image, border=padding)
            #     image = image.resize((28, 28))
            #     image_array = np.array(image)

            if random.random() < 0.5:
                image = ImageOps.expand(image, border=max_padding)
                image = image.resize((28, 28))
                image_array = np.array(image)
                _g[i] = 1

            _x[i] = image_array
        return _x, _g
    else:
        return images, None


class Dataset:
    def __init__(self,
                 dataset,
                 dataset_seed=1,
                 n_valid=-1000,
                 sparsity_injection=False,
                 sparsity_normalization=False,
                 K_prime=None):
        # load data
        if dataset == 'mnist':
            datasets = datasets_mnist
        elif dataset == 'fashion':
            datasets = datasets_fashion
        elif dataset == 'cifar10':
            datasets = datasets_cifar10
        elif dataset == 'cifar100':
            datasets = datasets_cifar100
        else:
            raise NotImplementedError()
        assert dataset

        set_seed(dataset_seed)
        (train_x, train_y), (test_x, test_y) = datasets.load_data()

        # flatten and normalize
        if dataset == 'mnist' or dataset == 'fashion':
            train_x = train_x.reshape([-1, 28 * 28]).astype('float32') / 255.0
            test_x = test_x.reshape([-1, 28 * 28]).astype('float32') / 255.0
        elif dataset == 'cifar10' or dataset == 'cifar100':
            train_x = train_x.reshape([-1, 32 * 32 * 3
                                       ]).astype('float32') / 255.0
            test_x = test_x.reshape([-1, 32 * 32 * 3
                                     ]).astype('float32') / 255.0

            train_y = train_y.reshape([-1])
            test_y = test_y.reshape([-1])
        else:
            raise NotImplementedError()

        if dataset == "mnist" or dataset == "fashion":
            n_in, n_out = 784, 10
        elif dataset == 'cifar10':
            n_in, n_out = 32 * 32 * 3, 10
        elif dataset == 'cifar100':
            n_in, n_out = 32 * 32 * 3, 100
        else:
            raise NotImplementedError()
        self.n_in = n_in
        self.n_out = n_out

        # split
        train_perm = np.random.permutation(train_x.shape[0])
        train_x = train_x[train_perm]
        train_y = train_y[train_perm]

        test_perm = np.random.permutation(test_x.shape[0])
        test_x = test_x[test_perm]
        test_y = test_y[test_perm]

        if n_valid > 0:
            valid_x = test_x[:n_valid]
            test_x = test_x[n_valid:]
            valid_y = test_y[:n_valid]
            test_y = test_y[n_valid:]
        elif n_valid < 0:
            valid_x = train_x[n_valid:]
            train_x = train_x[:n_valid]
            valid_y = train_y[n_valid:]
            train_y = train_y[:n_valid]
        else:
            raise Exception("n_valid must not be zero.")

        # sparsity injection
        if sparsity_injection and (dataset == 'mnist' or dataset == 'fashion'):
            train_x, train_g = _inject_sparsity(train_x)
            valid_x, valid_g = _inject_sparsity(valid_x)
            test_x, test_g = _inject_sparsity(test_x)

        # sparsity normalization
        if sparsity_normalization:
            train_x, K_prime = _sparsity_normalization(train_x, K_prime)
            valid_x, _ = _sparsity_normalization(valid_x, K_prime)
            test_x, _ = _sparsity_normalization(test_x, K_prime)

        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y = test_y

        self.total_x = np.concatenate((train_x, valid_x, test_x))
        self.total_y = np.concatenate((train_y, valid_y, test_y))

        if sparsity_injection:
            self.dense_test_x = test_x[test_g == 0]
            self.sparse_test_x = test_x[test_g == 1]
            self.dense_test_y = test_y[test_g == 0]
            self.sparse_test_y = test_y[test_g == 1]
        else:
            self.dense_test_x = None
            self.sparse_test_x = None
            self.dense_test_y = None
            self.sparse_test_y = None
