import os
import random

import numpy as np
from PIL import Image
from PIL import ImageOps
import numpy as np
from tqdm import tqdm

from common.utils import set_seed
from .datasets import mnist as datasets_mnist
from .datasets import fashion as datasets_fashion
from .datasets import cifar10 as datasets_cifar10
from .datasets import cifar100 as datasets_cifar100


def _sparsity_normalization(x, K_prime=None):
    if K_prime is None or K_prime <= 0:
        cs = []
        for i in range(x.shape[0]):
            row = x[i]
            count_nonzero = max(np.where(row > 1e-8)[0].shape[0], 1)
            cs.append(count_nonzero)
        K_prime = sum(cs) / len(cs)
        print('>> calculated K_prime:', K_prime)
        input('')

    for i in range(x.shape[0]):
        row = x[i]
        count_nonzero = max(np.where(row > 1e-8)[0].shape[0], 1)
        row *= K_prime
        row /= count_nonzero
        x[i] = row
    return x, K_prime


def _resize(images, max_padding=28):
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

        # max_dict, min_dict = {}, {}
        # for i in range(train_x.shape[0]):
        #     x = train_x[i]
        #     y = train_y[i]
        #     l0 = max(np.where(x > 1e-8)[0].shape[0], 1)
        #     max_dict[y] = max(max_dict.get(y, 0), l0)
        #     min_dict[y] = min(min_dict.get(y, 28 * 28), l0)
        # for i in range(test_x.shape[0]):
        #     x = test_x[i]
        #     y = test_y[i]
        #     l0 = max(np.where(x > 1e-8)[0].shape[0], 1)
        #     max_dict[y] = max(max_dict.get(y, 0), l0)
        #     min_dict[y] = min(min_dict.get(y, 28 * 28), l0)
        # print(max_dict)
        # print(min_dict)

        # for digit in range(10):
        #     l0_max = max_dict.get(digit)
        #     l0_min = min_dict.get(digit)
        #     print(digit, l0_min / l0_max, l0_max, l0_min)
        # input('')

        # train_valid_idx = train_y < 2
        # train_x = train_x[train_valid_idx, :]
        # train_y = train_y[train_valid_idx]
        #
        # test_valid_idx = test_y < 2
        # test_x = test_x[test_valid_idx, :]
        # test_y = test_y[test_valid_idx]

        # print(train_x.shape)
        # print(train_y.shape)
        # input('')

        if args.dataset == 'mnist' or args.dataset == 'fashion':
            train_x, train_g = _resize(train_x)
            test_x, test_g = _resize(test_x)

        # flatten and normalize
        if args.dataset == 'mnist' or args.dataset == 'fashion':
            train_x = train_x.reshape([-1, 28 * 28]).astype('float32') / 255.0
            test_x = test_x.reshape([-1, 28 * 28]).astype('float32') / 255.0
        elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
            train_x = 1.0 - train_x.reshape([-1, 32 * 32 * 3
                                             ]).astype('float32') / 255.0
            test_x = 1.0 - test_x.reshape([-1, 32 * 32 * 3
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
            train_x, K_prime = _sparsity_normalization(train_x, args.K_prime)
            test_x, _ = _sparsity_normalization(test_x, K_prime)

        # split
        perm = np.random.permutation(test_x.shape[0])
        test_x = test_x[perm]
        test_y = test_y[perm]
        test_g = test_g[perm]

        self.dense_test_x = test_x[test_g == 0]
        self.sparse_test_x = test_x[test_g == 1]
        self.dense_test_y = test_y[test_g == 0]
        self.sparse_test_y = test_y[test_g == 1]

        if args.dataset == 'mnist' or args.dataset == 'fashion':
            split_idx = test_x.shape[0] // 2
            valid_x = test_x[split_idx:]
            test_x = test_x[:split_idx]
            valid_y = test_y[split_idx:]
            test_y = test_y[:split_idx]
        elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
            split_idx = test_x.shape[0] // 2
            valid_x = test_x[split_idx:]
            test_x = test_x[:split_idx]
            valid_y = test_y[split_idx:]
            test_y = test_y[:split_idx]
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
