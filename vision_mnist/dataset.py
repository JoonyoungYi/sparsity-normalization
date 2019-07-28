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


def _sparsity_normalization(x, K_prime):
    for i in range(x.shape[0]):
        row = x[i]
        count_nonzero = max(np.where(row > 1e-8)[0].shape[0], 1)
        row *= K_prime
        row /= count_nonzero
    return x


def _resize(images):
    _x = np.zeros(images.shape)
    print('>> image resizing...')
    for i in tqdm(range(images.shape[0])):
        image_array = images[i]
        # if random.random() < 0.5:
        image = Image.fromarray(image_array)
        # image.save('vision_mnist/original.png')
        image = ImageOps.expand(image, border=random.randint(0, 28))
        image = image.resize((28, 28))
        image_array = np.array(image)
        _x[i] = image_array
        # image.save('vision_mnist/converted.png')
        #     input('!')
        # else:
        #     _x[i] = image_array
    return _x


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

        train_x = _resize(train_x)
        test_x = _resize(test_x)

        # flatten and normalize
        if args.dataset == 'mnist' or args.dataset == 'fashion':
            train_x = train_x.reshape([-1, 28 * 28]).astype('float32') / 255.0
            test_x = test_x.reshape([-1, 28 * 28]).astype('float32') / 255.0
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
            train_x = _sparsity_normalization(train_x, args.K_prime)
            test_x = _sparsity_normalization(test_x, args.K_prime)

        # split
        train_perm = np.random.permutation(train_x.shape[0])
        train_x = train_x[train_perm]
        train_y = train_y[train_perm]

        if args.dataset == 'mnist' or args.dataset == 'fashion':
            valid_x = train_x[50000:]
            train_x = train_x[:50000]
            valid_y = train_y[50000:]
            train_y = train_y[:50000]
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
