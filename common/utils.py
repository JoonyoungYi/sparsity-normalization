import random
import argparse

import tensorflow as tf
import numpy as np


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def inject_sparsity(x, args):
    if args.inject_sparsity:
        for i in range(x.shape[0]):
            row = x[i]

            m = None
            while m is None or m[0].shape[0] >= x.shape[1]:
                if random.random() < 0.5:
                    m = np.where(np.random.random(x.shape[1]) < 0.2)
                else:
                    m = np.where(np.random.random(x.shape[1]) < 0.8)

            row[m] = 0
            if args.normalize_input:
                # print(row)
                row /= (x.shape[1] - m[0].shape[0])
                # print(row)
                # input('')
            x[i] = row
    return x


def split_sparse_and_dense_group(test_x):
    sparse_group = np.sum(test_x > 1e-8, axis=1) <= (test_x.shape[1] // 2)
    dense_group = np.sum(test_x > 1e-8, axis=1) > (test_x.shape[1] // 2)
    return test_x[sparse_group], test_x[dense_group]


def set_seed(seed):
    # set seed
    random.seed(seed)
    np.random.seed(seed=seed)
    tf.random.set_random_seed(seed)
