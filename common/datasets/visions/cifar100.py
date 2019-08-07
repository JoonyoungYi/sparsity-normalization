import os

import numpy as np
from tensorflow.python.keras.datasets.cifar import load_batch

from ..utils import get_file_path


def load_data(label_mode='fine'):
    dir_path = get_file_path(
        'cifar-100-python.tar.gz',
        'cifar-100',
        url='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
        hash='eb9058c3a382ffc7106e4002c42a8d85',
        extract=True)

    path = dir_path + '/cifar-100-python'
    fpath = os.path.join(path, 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)
