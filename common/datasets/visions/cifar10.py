import os

import numpy as np
from tensorflow.python.keras.datasets.cifar import load_batch

from ..utils import get_file_path


def load_data():
    dir_path = get_file_path(
        'cifar-10-python.tar.gz',
        'cifar-10',
        url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
        hash='c58f30108f718f92721af3b95e74349a',
        extract=True)

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples, ), dtype='uint8')

    path = dir_path + '/cifar-10-batches-py'
    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)
    return (x_train, y_train), (x_test, y_test)
