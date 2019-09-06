import os

import numpy as np
from matplotlib import pyplot as plt


def _get_data(dropout_rate):
    f = open('.results/lenet/cifar10/epoch/false_{}_default.txt'.format(
        dropout_rate))
    rows = []
    for line in f:
        cols = line.strip().split('\t')
        assert len(cols) == 7
        rows.append([float(c) for c in cols])
    f.close()
    return np.array(rows)


for dropout_rate in [0.1, 0.3, 0.4, 0.5, 0.8, 0.9]:
    d = _get_data(dropout_rate)
    epoch_idx = d[:, 0]
    train_acc = d[:, 2]
    valid_acc = d[:, 4]
    test_acc = d[:, 6]
    print(dropout_rate, test_acc[np.argmax(valid_acc)])
