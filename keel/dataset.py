import math

import torch
import numpy as np


def _get_data(
        n_feature=None,
        train=True,
        output_indexes=[],
        categorical_dict={},
        imputation='zero',
        # sparsity_normalization=True,
        sparsity_normalization=False,
        fold_idx=2, ):
    assert imputation in ['zero', 'mean']
    assert n_feature and n_feature > 0
    assert output_indexes
    if sparsity_normalization:
        assert imputation == 'zero'

    f = open(
        'keel/datasets/marketing/10fold/marketing-10-{fold_idx}{train}.dat'.
        format(fold_idx=fold_idx, train=('tra' if train else 'tst')), 'r')

    # skip to data
    for line in f:
        line = line.strip()
        if line == '@data':
            break

    # parse data
    xs, ys = [], []
    for line in f:
        cols = line.strip().split(',')
        assert len(cols) == n_feature

        x, y = [], []
        n_known_entry = 0
        for j, c in enumerate(cols):
            c = c.strip()
            if c == '<null>':
                c = float('nan')
            else:
                c = float(c)
                n_known_entry += 1

            if j in output_indexes:
                y.append(c)
            else:
                if math.isnan(c):
                    if j in categorical_dict.keys():
                        n_start, n_end = categorical_dict[j]
                        x.extend([c for _j in range(n_start, n_end + 1)])
                    else:
                        raise NotImplementedError()
                        x.append(c)
                else:
                    if j in categorical_dict.keys():
                        n_start, n_end = categorical_dict[j]
                        assert n_start <= c <= n_end
                        _x = [
                            1 if _j == int(c) else 0
                            for _j in range(n_start, n_end + 1)
                        ]
                        assert sum(_x) == 1
                        x.extend(_x)
                    else:
                        raise NotImplementedError()
                        x.append(c)

        xs.append(x)
        ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)

    if imputation == 'zero':
        xs[np.isnan(xs)] = 0
        if sparsity_normalization:
            l0 = np.maximum(np.sum(xs, axis=1, keepdims=True), 1)
            xs = xs * n_feature / l0
    elif imputation == 'mean':
        for j in range(xs.shape[1]):
            _xs = xs[:, j]
            _mean = np.mean(_xs[~np.isnan(_xs)])
            _xs[np.isnan(_xs)] = _mean
    else:
        raise NotImplementedError()

    # print(np.histogram(np.sum(xs, axis=1), bins=n_feature))
    # input('')

    # print(np.sum(xs, axis=1))
    # input('!')
    xs = torch.from_numpy(xs).type(torch.FloatTensor).cuda()
    ys = torch.from_numpy(ys).type(torch.FloatTensor).cuda()
    return xs, ys


class Dataset:
    def __init__(self):
        n_feature = 14
        categorical_dict = {
            0: [1, 2],
            1: [1, 5],
            4: [1, 9],
            6: [1, 3],
            9: [1, 3],
            10: [1, 5],
            11: [1, 8],
            12: [1, 3],
            2: [1, 7],
            3: [1, 6],
            5: [1, 5],
            7: [1, 9],
            8: [0, 9],
        }
        output_indexes = [13]

        train_x, train_y = _get_data(
            n_feature=n_feature,
            train=True,
            output_indexes=output_indexes,
            categorical_dict=categorical_dict)
        test_x, test_y = _get_data(
            n_feature=n_feature,
            train=False,
            output_indexes=output_indexes,
            categorical_dict=categorical_dict)

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
