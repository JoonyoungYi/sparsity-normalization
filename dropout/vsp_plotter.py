import os
import pickle

import torch
import numpy as np
from matplotlib import pyplot as plt

target_index = 4

for dropout_mode in ['default', 'sparsity-normalization']:
    with open('dropout/results-{}.pickle'.format(dropout_mode), 'rb') as f:
        data = pickle.load(f)

    test_y_preds = data['test_y_preds']
    masks = data['masks']
    xs = []
    for ms in masks:
        mask = np.stack([m.numpy()[target_index:target_index + 1] for m in ms])
        n_unit = np.sum(mask, axis=0)
        xs.append(n_unit)

    # plt.ylim([1.5, max(results) * 2])
    x = np.concatenate(xs)
    y = np.concatenate(
        [p.cpu().numpy()[target_index:target_index + 1] for p in test_y_preds])

    d = {}
    for i in range(x.shape[0]):
        _x, _y = x[i], y[i]
        key = int(_x)

        s = d.get(key, [])
        s.append(_y)
        d[key] = s

    # results = np.array([[k, sum(v) / len(v)] for k, v in sorted(d.items())])
    # plt.plot(results[:, 0], results[:, 1], label=dropout_mode)
    plt.scatter(x, y, s=1, label=dropout_mode)
    plt.legend()
    # plt.xlim([70, 100])
    # plt.ylim([0, 60])
    # plt.savefig('dropout/vsp.png', format='png')
    plt.savefig('dropout/vsp-{}.png'.format(dropout_mode), format='png')
    plt.close()
