import os
import pickle

import torch
import numpy as np
from matplotlib import pyplot as plt

for dropout_mode in ['default', 'sparsity-normalization']:
    with open('dropout/results-{}.pickle'.format(dropout_mode), 'rb') as f:
        data = pickle.load(f)

    test_y_preds = data['test_y_preds']
    masks = data['masks']

    for layer_idx in range(2):
        for target_index in range(5):
            xs = []
            for ms in masks:
                # print(ms)
                # print(len(ms))
                mask = np.stack([
                    m.numpy()[target_index:target_index + 1]
                    for m in ms[layer_idx:layer_idx + 1]
                ])
                n_unit = np.sum(mask, axis=0)
                xs.append(n_unit)

            # plt.ylim([1.5, max(results) * 2])
            x = np.concatenate(xs)
            y = np.concatenate([
                p.cpu().numpy()[target_index:target_index + 1]
                for p in test_y_preds
            ]).reshape([-1])

            d = {}
            for i in range(x.shape[0]):
                _x, _y = x[i], y[i]
                key = int(_x)

                s = d.get(key, [])
                s.append(_y)
                d[key] = s

            # results = np.array([[k, sum(v) / len(v)] for k, v in sorted(d.items())])
            # plt.plot(results[:, 0], results[:, 1], label=dropout_mode)
            # print(x.shape)
            # print(y.shape)
            # input('')
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--")
            plt.scatter(x, y, s=1, label=dropout_mode)
            plt.legend()
            plt.xlim([40, 50])
            plt.ylim([
                [10, 24],
                [12, 20],
                [12, 26],
                [16, 26],
                [28, 44],
            ][target_index])
            # plt.savefig('dropout/vsp.png', format='png')
            plt.savefig(
                'dropout/target{target_idx}-vsp-layer{layer_idx}-{}.png'.
                format(
                    dropout_mode, target_idx=target_index,
                    layer_idx=layer_idx),
                format='png')
            plt.close()
