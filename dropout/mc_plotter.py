import os
import pickle

import torch
import numpy as np
from matplotlib import pyplot as plt

from .dataset import load_data

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(
    'standardization', dataset_seed=1)

max_t = 1000
test_y = test_y.cpu().numpy().reshape((-1, ))

for seed in range(1, 6):
    for dropout_mode in ['default', 'sparsity-normalization']:
        with open('dropout/results-{}-seed{seed}.pickle'.format(
                dropout_mode, seed=seed), 'rb') as f:
            data = pickle.load(f)

        test_y_preds = np.stack([
            test_y_pred.cpu().numpy().reshape((-1, ))
            for test_y_pred in data['test_y_preds']
        ])

        mses = []
        for t in range(1, max_t + 1):
            test_y_pred = np.mean(test_y_preds[-t:, :], axis=0)

            mse = np.mean((test_y_pred - test_y)**2)
            # print(dropout_mode, mse)
            mses.append((t, mse))

        result = np.array(mses)
        print(dropout_mode, mses[-1])
        plt.plot(result[:, 0], result[:, 1], label=dropout_mode)

    plt.legend()
    plt.xlim([0, 1000])
    plt.ylim([5, 8])
    plt.savefig('dropout/mc_mse-{seed}.png'.format(seed=seed), format='png')
    plt.close()
