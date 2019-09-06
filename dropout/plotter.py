import os

import numpy as np
from matplotlib import pyplot as plt

from .logger import get_file_name


def _get_data(file_name):
    f = open('.results/dropout/boston/epoch/{}.txt'.format(file_name))
    rows = []
    for line in f:
        cols = line.strip().split('\t')
        assert len(cols) == 5
        rows.append([float(c) for c in cols])
    f.close()
    return np.array(rows)


def _plot(dropout_rate, activation_fn, folder_name, n_batch, optimizer,
          learning_rate, n_hidden_layer, n_hidden_unit):
    results = []
    for dropout_mode in ['default', 'sparsity-normalization']:
        file_name = get_file_name(
            n_hidden_layer=n_hidden_layer,
            n_hidden_unit=n_hidden_unit,
            activation_fn=activation_fn,
            optimizer=optimizer,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            dropout_mode=dropout_mode,
            max_epoch=100000,
            n_batch=n_batch,
            input_normalization='standardization',
            seed=1)

        d = _get_data(file_name)

        epoch_idx = d[:, 0]
        train_rmse = d[:, 2]
        valid_rmse = d[:, 3]
        test_rmse = d[:, 4]

        results.append(test_rmse[np.argmin(valid_rmse)])

        # plt.plot(epoch_idx, train_rmse, label='{}-train'.format(dropout_mode))
        # plt.plot(epoch_idx, valid_rmse, label='{}-valid'.format(dropout_mode))
        plt.plot(epoch_idx, test_rmse, label='{}-test'.format(dropout_mode))
        plt.legend()

    file_name = get_file_name(
        n_hidden_layer=n_hidden_layer,
        n_hidden_unit=n_hidden_unit,
        activation_fn=activation_fn,
        optimizer=optimizer,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        dropout_mode='all',
        max_epoch=100000,
        n_batch=n_batch,
        input_normalization='standardization',
        seed=1)
    plt.ylim([1.5, max(results) * 2])
    plt.savefig(
        '.results/dropout/boston/figures/learning-curve/{}/{}.png'.format(
            folder_name, file_name),
        format='png')
    plt.close()
    return results


def _mkdir_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def _init_folders(folder_name):
    _mkdir_if_not_exists('.results')
    _mkdir_if_not_exists('.results/dropout')
    _mkdir_if_not_exists('.results/dropout/boston')
    _mkdir_if_not_exists('.results/dropout/boston/figures')
    _mkdir_if_not_exists('.results/dropout/boston/figures/learning-curve')
    _mkdir_if_not_exists('.results/dropout/boston/figures/learning-curve/{}'.
                         format(folder_name))
    _mkdir_if_not_exists('.results/dropout/boston/figures/test-rmse')


def main(activation_fn='relu',
         n_batch=128,
         optimizer='adam',
         learning_rate=1e-3,
         n_hidden_layer=2,
         n_hidden_unit=50):
    file_name = get_file_name(
        n_hidden_layer=n_hidden_layer,
        n_hidden_unit=n_hidden_unit,
        activation_fn=activation_fn,
        optimizer=optimizer,
        learning_rate=learning_rate,
        dropout_rate='all',
        dropout_mode='all',
        max_epoch=100000,
        n_batch=n_batch,
        input_normalization='standardization',
        seed=1)
    _init_folders(file_name)

    test_rmses = []
    for dropout_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        rmse_default, rmse_sparsity_normalization = _plot(
            dropout_rate=dropout_rate,
            activation_fn=activation_fn,
            folder_name=file_name,
            n_batch=n_batch,
            optimizer=optimizer,
            n_hidden_layer=n_hidden_layer,
            n_hidden_unit=n_hidden_unit,
            learning_rate=learning_rate)
        test_rmses.append(
            [dropout_rate, rmse_default, rmse_sparsity_normalization])
        print(dropout_rate, rmse_default, rmse_sparsity_normalization)
    test_rmses = np.array(test_rmses)

    plt.plot(test_rmses[:, 0], test_rmses[:, 1])
    plt.plot(test_rmses[:, 0], test_rmses[:, 2])
    plt.ylim([1.5, 4.5])
    plt.savefig(
        '.results/dropout/boston/figures/test-rmse/{}.png'.format(file_name),
        format='png')
    plt.close()


if __name__ == '__main__':
    main(
        activation_fn='relu',
        # activation_fn='leaky_relu',
        # activation_fn='elu',
        # activation_fn='sigmoid',
        # activation_fn='tanh',
        # n_hidden_layer=1,
        n_hidden_layer=2,
        # n_hidden_layer=3,
        # n_hidden_layer=4,
        # n_hidden_unit=n_hidden_unit,
        n_hidden_unit=50,
        # n_batch=128,
        n_batch=128,
        optimizer='adam',
        # optimizer='sgd',
        # optimizer='momentum',
        # learning_rate=1e-2,
        # learning_rate=1e-3,
        learning_rate=1e-4,
        # learning_rate=1e-5,
    )
