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


def _plot(dropout_rate, activation_fn):
    results = []
    for dropout_mode in ['default', 'sparsity-normalization']:
        file_name = get_file_name(
            n_hidden_layer=2,
            n_hidden_unit=50,
            activation_fn=activation_fn,
            optimizer='adam',
            learning_rate=1e-3,
            dropout_rate=dropout_rate,
            dropout_mode=dropout_mode,
            max_epoch=20000,
            n_batch=0,
            input_normalization='standardization',
            seed=1)

        d = _get_data(file_name)

        epoch_idx = d[:, 0]
        train_rmse = d[:, 2]
        test_rmse = d[:, 3]

        results.append(np.min(test_rmse))
        plt.plot(epoch_idx, train_rmse)
        plt.plot(epoch_idx, test_rmse)

    file_name = get_file_name(
        n_hidden_layer=2,
        n_hidden_unit=50,
        activation_fn=activation_fn,
        optimizer='adam',
        learning_rate=1e-3,
        dropout_rate=dropout_rate,
        dropout_mode='all',
        max_epoch=20000,
        n_batch=0,
        input_normalization='standardization',
        seed=1)
    plt.ylim([2, 4])
    plt.savefig(
        '.results/dropout/boston/epoch-figures/{}.learning-curve.png'.format(
            file_name),
        format='png')
    plt.close()
    return results


def main(activation_fn):
    min_test_rmses_by_dropout_rate = []
    for dropout_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        min_test_rmse_default, min_test_rmse_sparsity_normalization = _plot(
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        min_test_rmses_by_dropout_rate.append([
            dropout_rate, min_test_rmse_default,
            min_test_rmse_sparsity_normalization
        ])
        print(dropout_rate, min_test_rmse_default,
              min_test_rmse_sparsity_normalization)
    min_test_rmses_by_dropout_rate = np.array(min_test_rmses_by_dropout_rate)

    file_name = get_file_name(
        n_hidden_layer=2,
        n_hidden_unit=50,
        activation_fn=activation_fn,
        optimizer='adam',
        learning_rate=1e-3,
        dropout_rate='all',
        dropout_mode='all',
        max_epoch=20000,
        n_batch=0,
        input_normalization='standardization',
        seed=1)
    plt.plot(min_test_rmses_by_dropout_rate[:, 0],
             min_test_rmses_by_dropout_rate[:, 1])
    plt.plot(min_test_rmses_by_dropout_rate[:, 0],
             min_test_rmses_by_dropout_rate[:, 2])
    plt.savefig(
        '.results/dropout/boston/epoch-figures/{}.min-test-rmse.png'.format(
            file_name),
        format='png')
    plt.close()


if __name__ == '__main__':
    # activation_fn = 'leaky_relu'
    activation_fn = 'sigmoid'
    # activation_fn = 'tanh'
    main(activation_fn=activation_fn)
