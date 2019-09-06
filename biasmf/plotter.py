import math

import numpy as np
from matplotlib import pyplot as plt

from common.datasets.movielens import Dataset


def _get_count_dict(count_dict, train_data, use_user_vector=False):
    for i in range(train_data.shape[0]):
        _data = train_data[i]
        user_id = int(_data[0])
        item_id = int(_data[1])
        if use_user_vector:
            key = user_id
        else:
            key = item_id
        c = count_dict.get(key, 0)
        count_dict[key] = c + 1
    return count_dict


def get_name(normalize_input, standardization, lambda_w, K_prime,
             activation_fn):
    return '{}_{}_{}_{}_{}'.format(normalize_input, standardization, lambda_w,
                                   K_prime, activation_fn)


def _plot(use_user_vector, n_data):
    dataset_name = 'movielens-100k'
    # dataset_name = 'movielens-1m'
    # dataset_name = 'movielens-10m'
    dataset = Dataset(dataset_name, Dataset.SPLIT_DEFAULT_10_PERCENT, seed=1)
    train_data = dataset.train_data
    valid_data = dataset.valid_data
    test_data = dataset.test_data

    count_dict = _get_count_dict(
        {}, train_data, use_user_vector=use_user_vector)

    key = 'u' if use_user_vector else 'i'

    b = np.load('biasmf/weights/{}/b_{}.npy'.format(dataset_name, key))
    results = np.array([[count_dict.get(i, 0), b[i]] for i in range(n_data)])
    plt.scatter(results[:, 0], results[:, 1], s=1)
    # plt.ylim([1, 5])
    plt.savefig(
        'biasmf/figures/b_{key}-{}.png'.format(dataset_name, key=key),
        format='png')
    plt.close()


def main():
    _plot(False, 1682)
    _plot(True, 943)


if __name__ == '__main__':
    main()
