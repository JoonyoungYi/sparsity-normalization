import math
import random
import pickle

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from .batch import Batch
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
             activation_fn, use_user_vector):
    return '{}_{}_{}_{}_{}_{}'.format(normalize_input, standardization,
                                      lambda_w, K_prime, activation_fn,
                                      use_user_vector)


def _plot_dataset(use_user_vector=False):
    # dataset_name = 'movielens-100k'
    # dataset_name = 'movielens-1m'
    dataset_name = 'movielens-10m'
    dataset = Dataset(dataset_name, Dataset.SPLIT_DEFAULT_10_PERCENT, seed=1)
    train_data = dataset.train_data
    valid_data = dataset.valid_data
    test_data = dataset.test_data

    count_dict = _get_count_dict({}, train_data, use_user_vector)
    # count_dict = _get_count_dict(count_dict, valid_data)

    rating_dict = {}
    for i in range(train_data.shape[0]):
        _data = train_data[i]
        user_id = int(_data[0])
        item_id = int(_data[1])
        rating = float(_data[2])

        ratings = rating_dict.get(item_id, [])
        ratings.append(rating)
        rating_dict[item_id] = ratings

    n_data = max(count_dict.keys()) + 1

    results = np.array([[
        count_dict.get(i, 0),
        sum(rating_dict.get(i, [])) / len(rating_dict.get(i, []))
        if len(rating_dict.get(i, [])) else -1
    ] for i in range(n_data)])
    plt.scatter(results[:, 0], results[:, 1], s=1)
    plt.ylim([1, 5])
    plt.savefig(
        'autorec/figures/rating-{}-{}.png'.format(dataset_name,
                                                  use_user_vector),
        format='png')
    plt.close()


def _forward(x, weights, biases, activation_fn):
    assert len(weights) == 2
    assert len(biases) == 2

    layer = x
    layer = np.matmul(layer, weights[0]) + biases[0]
    layer = activation_fn(layer)
    layer = np.matmul(layer, weights[1]) + biases[1]
    return layer


def _plot(normalize_input, standardization, lambda_w, K_prime, activation_fn,
          use_user_vector):
    dataset = Dataset(
        'movielens-100k', Dataset.SPLIT_DEFAULT_10_PERCENT, seed=1)
    train_data = dataset.train_data
    valid_data = dataset.valid_data
    test_data = dataset.test_data

    name = get_name(normalize_input, standardization, lambda_w, K_prime,
                    activation_fn, use_user_vector)

    # l0 와 mean activation의 관계 scatter
    count_dict = _get_count_dict(
        {}, train_data, use_user_vector=use_user_vector)
    count_dict = _get_count_dict(
        count_dict, valid_data, use_user_vector=use_user_vector)

    if use_user_vector:
        n_data = max(np.max(train_data[:, 0]), np.max(valid_data[:, 0])) + 1
        n_feature = max(np.max(train_data[:, 1]), np.max(valid_data[:, 1])) + 1
    else:
        n_data = max(np.max(train_data[:, 1]), np.max(valid_data[:, 1])) + 1
        n_feature = max(np.max(train_data[:, 0]), np.max(valid_data[:, 0])) + 1
    n_data, n_feature = int(n_data), int(n_feature)

    c = np.array([count_dict.get(i, 0) for i in range(n_data)])
    # print(c.shape)
    with open('autorec/weights/{}.pickle'.format(name), 'rb') as f:
        test_weights = pickle.load(f)
    test_y_hat = test_weights['test_y_hat']
    test_y_hat_mean = np.mean(test_y_hat, axis=1)
    # print(test_y_hat_mean.shape)
    plt.scatter(c, test_y_hat_mean, s=1)
    plt.ylim([1, 5])
    plt.savefig(
        'autorec/figures/mean-activation-scatter-{}.png'.format(name),
        format='png')
    plt.close()

    # 각 count 끼리 모아서 plot
    mean_dict = {}
    for i in range(c.shape[0]):
        s = mean_dict.get(c[i], [])
        s.append(test_y_hat_mean[i])
        mean_dict[c[i]] = s
    xs = []
    ys = []
    for key, item in mean_dict.items():
        xs.append(key)
        ys.append(sum(item) / len(item))
    plt.scatter(xs, ys, s=1)
    plt.ylim([1, 5])
    plt.savefig(
        'autorec/figures/mean-activation-mean-{}.png'.format(name),
        format='png')
    plt.close()

    # residual plotting
    target_test_y_count = test_data[:, 1].copy()
    target_test_y_hat = test_data[:, 2].copy()
    for i in range(test_data.shape[0]):
        _data = test_data[i]
        user_id = int(_data[0])
        item_id = int(_data[1])
        rating = float(_data[2])
        if use_user_vector:
            _y_hat = test_y_hat[user_id][item_id]
        else:
            _y_hat = test_y_hat[item_id][user_id]

        if use_user_vector:
            target = user_id
        else:
            target = item_id
        target_test_y_hat[i] = _y_hat - rating
        target_test_y_count[i] = c[target]

    plt.scatter(target_test_y_count, target_test_y_hat, s=1)
    plt.ylim([-3, 3])
    plt.savefig('autorec/figures/residual-{}.png'.format(name), format='png')
    plt.close()

    # error plotting
    error_dict = {}
    target_activation_dict = {}
    for i in range(test_data.shape[0]):
        _data = test_data[i]
        user_id = int(_data[0])
        item_id = int(_data[1])
        rating = float(_data[2])
        if use_user_vector:
            _y_hat = test_y_hat[user_id][item_id]
        else:
            _y_hat = test_y_hat[item_id][user_id]

        if use_user_vector:
            target = user_id
        else:
            target = item_id

        error = (_y_hat - rating)**2
        errors = error_dict.get(target, [])
        errors.append(error)
        error_dict[item_id] = errors

        target_activations = target_activation_dict.get(target, [])
        target_activations.append(_y_hat)
        target_activation_dict[target] = target_activations

    results = np.array([[
        count_dict.get(i, 0),
        math.sqrt(sum(error_dict.get(i, [])) / len(error_dict.get(i, [])))
        if len(error_dict.get(i, [])) else -1
    ] for i in range(n_data)])
    plt.scatter(results[:, 0], results[:, 1], s=1)
    plt.ylim([0, 2])
    plt.savefig('autorec/figures/rmse-{}.png'.format(name), format='png')
    plt.close()

    results = np.array([[
        count_dict.get(i, 0),
        sum(target_activation_dict.get(i, [])) /
        len(target_activation_dict.get(i, []))
        if len(target_activation_dict.get(i, [])) else -1
    ] for i in range(n_data)])
    plt.scatter(results[:, 0], results[:, 1], s=1)
    plt.ylim([1, 5])
    plt.savefig(
        'autorec/figures/target-activation-{}.png'.format(name), format='png')
    plt.close()

    # 1개씩 train 갯수 늘려가며 찍어보기
    weights = test_weights['weights']
    biases = test_weights['biases']

    train_rating_dict = {}
    for i in range(train_data.shape[0]):
        _data = train_data[i]
        user_id = int(_data[0])
        item_id = int(_data[1])
        rating = float(_data[2])

        if use_user_vector:
            ratings = train_rating_dict.get(user_id, [])
            ratings.append((item_id, rating))
            train_rating_dict[user_id] = ratings
        else:
            ratings = train_rating_dict.get(item_id, [])
            ratings.append((user_id, rating))
            train_rating_dict[item_id] = ratings

    target_idx = 0

    test_ratings = []
    for i in range(test_data.shape[0]):
        _data = test_data[i]
        user_id = int(_data[0])
        item_id = int(_data[1])
        rating = float(_data[2])

        if user_id == target_idx:
            test_ratings.append((item_id, rating))
    # print(test_ratings)
    # input('')
    assert test_ratings
    target_column_index, target_rating = test_ratings[0]
    print(target_column_index)
    # input('')

    result_dict = {}
    # for key in tqdm(range(n_data)):
    for key in tqdm(range(target_idx, target_idx + 1)):
        if activation_fn == 'sigmoid':
            activation_f = lambda x: 1 / (1 + np.exp(-x))
        elif activation_fn == 'identity':
            activation_f = lambda x: x
        else:
            # print(activation_fn)
            raise NotImplementedError()

        train_ratings = train_rating_dict.get(key, None)
        if not train_ratings:
            continue

        random.shuffle(train_ratings)
        for n_known_entry in range(1, len(train_ratings)):
            _batch_ratings = train_ratings[:n_known_entry]
            x = np.zeros(n_feature)
            for i, r in _batch_ratings:
                x[i] = r
            y_hat = _forward(x, weights, biases, activation_fn=activation_f)

            s = result_dict.get(n_known_entry, [])
            # s.append(np.mean(y_hat))
            s.append(y_hat[target_column_index])
            result_dict[n_known_entry] = s

    xs, ys = [], []
    for key, item in sorted(result_dict.items()):
        xs.append(key)
        ys.append(sum(item) / len(item))

    plt.scatter(xs, ys, s=1)
    plt.axhline(y=target_rating, color='r', linestyle='-')
    plt.ylim([0.5, 5.5])
    plt.savefig(
        'autorec/figures/activation-increasing-known-entry-{}.png'.format(
            name),
        format='png')
    plt.close()


def main():
    # _plot(
    #     False,
    #     False,
    #     lambda_w=300.0,
    #     K_prime=500,
    #     activation_fn='sigmoid',
    #     use_user_vector=False)
    # _plot(
    #     True,
    #     False,
    #     lambda_w=600.0,
    #     K_prime=500,
    #     activation_fn='sigmoid',
    #     use_user_vector=False)
    # _plot(
    #     True,
    #     False,
    #     lambda_w=200.0,
    #     K_prime=52.44,
    #     activation_fn='sigmoid',
    #     use_user_vector=False)

    _plot(
        False,
        False,
        lambda_w=300.0,
        K_prime=500,
        activation_fn='sigmoid',
        use_user_vector=True)
    _plot(
        True,
        False,
        lambda_w=600.0,
        K_prime=500,
        activation_fn='sigmoid',
        use_user_vector=True)

    # _plot(
    #     True,
    #     False,
    #     lambda_w=0.0,
    #     K_prime=500,
    #     activation_fn='identity',
    #     use_user_vector=False)
    # _plot(
    #     False,
    #     False,
    #     lambda_w=0.0,
    #     K_prime=500,
    #     activation_fn='identity',
    #     use_user_vector=False)
    # _plot(
    #     True,
    #     False,
    #     lambda_w=0.0,
    #     K_prime=500,
    #     activation_fn='identity',
    #     use_user_vector=True)
    # _plot(
    #     False,
    #     False,
    #     lambda_w=0.0,
    #     K_prime=500,
    #     activation_fn='identity',
    #     use_user_vector=True)


if __name__ == '__main__':
    main()
    # _plot_dataset()
