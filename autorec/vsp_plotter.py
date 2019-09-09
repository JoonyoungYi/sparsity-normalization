import math
import random
import pickle

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from .batch import Batch
from .plotter import get_name
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


def _forward(x, weights, biases, activation_fn):
    assert len(weights) == 2
    assert len(biases) == 2
    layer = x
    layer = np.matmul(layer, weights[0]) + biases[0]
    layer = activation_fn(layer)
    layer = np.matmul(layer, weights[1]) + biases[1]
    return layer


def _plot(normalize_input,
          standardization,
          lambda_w,
          K_prime,
          activation_fn,
          use_user_vector,
          target_idx=0,
          clipping=True):
    dataset = Dataset(
        'movielens-100k', Dataset.SPLIT_DEFAULT_10_PERCENT, seed=1)
    train_data = dataset.train_data
    valid_data = dataset.valid_data
    test_data = dataset.test_data

    name = get_name(
        normalize_input,
        standardization,
        lambda_w,
        K_prime,
        activation_fn,
        use_user_vector,
        clipping=clipping)

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
    # print(target_column_index)
    # input('')

    T = 100
    xs, ys = [], []
    d = {}
    for _ in tqdm(range(T)):
        if activation_fn == 'sigmoid':
            activation_f = lambda x: 1 / (1 + np.exp(-x))
        elif activation_fn == 'tanh':
            activation_f = lambda x: np.tanh(x)
        elif activation_fn == 'elu':
            activation_f = lambda x: np.where(x > 0, x, np.exp(x) - 1)
        elif activation_fn == 'identity':
            activation_f = lambda x: x
        elif activation_fn == 'relu':
            activation_f = lambda x: np.clip(x, a_min=0, a_max=None)
        elif activation_fn == 'relu6':
            activation_f = lambda x: np.clip(x, a_min=0, a_max=6)
        elif activation_fn == 'leaky_relu':
            activation_f = lambda x: np.where(x > 0, x, x * 0.2)
        elif activation_fn == 'softplus':
            activation_f = lambda x: np.log(np.exp(x) + 1)
        else:
            # print(activation_fn)
            raise NotImplementedError()

        train_ratings = train_rating_dict.get(target_idx, None)
        if not train_ratings:
            continue

        random.shuffle(train_ratings)
        for n_known_entry in range(1, len(train_ratings)):
            _batch_ratings = train_ratings[:n_known_entry]
            x = np.zeros(n_feature)
            for i, r in _batch_ratings:
                x[i] = r

            if normalize_input:
                x = x * K_prime / n_known_entry

            y_hat = _forward(x, weights, biases, activation_fn=activation_f)

            # s = result_dict.get(n_known_entry, [])
            # # s.append(np.mean(y_hat))
            # s.append(y_hat[target_column_index])
            # result_dict[n_known_entry] = s
            xs.append(n_known_entry)
            ys.append(y_hat[target_column_index])

            _hats = d.get(n_known_entry, [])
            _hats.append(y_hat[target_column_index])
            d[n_known_entry] = _hats

    plt.scatter(xs, ys, s=1, label='w/ SN' if normalize_input else 'w/o SN')
    plt.axhline(y=target_rating, color='g', linestyle='-')

    results = np.array([[key, sum(item) / len(item)]
                        for key, item in sorted(d.items())])
    plt.plot(results[:, 0], results[:, 1], "r-")

    plt.ylim([0.5, 5.5])
    plt.legend()
    plt.savefig(
        'autorec/vsp/T{T}_usevec{use_user_vector}-target{target_idx}-act{activation_fn}-clip{clipping}-sn{sn}-kprime{k_prime}.png'.
        format(
            T=T,
            sn=normalize_input,
            activation_fn=activation_fn,
            clipping=clipping,
            use_user_vector=use_user_vector,
            k_prime=K_prime,
            target_idx=target_idx).lower(),
        format='png')
    plt.close()


def main():
    # for target_idx in range(10):
    #     _plot(
    #         False,
    #         False,
    #         lambda_w=300.0,
    #         K_prime=500,
    #         activation_fn='sigmoid',
    #         use_user_vector=True,
    #         target_idx=target_idx,
    #         clipping=False)
    #     _plot(
    #         True,
    #         False,
    #         lambda_w=600.0,
    #         K_prime=500,
    #         activation_fn='sigmoid',
    #         use_user_vector=True,
    #         target_idx=target_idx,
    #         clipping=False)
    #
    #     activation_fn = 'relu'
    #     use_user_vector = True
    #     _plot(
    #         False,
    #         False,
    #         lambda_w=2000.0,
    #         K_prime=500.0,
    #         activation_fn=activation_fn,
    #         use_user_vector=use_user_vector,
    #         target_idx=target_idx,
    #         clipping=False)
    #     _plot(
    #         True,
    #         False,
    #         lambda_w=8000.0,
    #         K_prime=500.0,
    #         activation_fn=activation_fn,
    #         use_user_vector=use_user_vector,
    #         target_idx=target_idx,
    #         clipping=False)

    for target_idx in range(10):
        activation_fn = 'identity'
        _plot(
            False,
            False,
            lambda_w=2000.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=8000.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=800.0,
            K_prime=52.44,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)

        activation_fn = 'leaky_relu'
        _plot(
            False,
            False,
            lambda_w=2000.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=8000.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=900.0,
            K_prime=52.44,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)

        activation_fn = 'relu'
        _plot(
            False,
            False,
            lambda_w=2000.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=8000.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=900.0,
            K_prime=52.44,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)

        activation_fn = 'relu6'
        _plot(
            False,
            False,
            lambda_w=2000.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=7000.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=800.0,
            K_prime=52.44,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)

        activation_fn = 'elu'
        _plot(
            False,
            False,
            lambda_w=2000.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=800.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=1000.0,
            K_prime=52.44,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)

        activation_fn = 'sigmoid'
        _plot(
            False,
            False,
            lambda_w=300.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=500.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=200.0,
            K_prime=52.44,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)

        activation_fn = 'tanh'
        _plot(
            False,
            False,
            lambda_w=800.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=1000.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=500.0,
            K_prime=52.44,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)

        activation_fn = 'softplus'
        _plot(
            False,
            False,
            lambda_w=1000.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=7000.0,
            K_prime=500.0,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)
        _plot(
            True,
            False,
            lambda_w=700.0,
            K_prime=52.44,
            activation_fn=activation_fn,
            use_user_vector=True,
            target_idx=target_idx,
            clipping=True)

        # break


if __name__ == '__main__':
    main()
