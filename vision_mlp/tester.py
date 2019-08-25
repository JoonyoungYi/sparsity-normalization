import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import cosine

from common.utils import append_text_results

CODE_NAME = 'vision_mlp'
DATASET = 'mnist'
# DATASET = 'fashion'
# DATASET = 'cifar10'
# DATASET = 'cifar100'
# IS_TRAIN = True
IS_TRAIN = False


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    # x = np.exp(x)
    return x


def _load_activations(sparsity_normalization, is_train=IS_TRAIN):
    with open('.results/vision_mlp/{dataset}/activations/{}_w{}-sn.pickle'.
              format(
                  'train' if is_train else 'test',
                  '' if sparsity_normalization else 'o',
                  dataset=DATASET), 'rb') as f:
        data = pickle.load(f)
    assert len(data) == 3
    return data


def _load_test_x_or_y(x_or_y, is_train=IS_TRAIN):
    with open('.results/vision_mlp/{dataset}/{}_{}s/1.pickle'.format(
            'train'
            if is_train else 'test', x_or_y, dataset=DATASET), 'rb') as f:
        data = pickle.load(f)
    return data


def _plot(sparse_mean_activation,
          dense_mean_activation,
          class_idx,
          layer_idx,
          sparsity_normalization,
          bin_num=100):
    ranges = (min(sparse_mean_activation.min(), dense_mean_activation.min()),
              max(sparse_mean_activation.max(), dense_mean_activation.max()))
    # print(ranges)
    bins = [
        ranges[0] + (ranges[1] - ranges[0]) / bin_num * i
        for i in range(bin_num)
    ]

    plt.hist(
        sparse_mean_activation, color="#3F51B5AA", range=ranges, bins=bins)
    plt.hist(dense_mean_activation, color="#F44336AA", range=ranges, bins=bins)
    plt.savefig(
        '.results/vision_mlp/{dataset}/activations/hist-{}-{}-w{}-sn.png'.
        format(
            layer_idx,
            class_idx,
            '' if sparsity_normalization else 'o',
            dataset=DATASET),
        format='png')
    plt.close()

    sparse_hist, _ = np.histogram(
        sparse_mean_activation, bins=bins, density=True)
    dense_hist, _ = np.histogram(
        dense_mean_activation, bins=bins, density=True)

    overlap_portion = np.sum(
        np.minimum(sparse_hist, dense_hist)) / np.sum(sparse_hist)
    print(layer_idx, class_idx, overlap_portion)

    results = [layer_idx, class_idx, overlap_portion]
    append_text_results(CODE_NAME, DATASET, 'activations',
                        'w{}-sn'.format('' if sparsity_normalization else 'o'),
                        '\t'.join(str(t) for t in results))


def _get_mean_activation(activation):
    # print(activation.shape)
    mean_activation = np.mean(activation, axis=1)
    # mean_activation = activation[:, 0]
    # print(mean_activation.shape)
    # input('')
    return mean_activation


def main(sparsity_normalization):
    print('')
    print(sparsity_normalization)
    activations = _load_activations(sparsity_normalization)

    test_x = _load_test_x_or_y('x')
    test_y = _load_test_x_or_y('y')
    # print(test_x)
    test_x_l0 = np.count_nonzero(test_x, axis=1)

    d = {}
    for class_idx in range(10):
        _l0 = test_x_l0[test_y == class_idx]
        _x = test_x[test_y == class_idx]
        _y = test_y[test_y == class_idx]
        _activations = [a[test_y == class_idx] for a in activations]

        _sorted_idx = np.argsort(_l0)
        _x = _x[_sorted_idx]
        _y = _y[_sorted_idx]
        _activations = [a[_sorted_idx] for a in _activations]

        split_idx = _sorted_idx.shape[0] // 3
        _sparse_x, _dense_x = _x[:split_idx], _x[-split_idx:]
        _sparse_y, _dense_y = _y[:split_idx], _y[-split_idx:]
        _sparse_activations = [a[:split_idx] for a in _activations]
        _dense_activations = [a[-split_idx:] for a in _activations]

        d[class_idx] = {
            'sparse_x': _sparse_x,
            'dense_x': _dense_x,
            # 'sparse_y': _sparse_y,
            # 'dense_y': _dense_y,
            'sparse_activations': _sparse_activations,
            'dense_activations': _dense_activations,
        }

    for class_idx in range(10):
        _sparse_activation = d[class_idx]['sparse_activations'][-1]
        _dense_activation = d[class_idx]['dense_activations'][-1]
        sparse_mean_activation = softmax(_sparse_activation)[:, class_idx]
        dense_mean_activation = softmax(_dense_activation)[:, class_idx]

        _plot(sparse_mean_activation, dense_mean_activation, class_idx, -2,
              sparsity_normalization, bin_num=20)
        print(np.mean(sparse_mean_activation), np.mean(dense_mean_activation))

    return

    # for layer_idx in range(3):
    #
    #     _sparse_activation = np.concatenate(
    #         [
    #             d[class_idx]['sparse_activations'][layer_idx]
    #             for class_idx in range(10)
    #         ],
    #         axis=0)
    #     _dense_activation = np.concatenate(
    #         [
    #             d[class_idx]['dense_activations'][layer_idx]
    #             for class_idx in range(10)
    #         ],
    #         axis=0)
    #     sparse_mean_activation = _get_mean_activation(_sparse_activation)
    #     dense_mean_activation = _get_mean_activation(_dense_activation)
    #     _plot(
    #         sparse_mean_activation,
    #         dense_mean_activation,
    #         -1,
    #         layer_idx,
    #         sparsity_normalization,
    #         bin_num=20)
    #
    #     for class_idx in range(10):
    #         _sparse_activation = d[class_idx]['sparse_activations'][layer_idx]
    #         _dense_activation = d[class_idx]['dense_activations'][layer_idx]
    #
    #         # _sparse_mean = np.mean(_sparse_activation, axis=0)
    #         # _dense_mean = np.mean(_dense_activation, axis=0)
    #         # # dist = cosine(_sparse_mean, _dense_mean)
    #         # dist = np.linalg.norm(_sparse_mean - _dense_mean)
    #         # print(layer_idx, class_idx, dist)
    #
    #         sparse_mean_activation = _get_mean_activation(_sparse_activation)
    #         dense_mean_activation = _get_mean_activation(_dense_activation)
    #         _plot(sparse_mean_activation, dense_mean_activation, class_idx,
    #               layer_idx, sparsity_normalization)


if __name__ == '__main__':
    main(True)
    main(False)
