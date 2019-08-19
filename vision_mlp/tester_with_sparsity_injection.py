import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.covariance import EmpiricalCovariance

from common.utils import append_text_results

CODE_NAME = 'vision_mlp'


def _load_activations(sparsity_normalization, sparse_or_dense):
    assert sparse_or_dense in ['sparse', 'dense']

    with open('.results/vision_mlp/mnist/activations/{}_test_w{}-sn.pickle'.
              format(sparse_or_dense, ''
                     if sparsity_normalization else 'o'), 'rb') as f:
        data = pickle.load(f)

    assert len(data) == 3
    return data


def _load_test_y(sparse_or_dense):
    assert sparse_or_dense in ['sparse', 'dense']
    with open('.results/vision_mlp/mnist/{}_test_ys/1.pickle'.format(
            sparse_or_dense), 'rb') as f:
        data = pickle.load(f)
    return data


# def _calculate_mean_activations(activations):
#     return [np.mean(a, axis=0) for a in activations]

# def _plot_heatmap(sparsity_normalization, activation, layer_idx):
#     fig, axs = plt.subplots(1, 1, sharey=False, tight_layout=True)
#     ax = axs
#     activation = (activation - activation.min()) / (
#         activation.max() - activation.min())
#     # if activation.shape[1] > 10:
#     #     activation = activation[:, :100]
#     ax.imshow(activation, cmap='hot', interpolation='nearest')
#     fig.savefig(
#         '.results/vision_mlp/mnist/activations/heatmap-{}-w{}-sn.png'.format(
#             layer_idx, '' if sparsity_normalization else 'o'),
#         format='png')
#
#
# def _plot_tsne(sparsity_normalization,
#                sparse_activation,
#                dense_activation,
#                sparse_test_y,
#                dense_test_y,
#                layer_idx,
#                digit=2):
#
#     feature = np.concatenate(
#         [
#             sparse_activation[sparse_test_y == 8][:100],
#             dense_activation[dense_test_y == 8][:100],
#             sparse_activation[sparse_test_y == 6][:100],
#             dense_activation[dense_test_y == 6][:100]
#         ],
#         axis=0)
#
#     feature = (feature - np.mean(feature, axis=0, keepdims=True)) / (
#         1e-8 + np.std(feature, axis=0, keepdims=True))
#     # input('')
#
#     # print(feature.shape)
#
#     model = TSNE(learning_rate=100, verbose=1)
#     transformed = model.fit_transform(feature)
#     print('finish transform!')
#     xs = transformed[:, 0]
#     ys = transformed[:, 1]
#     labels = [i // 100 for i in range(400)]
#     plt.scatter(xs, ys, c=labels)  # , c=labels
#     plt.savefig(
#         '.results/vision_mlp/mnist/activations/tsne-{}-w{}-sn.png'.format(
#             layer_idx, '' if sparsity_normalization else 'o'),
#         format='png')
#     plt.close()


def _plot(sparse_mean_activation, dense_mean_activation, class_idx, layer_idx,
          sparsity_normalization):
    ranges = (min(sparse_mean_activation.min(), dense_mean_activation.min()),
              max(sparse_mean_activation.max(), dense_mean_activation.max()))
    # print(ranges)
    bin_num = 100
    bins = [
        ranges[0] + (ranges[1] - ranges[0]) / bin_num * i
        for i in range(bin_num)
    ]

    plt.hist(
        sparse_mean_activation, color="#3F51B5AA", range=ranges, bins=bins)
    plt.hist(dense_mean_activation, color="#F44336AA", range=ranges, bins=bins)
    plt.savefig(
        '.results/vision_mlp/mnist/activations/hist-{}-{}-w{}-sn.png'.format(
            layer_idx, class_idx, '' if sparsity_normalization else 'o'),
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
    append_text_results(CODE_NAME, 'mnist', 'activations',
                        'w{}-sn'.format('' if sparsity_normalization else 'o'),
                        '\t'.join(str(t) for t in results))


def main(sparsity_normalization):
    print('')
    print(sparsity_normalization)
    sparse_activations_without_sn = _load_activations(sparsity_normalization,
                                                      'sparse')
    dense_activations_without_sn = _load_activations(sparsity_normalization,
                                                     'dense')

    sparse_test_y = _load_test_y('sparse')
    dense_test_y = _load_test_y('dense')

    for layer_idx, (sparse_activation, dense_activation) in enumerate(
            zip(sparse_activations_without_sn, dense_activations_without_sn)):

        _sparse_activation = sparse_activation
        _dense_activation = dense_activation
        _activation = np.concatenate(
            [_sparse_activation, _dense_activation], axis=0)

        sparse_mean_activation = np.mean(_sparse_activation, axis=1)
        dense_mean_activation = np.mean(_dense_activation, axis=1)
        # mean_activation = np.mean(_activation, axis=1)
        _plot(sparse_mean_activation, dense_mean_activation, -1, layer_idx,
              sparsity_normalization)

        for class_idx in range(10):
            _sparse_activation = sparse_activation[sparse_test_y == class_idx]
            _dense_activation = dense_activation[dense_test_y == class_idx]
            # _activation = np.concatenate(
            #     [_sparse_activation, _dense_activation], axis=0)

            sparse_mean_activation = np.mean(_sparse_activation, axis=1)
            dense_mean_activation = np.mean(_dense_activation, axis=1)
            # mean_activation = np.mean(_activation, axis=1)
            _plot(sparse_mean_activation, dense_mean_activation, class_idx,
                  layer_idx, sparsity_normalization)

        # print(sparse_mean_activation, mean_activation, dense_mean_activation)

        # print(np.sum(np.std(_sparse_activation, axis=0)))
        # print(np.sum(np.std(_activation, axis=0)))
        # print(np.sum(np.std(_dense_activation, axis=0)))

        # _plot_heatmap(sparsity_normalization, mean_activation, layer_idx)

        # mus = []
        # sigmas = []
        # precisions = []
        # for class_idx in range(10):
        #     _sparse_activation = sparse_activation[sparse_test_y == class_idx]
        #     _dense_activation = dense_activation[dense_test_y == class_idx]
        #
        #     _activation = np.concatenate(
        #         [_sparse_activation, _dense_activation], axis=0)
        #
        #     cov = EmpiricalCovariance().fit(_activation)
        #     mu = cov.location_
        #     sigma = cov.covariance_
        #     precision = cov.precision_
        #     # mu = np.mean(_activation, axis=0)
        #     # sigma = np.std(_activation, axis=0)
        #     # sigma = np.matmul(sigma, sigma.T)
        #     # sigma = np.cov(_activation.T)
        #     mus.append(mu)
        #     sigmas.append(sigma)
        #     precisions.append(precision)
        #
        # for left_class_idx in range(10):
        #     for right_class_idx in range(left_class_idx + 1, 10):
        #         left_mu = mus[left_class_idx]
        #         right_mu = mus[right_class_idx]
        #         left_sigma = sigmas[left_class_idx]
        #         right_sigma = sigmas[right_class_idx]
        #
        #         # sigma = (left_sigma + right_sigma) / 2
        #
        #         print(np.linalg.det(left_sigma))
        #         print(np.linalg.det(right_sigma))
        #
        #         # left_distance = np.matmul(
        #         #     np.matmul((left_mu - right_mu).reshape([1, -1]),
        #         #               np.linalg.inv(sigma)),
        #         #     (left_mu - right_mu).reshape([-1, 1]))
        #         # print(left_distance)
        #         #
        #         # bhattacharyya_distance = (1 / 8) * left_distance
        #         # # print(left_class_idx, right_class_idx)
        #         # print((left_mu - right_mu).shape)
        #
        #         input(';')
        # break

        # for layer_idx, (sparse_activation, dense_activation) in enumerate(
        #         zip(sparse_activations_without_sn, dense_activations_without_sn)):
        #     print('>>', layer_idx)
        #     # print(np.mean(np.count_nonzero(sparse_activation, axis=1)))
        #     # print(np.mean(np.count_nonzero(dense_activation, axis=1)))
        #
        #     _plot_tsne(sparsity_normalization, sparse_activation, dense_activation,
        #                sparse_test_y, dense_test_y, layer_idx)

        # sparse_mean_activations_without_sn = _calculate_mean_activations(
        #     sparse_activations_without_sn)
        # dense_mean_activations_without_sn = _calculate_mean_activations(
        #     dense_activations_without_sn)
        #
        # for layer_idx, (sparse_mean_activation,
        #                 dense_mean_activation) in enumerate(
        #                     zip(sparse_mean_activations_without_sn,
        #                         dense_mean_activations_without_sn)):
        #     mean_activation = np.stack(
        #         [sparse_mean_activation, dense_mean_activation], axis=0)
        #     _plot_heatmap(sparsity_normalization, mean_activation, layer_idx)


if __name__ == '__main__':
    main(True)
    main(False)
