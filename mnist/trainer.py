import random

import tensorflow as tf
from tensorflow import keras
import numpy as np

from .model import init_models
from common.utils import str2bool
from common.utils import set_seed

# https://www.researchgate.net/figure/MLP-Test-Accuracy-on-CIFAR100-dataset_fig4_317679430
# https://www.researchgate.net/figure/MLP-Test-Accuracy-on-CIFAR10-dataset_fig5_317679430

# def _normalize_input(x):
#     for i in range(x.shape[0]):
#         row = x[i]
#         count_nonzero = max(np.where(row > 1e-8)[0].shape[0], 1)
#         row /= count_nonzero
#     return x


def inject_sparsity(x, args):
    g = np.zeros(x.shape[0])
    if args.inject_sparsity:
        for i in range(x.shape[0]):
            row = x[i]

            m = None
            while m is None or m[0].shape[0] >= x.shape[1]:
                if random.random() < 0.5:
                    m = np.where(np.random.random(x.shape[1]) < 0.2)
                    g[i] = 1
                else:
                    m = np.where(np.random.random(x.shape[1]) < 0.8)
                    g[i] = -1

            row[m] = 0
            if args.normalize_input:
                divider = x.shape[1] - m[0].shape[0]
                assert divider >= 1
                row /= divider
            x[i] = row
    return x, g


def _init_x_and_y(args):
    # load data
    if args.dataset == 'mnist':
        dataset = keras.datasets.mnist
    elif args.dataset == 'fashion':
        dataset = keras.datasets.fashion_mnist
    # elif args.dataset == 'cifar10':
    #     dataset = keras.datasets.cifar10
    else:
        raise NotImplementedError()

    (train_x, train_y), (test_x, test_y) = dataset.load_data()

    # normalize
    train_x = train_x.reshape([-1, 28 * 28]).astype('float32') / 255.0
    test_x = test_x.reshape([-1, 28 * 28]).astype('float32') / 255.0

    # sparsity injection
    train_x, train_g = inject_sparsity(train_x, args)
    test_x, test_g = inject_sparsity(test_x, args)

    # if args.normalize_input:
    #     train_x = _normalize_input(train_x)
    #     test_x = _normalize_input(test_x)

    # split
    train_perm = np.random.permutation(train_x.shape[0])
    train_x = train_x[train_perm]
    train_y = train_y[train_perm]

    valid_x = train_x[50000:]
    train_x = train_x[:50000]
    valid_y = train_y[50000:]
    train_y = train_y[:50000]

    dense_test_x = test_x[test_g > 0.5]
    dense_test_y = test_y[test_g > 0.5]

    sparse_test_x = test_x[test_g < -0.5]
    sparse_test_y = test_y[test_g < -0.5]

    return train_x, train_y, valid_x, valid_y, test_x, test_y, sparse_test_x, sparse_test_y, dense_test_x, dense_test_y


def train(args, seed):
    set_seed(seed)
    train_x, train_y, valid_x, valid_y, test_x, test_y, sparse_test_x, sparse_test_y, dense_test_x, dense_test_y = _init_x_and_y(
        args)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models(args)
    session.run(tf.global_variables_initializer())

    max_valid_acc = 0.0
    max_valid_iter_idx = 0
    final_test_acc = 0.0

    max_test_acc = 0.0
    max_test_iter_idx = 0

    for iter_idx in range(args.max_epoch):
        _, train_loss, train_acc = session.run(
            [models['train_op'], models['loss'], models['accuracy']],
            feed_dict={
                models['x']: train_x,
                models['y']: train_y,
                models['training']: True,
            })

        valid_acc = session.run(
            models['accuracy'],
            feed_dict={
                models['x']: valid_x,
                models['y']: valid_y,
                models['training']: False,
            })
        test_acc = session.run(
            models['accuracy'],
            feed_dict={
                models['x']: test_x,
                models['y']: test_y,
                models['training']: False,
            })

        print('')
        print('%6d' % iter_idx, '%.2f' % train_loss, '%.4f' % train_acc,
              '%.4f' % valid_acc, '%.4f' % test_acc)
        print('%6d' % max_valid_iter_idx, '%.4f' % max_valid_acc,
              '%.4f' % final_test_acc)
        print('%6d' % max_test_iter_idx, '%.4f' % max_test_acc)

        if max_valid_acc + 1e-5 < valid_acc:
            max_valid_acc = valid_acc
            max_valid_iter_idx = iter_idx
            final_test_acc = test_acc

            test_activations = session.run(
                models['activations'],
                feed_dict={
                    models['x']: test_x,
                    models['training']: False,
                })
            test_variances = session.run(
                models['variances'],
                feed_dict={
                    models['x']: test_x,
                    models['training']: False,
                })

            test_sparse_activations, test_sparse_acc = session.run(
                [models['activations'], models['accuracy']],
                feed_dict={
                    models['x']: sparse_test_x,
                    models['y']: sparse_test_y,
                    models['training']: False,
                })
            test_dense_activations, test_dense_acc = session.run(
                [models['activations'], models['accuracy']],
                feed_dict={
                    models['x']: dense_test_x,
                    models['y']: dense_test_y,
                    models['training']: False,
                })

        print(test_sparse_activations)
        print(test_activations)
        # print(test_variances)
        print(test_dense_activations)
        print(
            '{} {} {}'.format(test_sparse_acc, final_test_acc, test_dense_acc))

        if max_test_acc + 1e-5 < test_acc:
            max_test_acc = test_acc
            max_test_iter_idx = iter_idx

        if iter_idx > max_valid_iter_idx + args.epoch_early_stop:
            break

    f = open('results/mnist/{}.txt'.format(args.dataset), 'a')
    results = []
    results.append(args.normalize_hidden)
    results.append(args.use_first_bias)
    results.append(args.n_hidden_unit)
    results.append(test_sparse_acc)
    results.append(final_test_acc)
    results.append(test_dense_acc)
    results.append(test_sparse_activations[-1])
    results.append(test_activations[-1])
    results.append(test_dense_activations[-1])
    line = '\t'.join([str(r) for r in results]) + '\n'
    f.write(line)
    f.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        # default='fashion',
        # default='cifar10',
        choices=['mnist', 'fashion', 'cifar10'],
        help='MNIST or Fashion MNIST or CIFAR-10')
    parser.add_argument(
        '--max_epoch', type=int, default=1000000, help='Max epoch')
    parser.add_argument(
        '--epoch_early_stop',
        type=int,
        default=1000,
        help='the number of epochation for early stop.')
    parser.add_argument(
        '--n_hidden_layer',
        type=int,
        default=2,
        help='the number of hidden layers.')
    parser.add_argument(
        '--n_hidden_unit',
        type=int,
        default=5,
        help='the number of hidden unit of each layer.')
    parser.add_argument(
        '--activation_fn',
        type=str,
        # default='tanh',
        default='relu',
        # default='sigmoid',
        choices=['relu', 'tanh', 'sigmoid'],
        help='the activation function')
    parser.add_argument(
        '--seed', type=int, default=1, help='the seed for training')
    parser.add_argument(
        '--drop_rate', type=float, default=0.0, help='drop rate.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-2,
        help='learning rate of optimizer')
    parser.add_argument(
        '--inject_sparsity',
        type=str2bool,
        default=True,
        # default=False,
        help='Whether inject sparsity.')

    parser.add_argument(
        '--normalize_hidden',
        type=str2bool,
        # default=True,
        default=False,
        help='Whether normalize hidden layers')
    parser.add_argument(
        '--normalize_input',
        type=str2bool,
        # default=True,
        default=False,
        help='Whether normalize input layers')
    parser.add_argument(
        '--use_first_bias',
        type=str2bool,
        default=True,
        # default=False,
        help='')

    args = parser.parse_args()
    train(args, args.seed)


if __name__ == '__main__':
    main()
