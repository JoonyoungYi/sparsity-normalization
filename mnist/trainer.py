import random

import tensorflow as tf
from tensorflow import keras
import numpy as np

from .model import init_models
from common.utils import str2bool
from common.utils import inject_sparsity
from common.utils import split_sparse_and_dense_group

# https://www.researchgate.net/figure/MLP-Test-Accuracy-on-CIFAR100-dataset_fig4_317679430
# https://www.researchgate.net/figure/MLP-Test-Accuracy-on-CIFAR10-dataset_fig5_317679430


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
    train_x = train_x.reshape([-1, 28 * 28]).astype('float32') / 255
    test_x = test_x.reshape([-1, 28 * 28]).astype('float32') / 255

    # sparsity injection
    train_x = inject_sparsity(train_x)
    test_x = inject_sparsity(test_x)

    # split
    random.seed(args.data_seed)
    np.random.seed(seed=args.data_seed)
    train_perm = np.random.permutation(train_x.shape[0])
    train_x = train_x[train_perm]
    train_y = train_y[train_perm]

    valid_x = train_x[50000:]
    train_x = train_x[:50000]
    valid_y = train_y[50000:]
    train_y = train_y[:50000]

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def train(args, seed):
    set_seed(seed)
    train_x, train_y, valid_x, valid_y, test_x, test_y = _init_x_and_y(args)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models(args)
    session.run(tf.global_variables_initializer())

    max_valid_acc = 0.0
    max_valid_iter_idx = 0
    final_test_acc = 0.0

    max_test_acc = 0.0
    max_test_iter_idx = 0

    sparse_test_x, dense_test_x = split_sparse_and_dense_group(test_x)

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

        if max_test_acc + 1e-5 < test_acc:
            max_test_acc = test_acc
            max_test_iter_idx = iter_idx

        if iter_idx > max_valid_iter_idx + args.epoch_early_stop:
            break


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
        default=300,
        help='the number of epochation for early stop.')
    parser.add_argument(
        '--n_hidden_layer',
        type=int,
        default=2,
        help='the number of hidden layers.')
    parser.add_argument(
        '--n_hidden_unit',
        type=int,
        default=100,
        help='the number of hidden unit of each layer.')
    parser.add_argument(
        '--activation_fn',
        type=str,
        # default='tanh',
        default='relu',
        choices=['relu', 'tanh'],
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
        default=True,
        # default=False,
        help='Whether normalize input layers')

    args = parser.parse_args()
    train(args, args.seed)


if __name__ == '__main__':

    main()
