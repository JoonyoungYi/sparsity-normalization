import random

import tensorflow as tf
from tensorflow import keras
import numpy as np

from .model import init_models


def _init_x_and_y(args):
    # load data
    fashion_mnist = keras.datasets.fashion_mnist
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

    # normalize
    train_x = train_x.reshape([-1, 28 * 28]).astype('float32') / 255
    test_x = test_x.reshape([-1, 28 * 28]).astype('float32') / 255

    # split
    np.random.seed(seed=args.data_seed)
    perm = np.random.permutation(train_x.shape[0])
    train_x = train_x[perm]
    train_y = train_y[perm]

    return train_x[:50000], train_y[:50000], train_x[50000:], train_y[
        50000:], test_x, test_y


def train(args):
    train_x, train_y, valid_x, valid_y, test_x, test_y = _init_x_and_y(args)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
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

        if max_test_acc + 1e-5 < test_acc:
            max_test_acc = test_acc
            max_test_iter_idx = iter_idx

        if iter_idx > max_valid_iter_idx + args.epoch_early_stop:
            break


def main():
    import argparse

    parser = argparse.ArgumentParser(description='FashionMNIST')
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
        default=100,
        help='the number of hidden unit of each layer.')
    parser.add_argument(
        '--activation_fn',
        type=str,
        default='tanh',
        choices=['relu', 'tanh'],
        help='the activation function')
    parser.add_argument(
        '--data_seed', type=int, default=1, help='the seed for dataset')
    parser.add_argument(
        '--drop_rate', type=float, default=0.0, help='drop rate.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='learning rate of optimizer')

    parser.add_argument(
        '--normalize',
        type=bool,
        default=False,
        help='Whether normalize input layer by our method.')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':

    main()
