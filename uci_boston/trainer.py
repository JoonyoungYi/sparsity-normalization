import re
import random

import tensorflow as tf
import numpy as np

from .model import init_models


def _init_x_and_y(args):
    # This file comes from "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/"
    f = open('uci_boston/boston.dat', 'r')
    rows = []
    for line in f:
        line = line.strip()
        cols = re.split('\s+', line)
        assert len(cols) == 14

        row = [float(c) for c in cols]
        row.insert(4, 1 - row[3])
        rows.append(row)
    data = np.array(rows)

    # feature-wise normalization min-max=0-1
    feature_min = np.min(data, axis=0)
    feature_max = np.max(data, axis=0)
    for j in range(14):
        data[:, j] = (data[:, j] - feature_min[j]) / (
            feature_max[j] - feature_min[j])

    # set seed
    random.seed(args.data_seed)
    np.random.seed(seed=args.data_seed)

    # sparsity injection
    for i in range(data.shape[0]):
        row = data[i]

        m = None
        while m is None or m[0].shape[0] >= 14:
            if random.random() < 0.5:
                m = np.where(np.random.random(14) < 0)
            else:
                m = np.where(np.random.random(14) < 0.5)

        row[m] = 0
        # if args.normalize:
        #     row[:14] /= (14 - m[0].shape[0])
        data[i] = row

    # split
    perm = np.random.permutation(data.shape[0])
    data = data[perm]

    train_data = data[:300]
    valid_data = data[300:400]
    test_data = data[400:]

    train_x, train_y = train_data[:, :14], train_data[:, 14]
    valid_x, valid_y = valid_data[:, :14], valid_data[:, 14]
    test_x, test_y = test_data[:, :14], test_data[:, 14]

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def train(args):
    train_x, train_y, valid_x, valid_y, test_x, test_y = _init_x_and_y(args)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models(args)
    session.run(tf.global_variables_initializer())

    max_valid_acc = float('inf')
    max_valid_iter_idx = 0
    final_test_acc = float('inf')

    max_test_acc = float('inf')
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

        if max_valid_acc - 1e-5 > valid_acc:
            max_valid_acc = valid_acc
            max_valid_iter_idx = iter_idx
            final_test_acc = test_acc

        if max_test_acc - 1e-5 > test_acc:
            max_test_acc = test_acc
            max_test_iter_idx = iter_idx

        if iter_idx > max_valid_iter_idx + args.epoch_early_stop:
            break


def main():
    import argparse

    parser = argparse.ArgumentParser(description='UCI-Boston-Housing')
    parser.add_argument(
        '--max_epoch', type=int, default=1000000, help='Max epoch')
    parser.add_argument(
        '--epoch_early_stop',
        type=int,
        default=10000,
        help='the number of epochation for early stop.')
    parser.add_argument(
        '--n_hidden_layer',
        type=int,
        default=3,
        help='the number of hidden layers.')
    parser.add_argument(
        '--n_hidden_unit',
        type=int,
        default=500,
        help='the number of hidden unit of each layer.')
    parser.add_argument(
        '--activation_fn',
        type=str,
        # default='relu',
        # default='tanh',
        default='sigmoid',
        choices=['relu', 'tanh', 'sigmoid'],
        help='the activation function')
    parser.add_argument(
        '--data_seed', type=int, default=1, help='the seed for dataset')
    parser.add_argument(
        '--drop_rate', type=float, default=0.5, help='drop rate.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='learning rate of optimizer')

    parser.add_argument(
        '--normalize',
        type=bool,
        default=True,
        # default=False,
        help='Whether normalize input layer by our method.')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
