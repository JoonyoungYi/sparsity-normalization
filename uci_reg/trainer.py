import re
import random

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston

from .model import init_models
from common.utils import str2bool


def _init_x_and_y(args):
    # https://datascienceschool.net/view-notebook/4b948c4d99c94213bf02bcff2ced88eb/
    if args.dataset == 'california':
        dataset = fetch_california_housing()
    elif args.dataset == 'diabetes':
        dataset = load_diabetes()
    elif args.dataset == 'boston':
        dataset = load_boston()
    else:
        raise NotImplementedError()

    x = dataset.data
    x = np.concatenate([x] * args.column_multiplier, axis=1)
    y = dataset.target

    # feature-wise normalization min-max=0-1
    if args.normalize_feature:
        feature_min = np.min(x, axis=0)
        feature_max = np.max(x, axis=0)
        for j in range(x.shape[1]):
            x[:, j] = (x[:, j] - feature_min[j]) / (
                feature_max[j] - feature_min[j])

    # sparsity injection
    for i in range(x.shape[0]):
        row = x[i]

        m = None
        while m is None or m[0].shape[0] >= x.shape[1]:
            if random.random() < 0.5:
                m = np.where(np.random.random(x.shape[1]) < 0.2)
            else:
                m = np.where(np.random.random(x.shape[1]) < 0.8)

        row[m] = 0
        if args.normalize_input:
            # print(row)
            row /= (x.shape[1] - m[0].shape[0])
            # print(row)
            # input('')
        x[i] = row

    # split
    perm = np.random.permutation(x.shape[0])
    x = x[perm]
    y = y[perm]

    train_x = x[:int(x.shape[0] * 0.6)]
    valid_x = x[int(x.shape[0] * 0.6):int(x.shape[0] * 0.8)]
    test_x = x[int(x.shape[0] * 0.8):]

    train_y = y[:int(x.shape[0] * 0.6)]
    valid_y = y[int(x.shape[0] * 0.6):int(x.shape[0] * 0.8)]
    test_y = y[int(x.shape[0] * 0.8):]

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def _train(args, seed):
    # set seed
    random.seed(seed)
    np.random.seed(seed=seed)
    tf.random.set_random_seed(seed)

    train_x, train_y, valid_x, valid_y, test_x, test_y = _init_x_and_y(args)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models(args, n_feature=train_x.shape[1])
    session.run(tf.global_variables_initializer())

    max_valid_rmse = float('inf')
    max_valid_iter_idx = 0
    final_test_rmse = float('inf')

    max_test_rmse = float('inf')
    max_test_iter_idx = 0

    for iter_idx in range(args.max_epoch):
        _, train_loss, train_rmse = session.run(
            [models['train_op'], models['loss'], models['rmse']],
            feed_dict={
                models['x']: train_x,
                models['y']: train_y,
                models['training']: True,
            })

        valid_rmse = session.run(
            models['rmse'],
            feed_dict={
                models['x']: valid_x,
                models['y']: valid_y,
                models['training']: False,
            })
        test_rmse = session.run(
            models['rmse'],
            feed_dict={
                models['x']: test_x,
                models['y']: test_y,
                models['training']: False,
            })

        print('')
        print('%6d' % iter_idx, '%.2f' % train_loss, '%.4f' % train_rmse,
              '%.4f' % valid_rmse, '%.4f' % test_rmse)
        print('%6d' % max_valid_iter_idx, '%.4f' % max_valid_rmse,
              '%.4f' % final_test_rmse)
        # print('%6d' % max_test_iter_idx, '%.4f' % max_test_rmse)

        if max_valid_rmse - 1e-5 > valid_rmse:
            max_valid_rmse = valid_rmse
            max_valid_iter_idx = iter_idx
            final_test_rmse = test_rmse

        if max_test_rmse - 1e-5 > test_rmse:
            max_test_rmse = test_rmse
            max_test_iter_idx = iter_idx

        if iter_idx > max_valid_iter_idx + args.epoch_early_stop:
            break

    session.close()
    return final_test_rmse


def main():
    import argparse

    parser = argparse.ArgumentParser(description='uci-dataset-regression')
    parser.add_argument(
        '--dataset',
        type=str,
        default='boston',
        # default='california',
        # default='diabetes',
        choices=['boston', 'california', 'diabetes'],
        help='dataset.')
    parser.add_argument(
        '--max_epoch', type=int, default=1000000, help='Max epoch')
    parser.add_argument(
        '--epoch_early_stop',
        type=int,
        default=100,
        help='the number of epochation for early stop.')
    parser.add_argument(
        '--n_hidden_layer',
        type=int,
        default=3,
        help='the number of hidden layers.')
    parser.add_argument(
        '--n_hidden_unit',
        type=int,
        default=100,
        help='the number of hidden unit of each layer.')
    parser.add_argument(
        '--activation_fn',
        type=str,
        default='relu',
        # default='tanh',
        # default='sigmoid',
        choices=['relu', 'tanh', 'sigmoid'],
        help='the activation function')
    parser.add_argument(
        '--seed', type=int, default=1, help='the seed for dataset')
    parser.add_argument(
        '--seeds',
        type=str,
        default=None,
        help=
        'the seeds for dataset. if seeds is not None, ignore seed. ex. 1,2,3,4,5'
    )
    parser.add_argument(
        '--column_multiplier', type=int, default=1, help='column multiplier')
    parser.add_argument(
        '--drop_rate', type=float, default=0.0, help='drop rate.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='learning rate of optimizer')

    parser.add_argument(
        '--normalize_input',
        type=str2bool,
        # default=True,
        default=False,
        help='Whether normalize input layer by our method.')
    parser.add_argument(
        '--normalize_feature',
        type=str2bool,
        # default=True,
        default=False,
        help='Whether normalize each features by min 0, max 1.')

    args = parser.parse_args()

    if args.seeds:
        try:
            seeds = [int(t) for t in args.seeds.split(',')]
        except:
            print('seeds is not valid.')
            assert False

        rmses = []
        for seed in seeds:
            rmse = _train(args, seed=seed)
            rmses.append(rmse)
        print('\n>> average final_test_rmse:', sum(rmses) / len(seeds))

    else:
        rmse = _train(args, seed=args.seed)
        print('\n>> final_test_rmse:', rmse)


if __name__ == '__main__':
    main()
