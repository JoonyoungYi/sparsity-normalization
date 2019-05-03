import re
import random

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston

from .model import init_models
from common.utils import str2bool
from common.utils import set_seed
from common.utils import inject_sparsity
from common.utils import split_sparse_and_dense_group


def _init_x_and_y(args, column_multiplier):
    # https://scikit-learn.org/stable/datasets/index.html
    # https://datascienceschool.net/view-notebook/4b948c4d99c94213bf02bcff2ced88eb/
    # https://datascienceschool.net/view-notebook/577a01e24d4f456bb5060da6e47054e1/
    if args.dataset == 'boston':
        dataset = load_boston()
    elif args.dataset == 'diabetes':
        dataset = load_diabetes()
    elif args.dataset == 'california':
        # tf 16~19에서 돌리는 중.
        dataset = fetch_california_housing()
    else:
        raise NotImplementedError()

    x = dataset.data
    y = dataset.target

    # feature-wise normalization min-max=0-1
    if args.normalize_feature:
        feature_min = np.min(x, axis=0)
        feature_max = np.max(x, axis=0)
        for j in range(x.shape[1]):
            x[:, j] = (x[:, j] - feature_min[j]) / (
                feature_max[j] - feature_min[j])

    # augment features
    x = np.concatenate([x] * column_multiplier, axis=1)

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

    # sparsity injection
    train_x, train_g = inject_sparsity(train_x, args)
    valid_x, valid_g = inject_sparsity(valid_x, args)
    test_x, test_g = inject_sparsity(test_x, args)

    # mean = np.mean(train_y)
    # print('>> Baseline (mean predictor) RMSE:',
    #       np.sqrt(np.mean(np.square(mean - test_y))))
    # train_mean = np.mean(train_y)
    # print('>> Train mean:', train_mean)
    # test_mean = np.mean(test_y)
    # print('>> Test mean:', test_mean)
    # input('')

    return train_x, train_y, valid_x, valid_y, test_x, test_y, test_g


def _train(args, seed, column_multiplier):
    set_seed(seed)
    train_x, train_y, valid_x, valid_y, test_x, test_y, g = _init_x_and_y(
        args, column_multiplier)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models(args, n_feature=train_x.shape[1])
    session.run(tf.global_variables_initializer())

    max_valid_rmse = float('inf')
    max_valid_iter_idx = 0
    final_test_rmse = float('inf')

    max_test_rmse = float('inf')
    max_test_iter_idx = 0

    sparse_test_x, dense_test_x = split_sparse_and_dense_group(test_x, g)

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

        if max_valid_rmse - 1e-5 > valid_rmse:
            max_valid_rmse = valid_rmse
            max_valid_iter_idx = iter_idx
            final_test_rmse = test_rmse

            test_activations = session.run(
                models['activations'],
                feed_dict={
                    models['x']: test_x,
                    models['training']: False,
                })
            test_sparse_activations = session.run(
                models['activations'],
                feed_dict={
                    models['x']: sparse_test_x,
                    models['training']: False,
                })
            test_dense_activations = session.run(
                models['activations'],
                feed_dict={
                    models['x']: dense_test_x,
                    models['training']: False,
                })

        if max_test_rmse - 1e-5 > test_rmse:
            max_test_rmse = test_rmse
            max_test_iter_idx = iter_idx

        if iter_idx > max_valid_iter_idx + args.epoch_early_stop:
            break

    session.close()
    return final_test_rmse, test_activations[-1], test_sparse_activations[
        -1], test_dense_activations[-1]


def main():
    import argparse

    parser = argparse.ArgumentParser(description='uci-dataset-regression')
    parser.add_argument(
        '--dataset',
        type=str,
        default='boston',
        # default='diabetes',
        # default='california',
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
        '--incremental_column_multipliers',
        type=str2bool,
        default=False,
        help='the incremental column_multipliers.')
    parser.add_argument(
        '--inject_sparsity',
        type=str2bool,
        default=True,
        help='Whether inject sparsity.')
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
        default=True,
        # default=False,
        help='Whether normalize each features by min 0, max 1.')

    args = parser.parse_args()

    assert args.inject_sparsity

    if args.incremental_column_multipliers:
        column_multipliers = list(range(1, 10)) + list(
            range(10, 100, 10)) + list(range(100, 1000, 100))
    else:
        column_multipliers = [args.column_multiplier]

    for column_multiplier in column_multipliers:
        if args.seeds:
            try:
                seeds = [int(t) for t in args.seeds.split(',')]
            except:
                print('seeds is not valid.')
                assert False

            rmses, acts, sparse_acts, dense_acts = [], [], [], []
            for seed in seeds:
                rmse, act, sparse_act, dense_act = _train(
                    args, seed=seed, column_multiplier=column_multiplier)

                rmses.append(rmse)
                acts.append(act)
                dense_acts.append(dense_act)
                sparse_acts.append(sparse_act)

            rmse = sum(rmses) / len(seeds)
            sparse_act = sum(sparse_acts) / len(seeds)
            act = sum(acts) / len(seeds)
            dense_act = sum(dense_acts) / len(seeds)

            print('\n>> average final_test_rmse:', rmse)
            f = open('results/uci_reg/{}-{}.txt'.format(
                args.dataset, args.normalize_input), 'a')
            results = [
                column_multiplier,
                rmse,
                sparse_act,
                act,
                dense_act,
            ]
            line = '\t'.join([str(r) for r in results]) + '\n'
            f.write(line)
            f.close()

        else:
            rmse = _train(
                args, seed=args.seed, column_multiplier=args.column_multiplier)
            print('\n>> final_test_rmse:', rmse)


if __name__ == '__main__':
    main()
