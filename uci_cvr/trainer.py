import random

import tensorflow as tf
import numpy as np

from .model import init_models
from .configs import *
from common.utils import set_seed


def __get_x_from_t(t):
    if t == 'y':
        return [1, 0]
    elif t == '?':
        return [0, 0]
    else:
        return [0, 1]


def __get_m_from_t(t):
    if t == 'y':
        return 1
    elif t == '?':
        return 0
    else:
        return 1


def _prepare_x_y_m():
    f = open('uci_cvr/data.csv', 'r')
    x = []
    m = []
    y = []
    for line in f:
        tokens = line.strip().split(',')
        assert len(tokens) == 17
        y.append([1, 0] if tokens[0] == 'republican' else [0, 1])

        tokens = tokens[1:]
        prob = 0.2 if random.random() < 0.5 else 0.8
        _x, _m = [], []
        for t in tokens:
            if random.random() < prob:
                _x.extend(__get_x_from_t(t))
                _m.append(__get_m_from_t(t))
            else:
                _x.extend([0, 0])
                _m.append(0)

        if sum(_m) == 0:
            # print(tokens)
            # input('')
            continue

        m.append(_m)
        x.append(_x)
    f.close()
    return x, y, m


def _split_train_test(xs, ys, ms):
    random.seed(1)
    train_x, test_x = [], []
    train_y, test_y = [], []
    train_m, test_m = [], []
    for _x, _y, _m in zip(xs, ys, ms):
        if random.random() < 0.2:
            train_x.append(_x)
            train_y.append(_y)
            train_m.append(_m)
        else:
            test_x.append(_x)
            test_y.append(_y)
            test_m.append(_m)

    return train_x, train_y, train_m, test_x, test_y, test_m


def _get_handled_x(x, m, args):
    if args.kind == KIND_IMPUTE_NORMAL:
        for i, _m in enumerate(m):
            for j, __m in enumerate(_m):
                if __m == 0:
                    x[i][j * 2] = 0.5
                    x[i][j * 2 + 1] = 0.5
        return x
    elif args.kind == KIND_SPARSITY_NORMAL:
        for i, _x in enumerate(x):
            d = sum(_x)
            _x = [__x * args.K_prime / d for __x in _x]
            x[i] = _x
        return x
    else:
        return x


def _train(args):
    set_seed(args.seed)
    x, y, m = _prepare_x_y_m()
    x = _get_handled_x(x, m, args)
    train_x, train_y, train_m, test_x, test_y, test_m = _split_train_test(
        x, y, m)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models(args)
    session.run(tf.global_variables_initializer())

    max_test_acc = 0.0
    max_test_iter_idx = 0
    for iter_idx in range(1000):
        _, train_loss, train_acc = session.run(
            [models['train_op'], models['loss'], models['accuracy']],
            feed_dict={
                models['x']: train_x,
                models['y']: train_y,
                models['m']: train_m,
            })

        test_loss, test_acc = session.run(
            [models['loss'], models['accuracy']],
            feed_dict={
                models['x']: test_x,
                models['y']: test_y,
                models['m']: test_m,
            })
        print('%4d' % iter_idx, '%5.2f' % train_loss, '%.6f' % train_acc,
              '%.2f' % test_loss, '%.6f' % test_acc, '%.6f' % max_test_acc)

        if max_test_acc < test_acc:
            max_test_acc = test_acc
            max_test_iter_idx = iter_idx

        if iter_idx > max_test_iter_idx + 100:
            break


def main():
    import argparse

    parser = argparse.ArgumentParser(description='uci-cvr')
    parser.add_argument(
        '--seed', type=int, default=1, help='the seed for dataset')
    parser.add_argument(
        '--kind',
        type=str,
        default=KIND_DEFAULT,
        choices=[
            KIND_DEFAULT,
            KIND_SPARSITY_NORMAL,
            KIND_CONCAT_POS,
            KIND_CONCAT_NEG,
            KIND_IMPUTE_NORMAL,
        ],
        help='dataset.')
    parser.add_argument(
        '--n_hidden_layer',
        type=int,
        default=2,
        help='the number of hidden layers.')
    parser.add_argument('--K_prime', type=int, default=1, help='K_prime')
    parser.add_argument(
        '--n_hidden_unit',
        type=int,
        default=16,
        help='the number of hidden unit of each layer.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='learning rate of optimizer')

    # parser.add_argument(
    #     '--max_epoch', type=int, default=1000000, help='Max epoch')
    # parser.add_argument(
    #     '--epoch_early_stop',
    #     type=int,
    #     default=100,
    #     help='the number of epochation for early stop.')
    # parser.add_argument(
    #     '--activation_fn',
    #     type=str,
    #     default='relu',
    #     # default='tanh',
    #     # default='sigmoid',
    #     choices=['relu', 'tanh', 'sigmoid'],
    #     help='the activation function')
    # parser.add_argument(
    #     '--seeds',
    #     type=integer_list,
    #     default=[],
    #     help=
    #     'the seeds for dataset. if seeds is not None, ignore seed. ex. 1,2,3,4,5'
    # )
    # parser.add_argument(
    #     '--incremental_column_multipliers',
    #     type=str2bool,
    #     default=False,
    #     help='the incremental column_multipliers.')
    # parser.add_argument(
    #     '--inject_sparsity',
    #     type=str2bool,
    #     default=True,
    #     # default=False,
    #     help='Whether inject sparsity.')
    # parser.add_argument(
    #     '--column_multiplier', type=int, default=1, help='column multiplier')
    # parser.add_argument(
    #     '--drop_rate', type=float, default=0.0, help='drop rate.')
    # parser.add_argument(
    #     '--normalize_input',
    #     type=str2bool,
    #     default=True,
    #     # default=False,
    #     help='Whether normalize input layer by our method.')
    # parser.add_argument(
    #     '--normalize_feature',
    #     type=str2bool,
    #     default=True,
    #     # default=False,
    #     help='Whether normalize each features by min 0, max 1.')

    args = parser.parse_args()
    _train(args)


if __name__ == '__main__':
    main()
