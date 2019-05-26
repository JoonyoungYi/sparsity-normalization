import re
import random

import tensorflow as tf
import numpy as np

from .model import init_models
from .dataset import Dataset
from common.utils import str2bool, integer_list
from common.utils import set_seed
from common.utils import save_results_to_file


def _train(args):
    # print(args)
    # input('')

    set_seed(args.seed)
    dataset = Dataset(args)
    train_x = dataset.train_x
    train_y = dataset.train_y
    valid_x = dataset.valid_x
    valid_y = dataset.valid_y
    test_x = dataset.test_x
    test_y = dataset.test_y
    n_class = dataset.n_class
    sparse_test_x = dataset.sparse_test_x
    dense_test_x = dataset.dense_test_x

    # print(train_x.shape)
    # print(valid_x.shape)
    # print(test_x.shape)
    # print(test_x)
    # input('')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models(args, n_feature=train_x.shape[1], n_class=n_class)
    session.run(tf.global_variables_initializer())

    max_valid_error = float('inf')
    max_valid_iter_idx = 0
    final_test_error = float('inf')

    max_test_error = float('inf')
    max_test_iter_idx = 0

    for iter_idx in range(args.max_epoch):
        _, train_loss, train_error = session.run(
            [models['train_op'], models['loss'], models['error']],
            feed_dict={
                models['x']: train_x,
                models['y']: train_y,
                models['training']: True,
            })

        valid_error = session.run(
            models['error'],
            feed_dict={
                models['x']: valid_x,
                models['y']: valid_y,
                models['training']: False,
            })
        test_error = session.run(
            models['error'],
            feed_dict={
                models['x']: test_x,
                models['y']: test_y,
                models['training']: False,
            })

        print('')
        print('%6d' % iter_idx, '%.2f' % train_loss, '%.4f' % train_error,
              '%.4f' % valid_error, '%.4f' % test_error)
        print('%6d' % max_valid_iter_idx, '%.4f' % max_valid_error,
              '%.4f' % final_test_error)

        if max_valid_error - 1e-5 > valid_error:
            max_valid_error = valid_error
            max_valid_iter_idx = iter_idx
            final_test_error = test_error

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

        # input('')

        if max_test_error - 1e-5 > test_error:
            max_test_error = test_error
            max_test_iter_idx = iter_idx

        if iter_idx > max_valid_iter_idx + args.epoch_early_stop:
            break

    session.close()

    # final_test_error, test_activations[-1], test_sparse_activations[
    #     -1], test_dense_activations[-1]

    results = []
    results.append(args.kind)
    results.append(args.seed)
    results.append(args.n_replication)
    results.append(final_test_error)
    save_results_to_file('uci', args.dataset, results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='uci-dataset')
    parser.add_argument(
        '--dataset',
        type=str,
        default='boston',
        # default='diabetes',
        # default='california',
        # default='iris',
        # default='wine',
        # default='cancer',
        # default='congress',
        choices=[
            'boston',
            'diabetes',
            'california',
            'iris',
            'wine',
            'cancer',
            'congress',
        ],
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
        '--inject_sparsity',
        type=str2bool,
        default=True,
        # default=False,
        help='Whether inject sparsity.')
    parser.add_argument(
        '--n_replication', type=int, default=1, help='column multiplier')
    parser.add_argument(
        '--drop_rate', type=float, default=0.0, help='drop rate.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='learning rate of optimizer')

    parser.add_argument(
        '--kind',
        type=str,
        default='default',
        choices=[
            'default', 'sparsity-normal', 'impute-mean', 'concat-pos',
            'concat-neg'
        ],
        help='Whether normalize input layer by our method.')
    parser.add_argument(
        '--normalize_feature',
        type=str2bool,
        default=True,
        # default=False,
        help='Whether normalize each features by min 0, max 1.')

    args = parser.parse_args()
    _train(args)


if __name__ == '__main__':
    main()
