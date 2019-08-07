import re
import random

import tensorflow as tf
import numpy as np

from .model import init_models
from common.datasets.uci import Dataset
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
    sparse_test_y = dataset.sparse_test_y
    dense_test_x = dataset.dense_test_x
    dense_test_y = dataset.dense_test_y

    print(train_x.shape)
    print(valid_x.shape)
    print(test_x.shape)
    # print(test_x)
    input('!!!')

    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
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

        if max_valid_error - 1e-5 > valid_error:
            max_valid_error = valid_error
            max_valid_iter_idx = iter_idx
            final_test_error = test_error

            test_y_hat = session.run(
                models['y_hat'],
                feed_dict={
                    models['x']: test_x,
                    models['training']: False,
                })
            test_sparse_y_hat, test_sparse_error = session.run(
                [models['y_hat'], models['error']],
                feed_dict={
                    models['x']: sparse_test_x,
                    models['y']: sparse_test_y,
                    models['training']: False,
                })
            test_dense_y_hat, test_dense_error = session.run(
                [models['y_hat'], models['error']],
                feed_dict={
                    models['x']: dense_test_x,
                    models['y']: dense_test_y,
                    models['training']: False,
                })

            if n_class is None:
                test_activation = np.mean(test_y_hat)
                test_sparse_activation = np.mean(test_sparse_y_hat)
                test_dense_activation = np.mean(test_dense_y_hat)
            else:
                test_activation = np.mean(test_y_hat[:, test_y])
                test_sparse_activation = np.mean(
                    test_sparse_y_hat[:, sparse_test_y])
                test_dense_activation = np.mean(
                    test_dense_y_hat[:, dense_test_y])

                test_sparse_activation /= test_activation
                test_dense_activation /= test_activation
                test_activation = 1.0

        # input('')

        if max_test_error - 1e-5 > test_error:
            max_test_error = test_error
            max_test_iter_idx = iter_idx

        print('')
        print('%6d' % iter_idx, '%.2f' % train_loss, '%.8f' % train_error,
              '%.8f' % valid_error, '%.8f' % test_error)
        print('%6d' % max_valid_iter_idx, '%.8f' % max_valid_error,
              '%.8f' % final_test_error)
        print('%.8f' % test_sparse_activation, '%.8f' % test_activation,
              '%.8f' % test_dense_activation)
        print('%.8f' % test_sparse_error, '%.8f' % test_dense_error)

        if iter_idx > max_valid_iter_idx + args.epoch_early_stop:
            break

    session.close()

    # final_test_error, test_activations[-1], test_sparse_activations[
    #     -1], test_dense_activations[-1]

    results = []
<<<<<<< HEAD
    results.append(args.model)
=======
    results.append(args.kind)
>>>>>>> 8b11627036ac569e926c43e9ae35604f2a49f415
    results.append(args.seed)
    results.append(args.n_replication)
    results.append(final_test_error)
    save_results_to_file('uci', args.dataset, results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='uci')
    parser.add_argument(
        '--dataset',
        type=str,
        default='boston',
        # default='diabetes',
        # default='california',
        # default='iris',
        # default='cancer',
<<<<<<< HEAD
        # default='wine',
=======
>>>>>>> 8b11627036ac569e926c43e9ae35604f2a49f415
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
        default=1000,
        help='the number of epochation for early stop.')
    parser.add_argument(
        '--n_hidden_layer',
        type=int,
        default=3,
        help='the number of hidden layers.')
    parser.add_argument(
        '--n_hidden_unit',
        type=int,
        default=32,
        help='the number of hidden unit of each layer.')
    parser.add_argument(
        '--activation_fn',
        type=str,
        default='relu',
        # default='tanh',
        # default='sigmoid',
        choices=['relu', 'tanh', 'sigmoid'],
        help='the activation function')
    parser.add_argument('--seed', type=int, default=1, help='the seed')
    parser.add_argument(
<<<<<<< HEAD
        '--dataset_seed',
        type=int,
        default=1,
        help='the seed for dataset. None 일 때는 seed 값을 따라갑니다.')
=======
        '--seed', type=int, default=1, help='the seed for dataset')
>>>>>>> 8b11627036ac569e926c43e9ae35604f2a49f415
    parser.add_argument(
        '--inject_sparsity',
        type=str2bool,
        default=True,
        # default=False,
        help=
        'Whether inject sparsity. 반절에는 20%의 sparsity를 주입(dense group), 나머지 반절은 80%의 sparsity를 주입(sparse group)합니다.'
    )
    parser.add_argument(
<<<<<<< HEAD
        '--n_replication',
        type=int,
        default=100,
        help='column multiplier. 논문에서는 r로 사용하고 있습니다.')
=======
        '--n_replication', type=int, default=1, help='column multiplier')
>>>>>>> 8b11627036ac569e926c43e9ae35604f2a49f415
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.0,
        help='drop rate for dropout.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='learning rate of optimizer')
    parser.add_argument(
        '--K_prime',
        type=float,
        # default=None,
        default=32,
        help='K_prime이 None이면 자동으로 정합니다.')

    parser.add_argument(
<<<<<<< HEAD
        '--model',
        type=str,
        # default='default',
        default='sparsity-normalization',
        choices=[
            'default',
            'sparsity-normalization',  # sparsity normalization을 적용합니다.
            'mean-imputation',  # mean imputation을 수행합니다.
            'positive-mask-concatenation',  # 있으면 1 없으면 0인 mask vector를 concatenation 합니다.
            'negative-mask-concatenation',  # 있으면 0 없으면 1인 mask vector를 concatenation 합니다.
=======
        '--kind',
        type=str,
        default='default',
        choices=[
            'default', 'sparsity-normal', 'impute-mean', 'concat-pos',
            'concat-neg'
>>>>>>> 8b11627036ac569e926c43e9ae35604f2a49f415
        ],
        help='Whether normalize input layer by our method.')
    parser.add_argument(
        '--input_normalization',
        type=str,
        # default='min-0-max-1',
        # default='min--1-max-1',
        default='standardization',
        # default='none',
        choices=[
            'min-0-max-1',  # min을 0으로 만들고, max를 1로 만듭니다.
            'min--1-max-1',  # min을 -1로 만들고, max를 1로 만듭니다.
            'standardization',  # 표준화
            'none',  # 아무것도 하지 않습니다.
        ],
        help='어떻게 input을 normalization할지 정합니다. ')

    args = parser.parse_args()
<<<<<<< HEAD

    if args.dataset_seed is None:
        args.dataset_seed = args.seed

=======
>>>>>>> 8b11627036ac569e926c43e9ae35604f2a49f415
    _train(args)


if __name__ == '__main__':
    main()
