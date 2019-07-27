import time
import random
import math
from datetime import datetime

import numpy as np
import tensorflow as tf

from common.datasets.movielens import Dataset
from .configs import *
from .model import init_models
from .batch import Batch
from common.utils import str2bool
from common.utils import set_seed
from common.utils import save_results_to_file


def __get_rmse(session, models, batch, key, args):
    func = getattr(batch, 'batch_' + key)

    if args.n_batch is None:
        x, y = func()
        rmse = session.run(
            models['rmse'],
            feed_dict={
                models['x']: x,
                models['y']: y,
                models['training']: False,
            }, )
        return rmse
    else:
        sse, n = 0.0, 0.0
        for iter_idx in range(batch.n_row // args.n_batch):
            x, y = func(iter_idx)
            _rmse = session.run(
                (models['rmse']),
                feed_dict={
                    models['x']: x,
                    models['y']: y,
                    models['training']: False,
                }, )
            _n = x[2][0]
            _sse = math.pow(_rmse, 2) * _n
            n += _n
            sse += _sse
        return math.sqrt(sse / n)


def _validate(session, models, batch, args):
    train_rmse = __get_rmse(session, models, batch, 'train', args)
    valid_rmse = __get_rmse(session, models, batch, 'valid', args)
    test_rmse = __get_rmse(session, models, batch, 'test', args)
    return train_rmse, valid_rmse, test_rmse


def _train(args):
    dataset = Dataset(
        args.dataset, Dataset.SPLIT_DEFAULT_10_PERCENT, seed=args.dataset_seed)
    set_seed(args.seed)

    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    batch = Batch(dataset, args)

    models = init_models(batch.n_col, args)
    session.run(tf.global_variables_initializer())

    min_valid_rmse = float('inf')
    min_valid_rmse_epoch_idx = 0
    min_test_rmse = float('inf')
    min_test_rmse_epoch_idx = 0
    final_test_rmse = float('inf')
    final_train_rmse = float('inf')

    for epoch_idx in range(1, args.max_epoch):

        if args.n_batch is None:
            batch_x, batch_y = batch.batch_train()
            session.run(
                (models['train_op']),
                feed_dict={
                    models['x']: batch_x,
                    models['y']: batch_y,
                    models['training']: True,
                }, )
        else:
            for iter_idx in range(batch.n_row // args.n_batch):
                batch_x, batch_y = batch.batch_train(iter_idx)
                if epoch_idx == 1:
                    # print(list(batch_x[1]))
                    assert batch_x[1][batch_x[1] < 1e-8].shape[0] == 0

                session.run(
                    (models['train_op']),
                    feed_dict={
                        models['x']: batch_x,
                        models['y']: batch_y,
                        models['training']: True,
                    }, )

        if (epoch_idx) % args.epoch_validation == 0:
            train_rmse, valid_rmse, test_rmse = _validate(
                session, models, batch, args)

            if min_valid_rmse - 1e-5 > valid_rmse:
                min_valid_rmse = valid_rmse
                final_test_rmse = test_rmse
                final_train_rmse = train_rmse
                min_valid_rmse_epoch_idx = epoch_idx

            if min_test_rmse - 1e-5 > test_rmse:
                min_test_rmse = test_rmse
                min_test_rmse_epoch_idx = epoch_idx

            msg = '\n[EPOCH %6d]\n' % (epoch_idx)
            msg += '  - train rmse: {:12.8f}\n'.format(train_rmse)
            msg += '  - valid rmse: {:12.8f}    (min: {:12.8f}, epoch: {:d}, final: {:12.8f})\n'.format(
                valid_rmse, min_valid_rmse, min_valid_rmse_epoch_idx,
                final_test_rmse)
            msg += '  - test  rmse: {:12.8f}    (min: {:12.8f}, epoch: {:d})\n'.format(
                test_rmse, min_test_rmse, min_test_rmse_epoch_idx)
            print(msg)

            if (epoch_idx > min_valid_rmse_epoch_idx + args.epoch_early_stop):
                break

    session.close()

    results = []
    results.append(args.use_user_vector)
    results.append(args.normalize_input)
    results.append(args.sparse_tensor)
    results.append(args.n_batch)
    results.append(args.optimizer)
    results.append(args.lambda_w)
    results.append(args.seed)
    results.append(args.n_hidden_unit)
    results.append(args.dataset_seed)
    results.append(final_test_rmse)
    results.append(min_test_rmse)

    filename = args.dataset
    if args.filename:
        filename += '-' + args.filename
    save_results_to_file('autorec', filename, results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='AutoRec')
    parser.add_argument(
        '--epoch_validation',
        type=int,
        default=1,
        help='epoch unit for validation')
    parser.add_argument(
        '--max_epoch', type=int, default=100000, help='Max epoch')
    parser.add_argument(
        '--n_hidden_unit',
        type=int,
        default=500,
        help='The number of hidden unit')
    parser.add_argument(
        '--n_hidden_layer',
        type=int,
        default=1,
        help='The number of hidden unit')
    parser.add_argument(
        '--normalize_input',
        type=str2bool,
        default=True,
        # default=False,
        help='Whether normalize input.')
    parser.add_argument(
        '--use_user_vector',
        type=str2bool,
        # default=True,
        default=False,
        help='Whether use user vector.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='learning rate for optimizer.')
    parser.add_argument('--seed', type=int, default=1, help='the seed')
    parser.add_argument(
        '--dataset_seed',
        type=int,
        default=None,
        help='the seed for dataset. None 일 때는 seed 값을 따라갑니다.')
    parser.add_argument(
        '--epoch_early_stop',
        type=int,
        default=1000,
        help='the number of epochation for early stop.')

    parser.add_argument(
        '--lambda_w', type=float, default=0.0, help='lambda for weight decay.')
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['rprop', 'adam'],
        default='adam',
        help='Optimizer')
    parser.add_argument(
        '--filename', type=str, default='None', help='log filename')
    parser.add_argument(
        '--activation_fn',
        type=str,
        # default='relu',
        default='sigmoid',
        choices=['relu', 'sigmoid'],
        help='')
    parser.add_argument(
        '--K_denominator',
        type=float,
        default=500,
        # default=None,
        help='K')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=[
            Dataset.KIND_MOVIELENS_100k,
            Dataset.KIND_MOVIELENS_1M,
            Dataset.KIND_MOVIELENS_10M,
        ],
        # default=Dataset.KIND_MOVIELENS_100k,
        # default=Dataset.KIND_MOVIELENS_1M,
        default=Dataset.KIND_MOVIELENS_10M,
        help='dataset')
    parser.add_argument(
        '--sparse_tensor',
        type=str2bool,
        default=True,
        help='whether use sparse tensor.')
    parser.add_argument(
        '--n_batch',
        type=int,
        # default=0,
        default=1000,
        help='0 for full batch. integer for batch number.')

    args = parser.parse_args()

    if args.n_batch == 0:
        args.n_batch = None
    elif args.n_batch < 0:
        print('args.n_batch should not be negative.')
        assert False

    if args.dataset_seed is None:
        args.dataset_seed = args.seed

    if args.sparse_tensor:
        # RProp optimizer can be used only for dense tensor.
        assert args.optimizer != 'rprop'

    _train(args)


if __name__ == '__main__':
    main()

# python3 -m autorec.trainer --epoch_early_stop=1000 --use_user_vector=False --learning_rate=1e-3 --optimizer=adam --dataset=movielens-100k --sparse_tensor=True --seed=1 --filename=user-normal --normalize_input=True --n_hidden_layer=2 --lambda_w=500

# python3 -m autorec.trainer --epoch_early_stop=1000 --use_user_vector=False --learning_rate=1e-3 --optimizer=adam --dataset=movielens-1m --sparse_tensor=True --seed=1 --filename=user-normal --normalize_input=True --n_hidden_layer=2 --lambda_w=500

# python3 -m autorec.trainer --use_user_vector=False --learning_rate=1e-4 --optimizer=adam --dataset=movielens-10m --sparse_tensor=True --n_batch=1000 --filename=item-normal --normalize_input=True --epoch_early_stop=1000 --seed=1 --n_hidden_layer=1 --n_hidden_unit=2000 --K_denominator=2000 --lambda_w=100
