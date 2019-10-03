import time
import random
import math
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf

from common.datasets.movielens import Dataset
from .models import init_models
from .batch import Batch
from common.utils import str2bool
from common.utils import set_seed
from common.utils import append_text_results


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

    for epoch_idx in range(1, args.max_epoch + 1):

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

            if train_rmse < 0.1:
                break

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


def main():
    import argparse

    parser = argparse.ArgumentParser(description='AlternatingAutoencoders')
    parser.add_argument(
        '--epoch_validation',
        type=int,
        default=1,
        help='epoch unit for validation')
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=100000,
        # default=1,
        help='Max epoch')
    parser.add_argument(
        '--normalize_input',
        type=str2bool,
        default=True,
        # default=False,
        help='Whether normalize input.')
    parser.add_argument(
        '--clipping',
        type=str2bool,
        # default=True,
        default=False,
        help='')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
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
        '--lambda_w',
        type=float,
        default=600.0,
        help='lambda for weight decay.')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=[
            Dataset.KIND_MOVIELENS_100k,
            Dataset.KIND_MOVIELENS_1M,
            Dataset.KIND_MOVIELENS_10M,
        ],
        default=Dataset.KIND_MOVIELENS_100k,
        # default=Dataset.KIND_MOVIELENS_1M,
        # default=Dataset.KIND_MOVIELENS_10M,
        help='dataset')
    parser.add_argument(
        '--n_batch',
        type=int,
        default=512,
        # default=1000,
        help='0 for full batch. integer for batch number.')

    args = parser.parse_args()
    args.use_user_vector = False
    args.K_denominator = 500

    if args.n_batch == 0:
        args.n_batch = None
    elif args.n_batch < 0:
        print('args.n_batch should not be negative.')
        assert False

    if args.dataset_seed is None:
        args.dataset_seed = args.seed

    _train(args)


if __name__ == '__main__':
    main()
