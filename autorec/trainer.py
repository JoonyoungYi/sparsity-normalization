import time
import random
import math
from datetime import datetime

import numpy as np
import tensorflow as tf

from movielens.dataset import Dataset
from .configs import *
from .model import init_models


def _init_mask(matrix):
    mask = np.zeros(matrix.shape)
    mask[matrix > 0.1] = 1
    return mask


def _validate(
        session,
        models,
        train_x,
        valid_x,
        valid_y,
        test_x,
        test_y, ):
    train_rmse = session.run(
        models['test_rmse'],
        feed_dict={
            models['train_x']: train_x,
            models['test_y']: train_x,
        }, )

    valid_rmse = session.run(
        models['test_rmse'],
        feed_dict={
            models['train_x']: valid_x,
            models['test_y']: valid_y,
        }, )

    test_rmse = session.run(
        (models['test_rmse']),
        feed_dict={
            models['train_x']: test_x,
            models['test_y']: test_y,
        }, )
    return train_rmse, valid_rmse, test_rmse


def _train(args):
    dataset = Dataset(
        Dataset.KIND_MOVIELENS_100k,
        Dataset.SPLIT_DEFAULT_10_PERCENT,
        seed=args.data_seed)

    n_user, n_item = dataset.get_n_user_and_n_item()

    train_x = dataset.train_matrix
    valid_x = dataset.train_matrix
    valid_y = dataset.valid_matrix
    test_x = dataset.train_matrix + dataset.valid_matrix
    test_y = dataset.test_matrix

    n_col = n_item

    data_for_validation = {
        'train_x': train_x,
        'valid_x': valid_x,
        'valid_y': valid_y,
        'test_x': test_x,
        'test_y': test_y,
    }

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models(n_col, args)
    session.run(tf.global_variables_initializer())

    min_valid_rmse = float('inf')
    min_valid_rmse_iter_idx = 0
    min_test_rmse = float('inf')
    min_test_rmse_iter_idx = 0
    final_test_rmse = float('inf')
    final_train_rmse = float('inf')

    for iter_idx in range(1, args.max_iter):
        session.run(
            (models['train_op']),
            feed_dict={
                models['train_x']: train_x,
            }, )

        if (iter_idx) % args.iter_validation == 0:
            train_rmse, valid_rmse, test_rmse = _validate(
                session, models, **data_for_validation)

            if min_valid_rmse > valid_rmse:
                min_valid_rmse = valid_rmse
                final_test_rmse = test_rmse
                final_train_rmse = train_rmse
                min_valid_rmse_iter_idx = iter_idx

            if min_test_rmse > test_rmse:
                min_test_rmse = test_rmse
                min_test_rmse_iter_idx = iter_idx

            msg = '\n[EPOCH %6d]\n' % (iter_idx)
            msg += '  - train rmse: {:12.8f}\n'.format(train_rmse)
            msg += '  - valid rmse: {:12.8f}    (min: {:12.8f}, iter: {:d}, final: {:12.8f})\n'.format(
                valid_rmse, min_valid_rmse, min_valid_rmse_iter_idx,
                final_test_rmse)
            msg += '  - test  rmse: {:12.8f}    (min: {:12.8f}, iter: {:d})\n'.format(
                test_rmse, min_test_rmse, min_test_rmse_iter_idx)
            print(msg)

            if (iter_idx > min_valid_rmse_iter_idx + args.iter_early_stop) or (
                    min_valid_rmse < 0.95 and
                    min_valid_rmse + 0.04 < valid_rmse):
                break

    session.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='AutoRec')
    parser.add_argument(
        '--iter_validation',
        type=int,
        default=100,
        help='Iteration unit for validation')
    parser.add_argument(
        '--max_iter', type=int, default=100000, help='Max Iteration')
    parser.add_argument(
        '--n_hidden_unit',
        type=int,
        default=500,
        help='The number of hidden unit')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='learning rate for optimizer.')
    parser.add_argument(
        '--lambda_wd', type=float, default=1, help='lambda for weight decay.')
    parser.add_argument(
        '--iter_early_stop',
        type=int,
        default=1000,
        help='the number of iteration for early stop.')
    parser.add_argument(
        '--data_seed', type=int, default=1, help='the seed for dataset')

    args = parser.parse_args()
    _train(args)


if __name__ == '__main__':
    main()
