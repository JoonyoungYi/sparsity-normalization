import time
import random
import math
from datetime import datetime

import numpy as np
import tensorflow as tf

from movielens.dataset import Dataset
from .batch import BatchManager
from .configs import *
from .model import init_models
from common.utils import str2bool
from common.utils import set_seed


def _validate(
        session,
        models,
        train_x,
        valid_x,
        valid_y,
        test_x,
        test_y, ):

    train_rmse = session.run(
        models['rmse'],
        feed_dict={
            models['x']: train_x,
            models['y']: train_x,
            models['training']: False,
        }, )

    valid_rmse = session.run(
        models['rmse'],
        feed_dict={
            models['x']: valid_x,
            models['y']: valid_y,
            models['training']: False,
        }, )

    test_rmse = session.run(
        (models['rmse']),
        feed_dict={
            models['x']: test_x,
            models['y']: test_y,
            models['training']: False,
        }, )
    return train_rmse, valid_rmse, test_rmse


def _train(args):
    set_seed(args.seed)

    dataset = Dataset(
        Dataset.KIND_MOVIELENS_1M,
        # Dataset.KIND_MOVIELENS_100k,
        Dataset.SPLIT_DEFAULT_10_PERCENT,
        seed=args.seed)
    dataset.init_full_matrices()

    n_user, n_item = dataset.get_n_user_and_n_item()
    n_class = dataset.n_class

    train_x = dataset.train_matrix
    valid_x = dataset.train_matrix
    valid_y = dataset.valid_matrix
    test_x = dataset.train_matrix + dataset.valid_matrix
    test_y = dataset.test_matrix

    if args.use_user_vector:
        n_col = n_item
    else:
        train_x = train_x.T
        valid_x = valid_x.T
        valid_y = valid_y.T
        test_x = test_x.T
        test_y = test_y.T
        n_col = n_user

    data_for_validation = {
        'train_x': train_x,
        'valid_x': valid_x,
        'valid_y': valid_y,
        'test_x': test_x,
        'test_y': test_y,
    }

    batch_manager = BatchManager(train_x, args)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models(n_col, dataset.n_class, args)
    session.run(tf.global_variables_initializer())

    min_valid_rmse = float('inf')
    min_valid_rmse_iter_idx = 0
    min_test_rmse = float('inf')
    min_test_rmse_iter_idx = 0
    final_test_rmse = float('inf')
    final_train_rmse = float('inf')

    train_losses = []
    for iter_idx in range(1, args.max_iter):
        batch_x, batch_y = batch_manager.get_batch()

        # _x_hot = session.run(
        #     (models['_x_hot']),
        #     feed_dict={
        #         models['x']: batch_x,
        #         models['y']: batch_x,
        #         models['training']: True,
        #     }, )
        # print(_x_hot)
        # for a in list(_x_hot[0]):
        #     if sum(a) > 0:
        #         print('!', a, sum(a))
        # input('>>')

        _, train_loss = session.run(
            (models['train_op'], models['loss']),
            feed_dict={
                models['x']: batch_x,
                models['y']: batch_y,
                models['training']: True,
            }, )
        train_losses.append(train_loss)

        # _, a_i, a_o = session.run(
        #     (models['a_train_op'], models['a_i'], models['a_o']),
        #     feed_dict={
        #         models['x']: batch_x,
        #         models['y']: batch_x,
        #         models['training']: True,
        #     }, )

        if (iter_idx) % args.iter_validation == 0:
            train_rmse, valid_rmse, test_rmse = _validate(
                session, models, **data_for_validation)
            if math.isnan(train_rmse):
                print('Train RMSE is NaN.')
                assert False

            if min_valid_rmse > valid_rmse:
                min_valid_rmse = valid_rmse
                final_test_rmse = test_rmse
                final_train_rmse = train_rmse
                min_valid_rmse_iter_idx = iter_idx

            if min_test_rmse > test_rmse:
                min_test_rmse = test_rmse
                min_test_rmse_iter_idx = iter_idx

            msg = '\n[ITER %6d]\n' % (iter_idx)
            msg += '  - train loss: {:12.1f}\n'.format(
                sum(train_losses) / len(train_losses))
            msg += '  - train rmse: {:12.8f}\n'.format(train_rmse)
            msg += '  - valid rmse: {:12.8f}    (min: {:12.8f}, iter: {:d}, final: {:12.8f})\n'.format(
                valid_rmse, min_valid_rmse, min_valid_rmse_iter_idx,
                final_test_rmse)
            msg += '  - test  rmse: {:12.8f}    (min: {:12.8f}, iter: {:d})\n'.format(
                test_rmse, min_test_rmse, min_test_rmse_iter_idx)
            print(msg)
            # for i in range(5):
            #     print(["{:8.4f}".format(j) for j in a_i[i]])
            # print('')
            # for i in range(5):
            #     print(["{:8.4f}".format(j) for j in a_o[i]])

            if (iter_idx > min_valid_rmse_iter_idx + args.iter_early_stop) or (
                    min_valid_rmse < 0.95 and
                    min_valid_rmse + 0.04 < valid_rmse):
                break

    session.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CFNADE-tensorflow')
    parser.add_argument(
        '--dataset',
        default=Dataset.KIND_MOVIELENS_1M,
        choices=[
            Dataset.KIND_MOVIELENS_100k, Dataset.KIND_MOVIELENS_1M,
            Dataset.KIND_MOVIELENS_10M
        ],
        help='Specify dataset.')
    parser.add_argument(
        '--iter_validation',
        type=int,
        default=1,
        help='Iteration unit for validation')
    parser.add_argument(
        '--max_iter', type=int, default=10000000, help='Max Iteration')
    parser.add_argument(
        '--n_hidden_unit',
        type=int,
        default=500,
        help='The number of hidden unit')
    parser.add_argument(
        '--parameter_sharing',
        type=str2bool,
        default=True,
        help='parameter sharing')
    parser.add_argument(
        '--n_batch', type=int, default=512, help='the batch size')
    parser.add_argument(
        '--normalize_input',
        type=str2bool,
        default=False,
        help='whether normalize input')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0005,
        help='learning rate for optimizer.')
    parser.add_argument(
        '--use_ordinal_cost',
        type=str2bool,
        default=True,
        help='Whether use ordinal cost')
    parser.add_argument(
        '--use_user_vector',
        type=str2bool,
        default=False,
        help='Whether use user vector')
    parser.add_argument(
        '--gpu_memory_fraction',
        type=float,
        default=0.25,
        help='per_process_gpu_memory_fraction')

    parser.add_argument(
        '--lambda_1',
        type=float,
        default=0.015,
        help='lambda for weight decay.')
    parser.add_argument(
        '--lambda_2',
        type=float,
        default=0.015,
        help='lambda for weight decay.')
    parser.add_argument(
        '--iter_early_stop',
        type=int,
        default=1000,
        help='the number of iteration for early stop.')
    parser.add_argument(
        '--seed', type=int, default=1, help='the seed for dataset')

    args = parser.parse_args()
    _train(args)


if __name__ == '__main__':
    main()
