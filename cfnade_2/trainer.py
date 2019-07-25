import time
import random
import math
from datetime import datetime

import numpy as np
import tensorflow as tf

from common.dataset.movielens import Dataset
from .batch import BatchManager
from .configs import *
from .model import init_models
from common.utils import str2bool
from common.utils import set_seed
from common.utils import save_results_to_file


def __get_rmse(session, models, x, y, n_batch):
    sse, n = 0.0, 0
    for i in range(0, x.shape[0], n_batch):
        _sse, _n = session.run(
            (models['sse'], models['n']),
            feed_dict={
                models['x']: x[i:i + n_batch],
                models['y']: y[i:i + n_batch],
                models['training']: False,
            }, )
        n += _n
        sse += _sse
    return math.sqrt(sse / n)


def _validate(
        session,
        models,
        args,
        train_x,
        valid_x,
        valid_y,
        test_x,
        test_y, ):

    # n_batch = args.n_batch
    n_batch = test_x.shape[0]
    assert train_x.shape[0] == valid_x.shape[0]
    assert train_x.shape[0] == test_x.shape[0]
    assert train_x.shape[0] >= n_batch

    train_rmse = __get_rmse(session, models, train_x, train_x, n_batch)
    valid_rmse = __get_rmse(session, models, valid_x, valid_y, n_batch)
    test_rmse = __get_rmse(session, models, test_x, test_y, n_batch)
    return train_rmse, valid_rmse, test_rmse


def _train(args):
    set_seed(args.seed)

    dataset = Dataset(
        args.dataset, Dataset.SPLIT_DEFAULT_10_PERCENT, seed=args.seed)
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

        _, train_loss = session.run(
            (models['train_op'], models['loss']),
            feed_dict={
                models['x']: batch_x,
                models['y']: batch_y,
                models['training']: True,
            }, )
        train_losses.append(train_loss)

        if (iter_idx) % args.iter_validation == 0:
            train_rmse, valid_rmse, test_rmse = _validate(
                session, models, args, **data_for_validation)
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

    results = []
    results.append(args.use_user_vector)
    results.append(args.normalize_input)
    results.append(args.parameter_sharing)
    results.append(args.use_ordinal_cost)
    results.append(args.averaging_possible_choices)
    results.append(args.lambda_1)
    results.append(args.lambda_2)
    results.append(args.seed)
    results.append(final_test_rmse)
    results.append(min_test_rmse)
    save_results_to_file('cfnade', args.dataset, results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CFNADE-tensorflow')
    parser.add_argument(
        '--dataset',
        type=str,
        default=Dataset.KIND_MOVIELENS_100k,
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
        # default=True,
        default=False,
        help='parameter sharing')
    parser.add_argument(
        '--n_batch', type=int, default=512, help='the batch size')
    parser.add_argument(
        '--normalize_input',
        type=str2bool,
        default=True,
        help='whether normalize input')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='learning rate for optimizer.')
    parser.add_argument(
        '--use_ordinal_cost',
        type=str2bool,
        # default=False,
        default=True,
        help='Whether use ordinal cost')
    parser.add_argument(
        '--use_user_vector',
        type=str2bool,
        default=False,
        help='Whether use user vector')
    parser.add_argument(
        '--averaging_possible_choices',
        type=str2bool,
        # default=True,
        default=False,
        help='Whether use user averaging_possible_choices')
    parser.add_argument(
        '--gpu_memory_fraction',
        type=float,
        default=0.25,
        help='per_process_gpu_memory_fraction')

    parser.add_argument(
        '--lambda_w', type=float, default=0.0, help='lambda for weight decay.')
    parser.add_argument(
        '--iter_early_stop',
        type=int,
        default=1000,
        help='the number of iteration for early stop.')
    parser.add_argument(
        '--seed', type=int, default=1, help='the seed for dataset')

    args = parser.parse_args()
    args.lambda_1 = args.lambda_w
    args.lambda_2 = args.lambda_w
    _train(args)


if __name__ == '__main__':
    main()

# default+oc+ws+apc: python3 -m cfnade_2.trainer --dataset=movielens-1m --parameter_sharing=True --normalize_input=False --use_ordinal_cost=True --use_user_vector=False --averaging_possible_choices=True --gpu_memory_fraction=0.3 --lambda_w=
# default+oc: python3 -m cfnade_2.trainer --dataset=movielens-1m --parameter_sharing=False --normalize_input=False --use_ordinal_cost=True --use_user_vector=False --averaging_possible_choices=False --gpu_memory_fraction=0.3 --lambda_w=0.01
# normal+oc: python3 -m cfnade_2.trainer --dataset=movielens-1m --parameter_sharing=False --normalize_input=True --use_ordinal_cost=True --use_user_vector=False --averaging_possible_choices=False --gpu_memory_fraction=0.3 --lambda_w=0.01

# default+oc+ws+apc: python3 -m cfnade_2.trainer --dataset=movielens-1m --parameter_sharing=True --normalize_input=False --use_ordinal_cost=True --use_user_vector=True --averaging_possible_choices=True --gpu_memory_fraction=0.3 --lambda_w=
# default+oc: python3 -m cfnade_2.trainer --dataset=movielens-1m --parameter_sharing=False --normalize_input=False --use_ordinal_cost=True --use_user_vector=True --averaging_possible_choices=False --gpu_memory_fraction=0.3 --lambda_w=
# normal+oc: python3 -m cfnade_2.trainer --dataset=movielens-1m --parameter_sharing=False --normalize_input=True --use_ordinal_cost=True --use_user_vector=True --averaging_possible_choices=False --gpu_memory_fraction=0.3 --lambda_w=
