import time
import random
import math
from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from common.datasets.movielens import Dataset
from .batch import BatchManager
from .configs import *
from .model import init_models
from common.utils import str2bool
from common.utils import set_seed
from common.utils import save_results_to_file


def __get_rmse(session, models, x, y, n_batch, args):
    sse, n = 0.0, 0
    for i in args.tqdm(range(0, x.shape[0], n_batch)):
        _x = x[i:i + n_batch]
        _y = y[i:i + n_batch]
        _sse, _n = session.run(
            (models['sse'], models['n']),
            feed_dict={
                models['x']: _x,
                models['y']: _y,
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

    if args.use_tqdm:
        print('>> start validation...')

    n_batch = args.n_batch
    # n_batch = test_x.shape[0]
    assert train_x.shape[0] == valid_x.shape[0]
    assert train_x.shape[0] == test_x.shape[0]
    assert train_x.shape[0] >= n_batch

    if args.print_train_rmse:
        train_rmse = __get_rmse(session, models, train_x, train_x, n_batch,
                                args)
    else:
        train_rmse = 0.0
    valid_rmse = __get_rmse(session, models, valid_x, valid_y, n_batch, args)
    test_rmse = __get_rmse(session, models, test_x, test_y, n_batch, args)
    return train_rmse, valid_rmse, test_rmse


def _train(args):
    dataset = Dataset(args.dataset, args.dataset_split, seed=args.dataset_seed)
    dataset.init_full_matrices()

    set_seed(args.seed)

    n_user, n_item = dataset.get_n_user_and_n_item()
    n_class = dataset.n_class

    train_x = dataset.train_matrix
    valid_x = dataset.train_matrix
    valid_y = dataset.valid_matrix
    test_x = (dataset.train_matrix + dataset.valid_matrix)
    # test_x = dataset.train_matrix
    test_y = dataset.test_matrix

    if args.dataset == Dataset.KIND_MOVIELENS_10M:
        train_x = train_x * 2
        valid_x = valid_x * 2
        valid_y = valid_y * 2
        test_x = test_x * 2
        test_y = test_y * 2

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
    args.K = np.sum(train_x > 0.25) / train_x.shape[0] / 2 * args.K_multiplier
    assert args.K is not None

    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models(n_col, dataset.n_class, args)
    session.run(tf.global_variables_initializer())

    min_valid_rmse = float('inf')
    min_valid_rmse_epoch_idx = 0
    min_test_rmse = float('inf')
    min_test_rmse_epoch_idx = 0
    final_test_rmse = float('inf')
    final_train_rmse = float('inf')
    epoch_tolerance = 0
    learning_rate = args.learning_rate

    # saver = tf.train.Saver()
    # model_file_path = '.models/cfnade/test-seed2'

    train_losses = []
    for epoch_idx in range(1, args.max_epoch):
        for i in args.tqdm(range(train_x.shape[0] // args.n_batch)):
            batch_x, batch_y = batch_manager.get_batch()
            _, train_loss = session.run(
                (models['train_op'], models['loss']),
                feed_dict={
                    models['x']: batch_x,
                    models['y']: batch_y,
                    models['training']: True,
                    models['learning_rate']: learning_rate,
                }, )
            train_losses.append(train_loss)

        train_rmse, valid_rmse, test_rmse = _validate(session, models, args,
                                                      **data_for_validation)
        if math.isnan(train_rmse):
            print('Train RMSE is NaN.')
            assert False

        if min_valid_rmse > valid_rmse:
            min_valid_rmse = valid_rmse
            final_test_rmse = test_rmse
            final_train_rmse = train_rmse
            min_valid_rmse_epoch_idx = epoch_idx
            epoch_tolerance = 0
            # saver.save(session, model_file_path)
        else:
            epoch_tolerance += 1

        if min_test_rmse > test_rmse:
            min_test_rmse = test_rmse
            min_test_rmse_epoch_idx = epoch_idx

        msg = '\n[Epoch %6d]\n' % (epoch_idx)
        msg += '  - train loss: {:12.1f}    (learning_rate: {:.6f}, tolerance: {:d})\n'.format(
            sum(train_losses) / len(train_losses), learning_rate,
            epoch_tolerance)
        if args.print_train_rmse:
            msg += '  - train rmse: {:12.8f}\n'.format(train_rmse)
        msg += '  - valid rmse: {:12.8f}    (min: {:12.8f}, iter: {:d}, final: {:12.8f})\n'.format(
            valid_rmse, min_valid_rmse, min_valid_rmse_epoch_idx,
            final_test_rmse)
        msg += '  - test  rmse: {:12.8f}    (min: {:12.8f}, iter: {:d})\n'.format(
            test_rmse, min_test_rmse, min_test_rmse_epoch_idx)
        print(msg)
        # for i in range(5):
        #     print(["{:8.4f}".format(j) for j in a_i[i]])
        # print('')
        # for i in range(5):
        #     print(["{:8.4f}".format(j) for j in a_o[i]])

        if epoch_tolerance >= args.epoch_early_stop:
            if args.learning_rate_decay < 1:
                if learning_rate < args.min_learning_rate:
                    # learning rate가 min보다 작으면 stop합니다.
                    break
                else:
                    # saver.restore(session, model_file_path)
                    # min learning rate보다 크면, learrning rate를 줄입니다.
                    learning_rate *= args.learning_rate_decay
                    epoch_tolerance = 0
            else:
                # learning_rate decay를 안하면 바로 stop합니다.
                break

    session.close()

    results = []
    results.append(args.use_user_vector)
    results.append(args.sparsity_normalization)
    results.append(args.parameter_sharing)
    results.append(args.activation_fn)
    results.append(args.use_ordinal_cost)
    results.append(args.averaging_possible_choices)
    results.append(args.lambda_1)
    results.append(args.lambda_2)
    results.append(args.K_multiplier)
    results.append(args.K)
    results.append(args.seed)
    results.append(args.dataset_seed)
    results.append(args.epoch_early_stop)
    results.append(args.max_epoch)
    results.append(args.learning_rate)
    results.append(args.learning_rate_decay)
    results.append(args.min_learning_rate)
    results.append(final_test_rmse)
    results.append(min_test_rmse)
    save_results_to_file(
        'cfnade',
        args.dataset,
        results,
        filename='{}_{}_{}'.format(args.use_user_vector, args.activation_fn,
                                   args.dataset_split))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CFNADE-tensorflow')
    parser.add_argument(
        '--dataset',
        type=str,
        default=Dataset.KIND_MOVIELENS_100k,
        # default=Dataset.KIND_MOVIELENS_1M,
        choices=[
            Dataset.KIND_MOVIELENS_100k,
            Dataset.KIND_MOVIELENS_1M,
            Dataset.KIND_MOVIELENS_10M,
        ],
        help='Specify dataset.')
    parser.add_argument(
        '--dataset_split',
        type=str,
        # default='default-10-percent',
        default='5-fold',
        choices=[
            'default-10-percent',
            '5-fold',
        ],
        help='Specify dataset split.')
    parser.add_argument(
        '--activation_fn',
        type=str,
        default='tanh',
        choices=[
            'tanh',
            # 'relu',
        ],
        help='')
    parser.add_argument(
        '--max_epoch', type=int, default=10000000, help='Max Iteration')
    parser.add_argument(
        '--n_hidden_unit',
        type=int,
        # default=100,
        default=500,
        # default=1000,
        help='The number of hidden unit')
    parser.add_argument('--K_multiplier', type=float, default=2, help='')
    parser.add_argument(
        '--parameter_sharing',
        type=str2bool,
        default=True,
        # default=False,
        help='parameter sharing')
    parser.add_argument(
        '--n_batch', type=int, default=512, help='the batch size')
    parser.add_argument(
        '--sparsity_normalization',
        type=str2bool,
        default=True,
        # default=False,
        help='whether normalize input')
    parser.add_argument(
        '--learning_rate',
        type=float,
        # default=0.001,
        # default=0.0005,
        default=0.0002,
        help='learning rate for optimizer.')
    parser.add_argument(
        '--use_ordinal_cost',
        type=str2bool,
        default=True,
        # default=False,
        help='Whether use ordinal cost')
    parser.add_argument(
        '--use_tqdm',
        type=str2bool,
        # default=True,
        default=False,
        help='Whether use tqdm library.')
    parser.add_argument(
        '--use_user_vector',
        type=str2bool,
        # default=True,
        default=False,
        help='Whether use user vector')
    parser.add_argument(
        '--averaging_possible_choices',
        type=str2bool,
        default=True,
        # default=False,
        help='Whether use user averaging_possible_choices')
    parser.add_argument(
        '--lambda_w',
        type=float,
        # default=0.0,
        # default=0.005,
        # default=0.010,
        default=0.015,
        # default=0.020,
        # default=0.025,
        # default=0.030,
        # default=0.035,
        # default=0.040,
        help='lambda for weight decay.')
    parser.add_argument(
        '--epoch_early_stop',
        type=int,
        # default=30,
        default=60,
        # default=100,
        help='the number of iteration for early stop.')
    parser.add_argument(
        '--learning_rate_decay',
        type=float,
        default=0.25,
        # default=0.5,
        help='epoch early stop 만큼 개선 안되면 깎는 learning rate')
    parser.add_argument(
        '--min_learning_rate',
        type=float,
        default=1e-5,
        help='최소 learning rate')
    parser.add_argument('--seed', type=int, default=1, help='the seed')
    parser.add_argument(
        '--dataset_seed',
        type=int,
        default=None,
        # default=1,
        help='the seed for dataset. None 일 때는 seed 값을 따라갑니다.')
    parser.add_argument(
        '--print_train_rmse',
        type=str2bool,
        default=True,
        help='Whether print train_rmse')

    args = parser.parse_args()
    args.lambda_1 = args.lambda_w
    args.lambda_2 = args.lambda_w

    if args.use_tqdm:
        from tqdm import tqdm
        args.tqdm = tqdm
    else:
        args.tqdm = lambda x: x

    if args.dataset_seed is None:
        args.dataset_seed = args.seed
    _train(args)


if __name__ == '__main__':
    main()
