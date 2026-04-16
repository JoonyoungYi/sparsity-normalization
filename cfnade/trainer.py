import math
import argparse

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from common.datasets.movielens import Dataset
from .batch import BatchManager
from .model import init_models
from common.utils import str2bool, set_seed, save_results_to_file


def _get_rmse(session, models, x, y, n_batch):
    sse, n = 0.0, 0
    for i in range(0, x.shape[0], n_batch):
        _x, _y = x[i:i + n_batch], y[i:i + n_batch]
        _sse, _n = session.run(
            (models['sse'], models['n']),
            feed_dict={models['x']: _x, models['y']: _y, models['training']: False})
        sse += _sse
        n += _n
    return math.sqrt(sse / n)


def _validate(session, models, n_batch, train_x, valid_x, valid_y, test_x, test_y):
    train_rmse = _get_rmse(session, models, train_x, train_x, n_batch)
    valid_rmse = _get_rmse(session, models, valid_x, valid_y, n_batch)
    test_rmse = _get_rmse(session, models, test_x, test_y, n_batch)
    return train_rmse, valid_rmse, test_rmse


def _train(args):
    dataset = Dataset(args.dataset, args.dataset_split, seed=args.dataset_seed)
    dataset.init_full_matrices()
    set_seed(args.seed)

    n_user, n_item = dataset.get_n_user_and_n_item()

    train_x = dataset.train_matrix
    valid_y = dataset.valid_matrix
    test_x = dataset.train_matrix + dataset.valid_matrix
    test_y = dataset.test_matrix

    if args.dataset == Dataset.KIND_MOVIELENS_10M:
        train_x = train_x * 2
        valid_y = valid_y * 2
        test_x = test_x * 2
        test_y = test_y * 2

    if not args.use_user_vector:
        train_x = train_x.T
        valid_y = valid_y.T
        test_x = test_x.T
        test_y = test_y.T
        n_col = n_user
    else:
        n_col = n_item

    valid_x = train_x

    args.K = np.sum(train_x > 0.25) / train_x.shape[0] / 2 * args.K_multiplier

    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    models = init_models(n_col, dataset.n_class, args)
    session.run(tf.global_variables_initializer())

    batch_manager = BatchManager(train_x, args)

    min_valid_rmse = float('inf')
    min_valid_rmse_epoch = 0
    final_test_rmse = float('inf')
    min_test_rmse = float('inf')
    epoch_tolerance = 0
    learning_rate = args.learning_rate

    for epoch_idx in range(1, args.max_epoch):
        train_losses = []
        for _ in range(train_x.shape[0] // args.n_batch):
            bx, by = batch_manager.get_batch()
            _, loss = session.run(
                (models['train_op'], models['loss']),
                feed_dict={
                    models['x']: bx, models['y']: by,
                    models['training']: True,
                    models['learning_rate']: learning_rate,
                })
            train_losses.append(loss)

        train_rmse, valid_rmse, test_rmse = _validate(
            session, models, args.n_batch,
            train_x, valid_x, valid_y, test_x, test_y)

        if math.isnan(train_rmse):
            print('Train RMSE is NaN. Stopping.')
            break

        if valid_rmse < min_valid_rmse:
            min_valid_rmse = valid_rmse
            final_test_rmse = test_rmse
            min_valid_rmse_epoch = epoch_idx
            epoch_tolerance = 0
        else:
            epoch_tolerance += 1

        min_test_rmse = min(test_rmse, min_test_rmse)

        print(
            f'\n[Epoch {epoch_idx:6d}]  lr={learning_rate:.6f}  loss={sum(train_losses)/len(train_losses):.1f}'
            f'\n  train: {train_rmse:.6f}'
            f'\n  valid: {valid_rmse:.6f}  (best: {min_valid_rmse:.6f} @ {min_valid_rmse_epoch}, test: {final_test_rmse:.6f})'
            f'\n  test:  {test_rmse:.6f}  (best: {min_test_rmse:.6f})'
        )

        if epoch_tolerance >= args.epoch_early_stop:
            if args.learning_rate_decay < 1 and learning_rate >= args.min_learning_rate:
                learning_rate *= args.learning_rate_decay
                epoch_tolerance = 0
            else:
                break

    session.close()

    results = [
        args.use_user_vector, args.sparsity_normalization, args.parameter_sharing,
        args.use_ordinal_cost, args.averaging_possible_choices,
        args.lambda_1, args.K_multiplier, args.K,
        args.seed, args.dataset_seed, final_test_rmse, min_test_rmse,
    ]
    save_results_to_file(
        'cfnade', args.dataset, results,
        filename=f'{args.use_user_vector}_{args.dataset_split}')


def main():
    parser = argparse.ArgumentParser(description='CF-NADE with Sparsity Normalization')
    parser.add_argument('--dataset', type=str, default=Dataset.KIND_MOVIELENS_1M,
                        choices=[Dataset.KIND_MOVIELENS_100k,
                                 Dataset.KIND_MOVIELENS_1M,
                                 Dataset.KIND_MOVIELENS_10M])
    parser.add_argument('--dataset_split', type=str, default='default-10-percent',
                        choices=['default-10-percent', '5-fold'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset_seed', type=int, default=None)
    parser.add_argument('--max_epoch', type=int, default=10000000)
    parser.add_argument('--n_hidden_unit', type=int, default=500)
    parser.add_argument('--n_batch', type=int, default=512)
    parser.add_argument('--epoch_early_stop', type=int, default=60)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate_decay', type=float, default=0.25)
    parser.add_argument('--min_learning_rate', type=float, default=1e-5)
    parser.add_argument('--lambda_w', type=float, default=0.015)
    parser.add_argument('--K_multiplier', type=float, default=2.0)
    parser.add_argument('--activation_fn', type=str, default='tanh', choices=['tanh'])
    parser.add_argument('--sparsity_normalization', type=str2bool, default=False,
                        help='Sparsity Normalization 적용 여부')
    parser.add_argument('--parameter_sharing', type=str2bool, default=True)
    parser.add_argument('--use_ordinal_cost', type=str2bool, default=True)
    parser.add_argument('--use_user_vector', type=str2bool, default=False)
    parser.add_argument('--averaging_possible_choices', type=str2bool, default=True)
    parser.add_argument('--print_train_rmse', type=str2bool, default=True)
    args = parser.parse_args()

    args.lambda_1 = args.lambda_2 = args.lambda_w
    if args.dataset_seed is None:
        args.dataset_seed = args.seed

    _train(args)


if __name__ == '__main__':
    main()
