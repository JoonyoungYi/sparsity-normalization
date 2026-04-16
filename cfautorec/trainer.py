import math
import argparse

import numpy as np
import tensorflow as tf

from common.datasets.movielens import Dataset
from .model import init_models
from .batch import Batch
from common.utils import str2bool, set_seed, append_text_results

IMPUTATION_CHOICES = ['zero', 'mean', 'dropout', 'batch-norm', 'layer-norm']


def _get_rmse(session, models, batch, key, args):
    func = getattr(batch, 'batch_' + key)

    if args.n_batch is None:
        x, y = func()
        return session.run(models['rmse'], feed_dict={
            models['x']: x, models['y']: y, models['training']: False,
        })
    else:
        sse, n = 0.0, 0.0
        for iter_idx in range(batch.n_row // args.n_batch):
            x, y = func(iter_idx)
            _rmse, _n = session.run((models['rmse'], models['n']), feed_dict={
                models['x']: x, models['y']: y, models['training']: False,
            })
            n += _n
            sse += math.pow(_rmse, 2) * _n
        return math.sqrt(sse / n)


def _validate(session, models, batch, args):
    train_rmse = _get_rmse(session, models, batch, 'train', args)
    valid_rmse = _get_rmse(session, models, batch, 'valid', args)
    test_rmse = _get_rmse(session, models, batch, 'test', args)
    return train_rmse, valid_rmse, test_rmse


def _train(args):
    dataset = Dataset(args.dataset, Dataset.SPLIT_DEFAULT_10_PERCENT, seed=args.dataset_seed)
    set_seed(args.seed)

    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    batch = Batch(dataset, args)
    models = init_models(batch.n_col, args)
    session.run(tf.global_variables_initializer())

    min_valid_rmse = float('inf')
    min_valid_rmse_epoch = 0
    final_test_rmse = float('inf')
    final_train_rmse = float('inf')
    min_test_rmse = float('inf')
    min_test_rmse_epoch = 0

    for epoch_idx in range(1, args.max_epoch + 1):
        if args.n_batch is None:
            bx, by = batch.batch_train()
            session.run(models['train_op'], feed_dict={
                models['x']: bx, models['y']: by, models['training']: True,
            })
        else:
            for i in range(batch.n_row // args.n_batch):
                bx, by = batch.batch_train(i)
                session.run(models['train_op'], feed_dict={
                    models['x']: bx, models['y']: by, models['training']: True,
                })

        if epoch_idx % args.epoch_validation == 0:
            train_rmse, valid_rmse, test_rmse = _validate(session, models, batch, args)

            if train_rmse < 0.1:
                break

            if valid_rmse < min_valid_rmse - 1e-5:
                min_valid_rmse = valid_rmse
                final_test_rmse = test_rmse
                final_train_rmse = train_rmse
                min_valid_rmse_epoch = epoch_idx

            if test_rmse < min_test_rmse - 1e-5:
                min_test_rmse = test_rmse
                min_test_rmse_epoch = epoch_idx

            print(
                f'\n[Epoch {epoch_idx:6d}]'
                f'\n  train: {train_rmse:.6f}'
                f'\n  valid: {valid_rmse:.6f}  (best: {min_valid_rmse:.6f} @ {min_valid_rmse_epoch}, test: {final_test_rmse:.6f})'
                f'\n  test:  {test_rmse:.6f}  (best: {min_test_rmse:.6f} @ {min_test_rmse_epoch})'
            )

            if epoch_idx > min_valid_rmse_epoch + args.epoch_early_stop:
                break

    session.close()

    results = [
        args.dataset_seed, args.seed, args.dropout_rate, args.n_hidden_unit,
        args.optimizer, args.n_batch, args.lambda_w,
        final_train_rmse, min_valid_rmse, final_test_rmse, min_test_rmse,
    ]
    label = (f'sn{args.normalize_input}_imp{args.imputation}'
             f'_K{args.K_denominator}_uv{args.use_user_vector}')
    append_text_results('autorec', args.dataset, args.activation_fn, label,
                        '\t'.join(str(r) for r in results))


def main():
    parser = argparse.ArgumentParser(description='AutoRec with Sparsity Normalization')
    parser.add_argument('--dataset', type=str, default=Dataset.KIND_MOVIELENS_100k,
                        choices=[Dataset.KIND_MOVIELENS_100k,
                                 Dataset.KIND_MOVIELENS_1M,
                                 Dataset.KIND_MOVIELENS_10M])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset_seed', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=100000)
    parser.add_argument('--epoch_validation', type=int, default=1)
    parser.add_argument('--epoch_early_stop', type=int, default=1000)
    parser.add_argument('--n_hidden_unit', type=int, default=500)
    parser.add_argument('--n_hidden_layer', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--lambda_w', type=float, default=500.0,
                        help='L2 regularization coefficient')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'rprop'])
    parser.add_argument('--activation_fn', type=str, default='sigmoid',
                        choices=['relu', 'sigmoid', 'identity', 'tanh',
                                 'leaky_relu', 'relu6', 'elu', 'softplus'])
    parser.add_argument('--imputation', type=str, default='layer-norm',
                        choices=IMPUTATION_CHOICES)
    parser.add_argument('--normalize_input', type=str2bool, default=False,
                        help='Sparsity Normalization 적용 여부')
    parser.add_argument('--K_denominator', type=float, default=500.0,
                        help='SN의 K 값 (normalize_input=True 시 사용)')
    parser.add_argument('--use_user_vector', type=str2bool, default=True,
                        help='True: user 기준, False: item 기준')
    parser.add_argument('--sparse_tensor', type=str2bool, default=False)
    parser.add_argument('--clipping', type=str2bool, default=False)
    parser.add_argument('--standardization', type=str2bool, default=False)
    parser.add_argument('--n_batch', type=int, default=0,
                        help='0=full batch')
    args = parser.parse_args()

    args.n_batch = None if args.n_batch == 0 else args.n_batch
    assert args.n_batch is None or args.n_batch > 0

    if args.sparse_tensor:
        assert args.optimizer != 'rprop'
        assert args.imputation == 'zero'
    if args.imputation != 'zero':
        assert not args.normalize_input

    _train(args)


if __name__ == '__main__':
    main()
