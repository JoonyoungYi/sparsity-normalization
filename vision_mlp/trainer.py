import random
import time

import tensorflow as tf
import numpy as np

from .model import init_models
from common.datasets.vision import Dataset
from common.utils import str2bool, integer_list
from common.utils import set_seed
from common.utils import save_results_to_file
from common.utils import save_numpy_results


def _save_activations(args, session, models, dataset, epoch_idx,
                      max_valid_epoch_idx):
    if args.save_activations:
        activations = session.run(
            models['layers'],
            feed_dict={
                models['x']: dataset.total_x,
                models['y']: dataset.total_y,
                models['training']: False,
            })
        activations = np.array(activations[:args.n_hidden_layer])
        save_numpy_results('vision_mlp',
                           args.dataset, 'activations.{}.{}'.format(
                               epoch_idx, max_valid_epoch_idx), activations)


def train(args):
    dataset = Dataset(args)
    train_x = dataset.train_x
    train_y = dataset.train_y
    valid_x = dataset.valid_x
    valid_y = dataset.valid_y
    test_x = dataset.test_x
    test_y = dataset.test_y

    set_seed(args.seed)

    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models(args, dataset.n_in, dataset.n_out)
    session.run(tf.global_variables_initializer())

    _save_activation(args, session, models, dataset, -1, -1)

    max_valid_acc = 0.0
    max_valid_epoch_idx = 0
    final_test_acc = 0.0

    max_test_acc = 0.0
    max_test_epoch_idx = 0

    startTime = time.time()
    for epoch_idx in range(args.max_epoch):
        _, train_loss, train_acc = session.run(
            [models['train_op'], models['loss'], models['accuracy']],
            feed_dict={
                models['x']: train_x,
                models['y']: train_y,
                models['training']: True,
            })

        valid_acc = session.run(
            models['accuracy'],
            feed_dict={
                models['x']: valid_x,
                models['y']: valid_y,
                models['training']: False,
            })
        test_acc = session.run(
            models['accuracy'],
            feed_dict={
                models['x']: test_x,
                models['y']: test_y,
                models['training']: False,
            })

        if max_valid_acc + 1e-5 < valid_acc:
            max_valid_acc = valid_acc
            max_valid_epoch_idx = epoch_idx
            final_test_acc = test_acc
            _save_activations(args, session, models, dataset, epoch_idx,
                              max_valid_epoch_idx)

        if max_test_acc + 1e-5 < test_acc:
            max_test_acc = test_acc
            max_test_epoch_idx = epoch_idx

        print('')
        print('%6d' % epoch_idx, '%.2f' % train_loss, '%.4f' % train_acc,
              '%.4f' % valid_acc, '%.4f' % test_acc)
        print('%6d' % max_valid_epoch_idx, '%.4f' % max_valid_acc,
              '%.4f' % final_test_acc)
        print('%6d' % max_test_epoch_idx, '%.4f' % max_test_acc)

        if epoch_idx > max_valid_epoch_idx + args.epoch_early_stop:
            break

    endTime = time.time() - startTime
    endTime = int(endTime / 60)

    results = []
    results.append(args.seed)
    results.append(args.n_hidden_layer)
    results.append(args.activation_fn)
    results.append(args.sparsity_normalization)
    results.append(args.K_prime)
    results.append(args.n_hidden_unit)
    results.append(final_test_acc)
    results.append(endTime)
    save_results_to_file('vision_mlp', args.dataset, results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument(
        '--dataset',
        type=str,
        # default='mnist',
        default='fashion',
        # default='cifar10',
        # default='cifar100',
        choices=['mnist', 'fashion', 'cifar10', 'cifar100'],
        help='MNIST or Fashion MNIST or CIFAR-10' or 'CIFAR-100')
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
        default=2,
        help='the number of hidden layers.')
    parser.add_argument(
        '--n_hidden_unit',
        type=int,
        default=5,
        help='the number of hidden unit of each layer.')
    parser.add_argument(
        '--activation_fn',
        type=str,
        # default='tanh',
        default='relu',
        # default='sigmoid',
        choices=['relu', 'leaky_relu', 'tanh', 'sigmoid', 'softplus', 'elu'],
        help='the activation function')
    parser.add_argument(
        '--seed', type=int, default=1, help='the seed for training')
    parser.add_argument(
        '--dataset_seed',
        type=int,
        default=1,
        help='the seed for split dataset. the seed for dataset.'
        'None 일 때는 seed 값을 따라갑니다.')
    parser.add_argument(
        '--dropconnect_rate',
        type=float,
        default=0.0,
        help='drop rate for dropconnect.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='learning rate of optimizer')

    parser.add_argument(
        '--sparsity_normalization',
        type=str2bool,
        default=True,
        # default=False,
        help='Whether to apply sparsity normalization to input layer')
    parser.add_argument(
        '--K_prime',
        type=float,
        default=1,
        help='K prime in the paper. '
        'K_prime이 None으로 입력되면, 자동으로 K_prime을 선정합니다.')
    parser.add_argument(
        '--save_activations',
        type=str2bool,
        default=False,
        help='Whether save activations')
    parser.add_argument(
        '--n_valid',
        type=int,
        default=-10000,
        help='음수로 입력되면, train set에서 validation을 떼고, '
        '양수로 입력되면 test set에서 n_valid 갯수만큼 뗍니다. '
        'NeurIPS 2019에 제출했던 버전에서는 -10000을 사용하였고, '
        '리뷰탈 때는 +5000을 일부 실험에서 사용하였습니다.')

    args = parser.parse_args()

    if args.dataset_seed is None:
        args.dataset_seed = args.seed

    train(args)


if __name__ == '__main__':
    main()
