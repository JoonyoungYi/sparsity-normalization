import random

import tensorflow as tf
import numpy as np

from .model import init_models
from .dataset import Dataset
from common.utils import str2bool, integer_list
from common.utils import set_seed
from common.utils import save_results_to_file

import time


def train(args, seed):
    startTime = time.time()

    set_seed(seed)
    dataset = Dataset(args)
    train_x = dataset.train_x
    train_y = dataset.train_y
    valid_x = dataset.valid_x
    valid_y = dataset.valid_y
    test_x = dataset.test_x
    test_y = dataset.test_y
    total_x = dataset.total_x
    total_y = dataset.total_y

    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models(args, dataset.n_in, dataset.n_out)
    session.run(tf.global_variables_initializer())

    max_valid_acc = 0.0
    max_valid_iter_idx = 0
    final_test_acc = 0.0

    max_test_acc = 0.0
    max_test_iter_idx = 0

    if args.plot_activations:
        activations = session.run(
            models['layers'],
            feed_dict={
                models['x']: total_x,
                models['y']: total_y,
                models['training']: False,
            })
        activations = np.array(activations[:args.n_hidden_layer])
        np.save('.results/mnist/{}_activations_before_training'.format(
            args.dataset), activations)

    for iter_idx in range(args.max_epoch):
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

        print('')
        print('%6d' % iter_idx, '%.2f' % train_loss, '%.4f' % train_acc,
              '%.4f' % valid_acc, '%.4f' % test_acc)
        print('%6d' % max_valid_iter_idx, '%.4f' % max_valid_acc,
              '%.4f' % final_test_acc)
        print('%6d' % max_test_iter_idx, '%.4f' % max_test_acc)

        if max_valid_acc + 1e-5 < valid_acc:
            max_valid_acc = valid_acc
            max_valid_iter_idx = iter_idx
            final_test_acc = test_acc

        if max_test_acc + 1e-5 < test_acc:
            max_test_acc = test_acc
            max_test_iter_idx = iter_idx

        if iter_idx > max_valid_iter_idx + args.epoch_early_stop:
            if args.plot_activations:
                activations = session.run(
                    models['layers'],
                    feed_dict={
                        models['x']: total_x,
                        models['y']: total_y,
                        models['training']: False,
                    })
                activations = np.array(activations[:args.n_hidden_layer])
                np.save('.results/mnist/{}_activations_after_training'.format(
                    args.dataset), activations)
            break

    endTime = time.time() - startTime
    endTime = int(endTime / 60)

    results = []
    results.append(args.seed)
    results.append(args.n_hidden_layer)
    results.append(args.grey_scale)
    results.append(args.sparsity_normalization)
    results.append(args.n_hidden_unit)
    results.append(final_test_acc)
    results.append(endTime)
    save_results_to_file('mnist', args.dataset, results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        # default='fashion',
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
        default=512,
        help='the number of hidden unit of each layer.')
    parser.add_argument(
        '--activation_fn',
        type=str,
        # default='tanh',
        default='relu',
        # default='sigmoid',
        choices=['relu', 'tanh', 'sigmoid'],
        help='the activation function')
    parser.add_argument(
        '--seed', type=int, default=1, help='the seed for training')
    parser.add_argument(
        '--drop_rate', type=float, default=0.0, help='drop rate.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-2,
        help='learning rate of optimizer')

    parser.add_argument(
        '--sparsity_normalization',
        type=str2bool,
        # default=True,
        default=False,
        help='Whether to apply sparsity normalization to input layer')
    parser.add_argument(
        '--K_prime', type=float, default=1, help='K prime in the paper')
    parser.add_argument(
        '--plot_activations',
        type=str2bool,
        default=False,
        help='Whether plot raw activations')
    parser.add_argument(
        '--grey_scale',
        type=str2bool,
        default=False,
        help='Whether use grey scaling in cifar dataset.')

    args = parser.parse_args()
    if not (args.dataset == 'cifar10' or
            args.dataset == 'cifar100') and args.grey_scale:
        print('grey_scale only can be applied to cifar dataset!')
        assert not args.grey_scale or args.dataset == 'cifar10' or args.dataset == 'cifar100'

    train(args, args.seed)


if __name__ == '__main__':
    main()
