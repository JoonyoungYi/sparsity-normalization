import random

import tensorflow as tf
import numpy as np

from .model import init_models
from .dataset import Dataset
from common.utils import str2bool, integer_list
from common.utils import set_seed
from common.utils import save_results_to_file


def train(args, seed):
    set_seed(seed)
    dataset = Dataset(args)
    train_x = dataset.train_x
    train_y = dataset.train_y
    valid_x = dataset.valid_x
    valid_y = dataset.valid_y
    test_x = dataset.test_x
    test_y = dataset.test_y
    sparse_test_x = dataset.sparse_test_x
    sparse_test_y = dataset.sparse_test_y
    dense_test_x = dataset.dense_test_x
    dense_test_y = dataset.dense_test_y
    total_x = dataset.total_x
    total_y = dataset.total_y

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models(args, dataset.n_in, dataset.n_out)
    session.run(tf.global_variables_initializer())

    max_valid_acc = 0.0
    max_valid_iter_idx = 0
    final_test_acc = 0.0

    max_test_acc = 0.0
    max_test_iter_idx = 0

    if args.plot_raw_activations:
        raw_activations = session.run(
            [models['raw_activations']],
            feed_dict={
                models['x']: total_x,
                models['y']: total_y,
                models['training']: True,
            })
        raw_activations = np.array(raw_activations[0][:2])
        np.save('results/mnist/{}_raw_activations_before_training'.format(
            args.dataset), raw_activations)

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

            test_activations = session.run(
                models['activations'],
                feed_dict={
                    models['x']: test_x,
                    models['training']: False,
                })
            test_variances = session.run(
                models['variances'],
                feed_dict={
                    models['x']: test_x,
                    models['training']: False,
                })

            test_sparse_activations, test_sparse_acc = session.run(
                [models['activations'], models['accuracy']],
                feed_dict={
                    models['x']: sparse_test_x,
                    models['y']: sparse_test_y,
                    models['training']: False,
                })
            test_dense_activations, test_dense_acc = session.run(
                [models['activations'], models['accuracy']],
                feed_dict={
                    models['x']: dense_test_x,
                    models['y']: dense_test_y,
                    models['training']: False,
                })

        print(test_sparse_activations)
        print(test_activations)
        # print(test_variances)
        print(test_dense_activations)
        print(
            '{} {} {}'.format(test_sparse_acc, final_test_acc, test_dense_acc))

        if max_test_acc + 1e-5 < test_acc:
            max_test_acc = test_acc
            max_test_iter_idx = iter_idx

        if iter_idx > max_valid_iter_idx + args.epoch_early_stop:
            if args.plot_raw_activations:
                raw_activations = session.run(
                    [models['raw_activations']],
                    feed_dict={
                        models['x']: total_x,
                        models['y']: total_y,
                        models['training']: True,
                    })
                raw_activations = np.array(raw_activations[0][:2])
                np.save(
                    'results/mnist/{}_raw_activations_after_training'.format(
                        args.dataset), raw_activations)
            break

    results = []
    results.append(args.seed)
    results.append(args.n_hidden_layer)
    results.append(args.inject_sparsity)
    results.append(args.grey_scale)
    results.append(args.normalize_input)
    results.append(args.normalize_layer_indexes)
    results.append(args.ignore_bias_layer_indexes)
    results.append(args.n_hidden_unit)
    results.append(test_sparse_acc)
    results.append(final_test_acc)
    results.append(test_dense_acc)
    results.append(test_sparse_activations[-1])
    results.append(test_activations[-1])
    results.append(test_dense_activations[-1])
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
        '--gpu_memory_fraction',
        type=float,
        default=0.3,
        help='per_process_gpu_memory_fraction')
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
        '--inject_sparsity',
        type=str2bool,
        # default=True,
        default=False,
        help='Whether inject sparsity.')

    parser.add_argument(
        '--normalize_input',
        type=str2bool,
        # default=True,
        default=False,
        help='Whether normalize input layers')
    parser.add_argument(
        '--normalize_layer_indexes',
        type=integer_list,
        default=[],
        help=
        'normalize를 할 hidden layer index들의 리스트. 쉼표로 구분. 1,2,3 인 경우 1,2,3번 hidden layer를 normalize. 1 인 경우 1번째 hidden layer를 normalize.'
    )
    parser.add_argument(
        '--approximate_L0',
        type=str2bool,
        # default=True,
        default=False,
        help='Whether approximate L0 norm')
    parser.add_argument(
        '--ignore_bias_layer_indexes',
        type=integer_list,
        default=[],
        help='bias를 무시할 hidden layer index들의 리스트. 쉼표로 구분합니다. ')
    parser.add_argument(
        '--plot_raw_activations',
        type=str2bool,
        default=False,
        help='Whether plot raw activations')
    parser.add_argument(
        '--grey_scale',
        type=str2bool,
        default=False,
        help='Whether use grey scaling in cifar dataset.')

    args = parser.parse_args()

    for layer_index in args.normalize_layer_indexes:
        assert layer_index <= args.n_hidden_layer
        assert layer_index > 0
    for layer_index in args.ignore_bias_layer_indexes:
        assert layer_index <= args.n_hidden_layer
        assert layer_index > 0
    if not (args.dataset == 'cifar10' or
            args.dataset == 'cifar100') and args.grey_scale:
        print('grey_scale only can be applied to cifar dataset!')
        assert not args.grey_scale or args.dataset == 'cifar10' or args.dataset == 'cifar100'

    train(args, args.seed)


if __name__ == '__main__':
    main()
