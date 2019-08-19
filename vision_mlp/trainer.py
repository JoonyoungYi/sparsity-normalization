import random
import time

import tensorflow as tf
import numpy as np

from .model import init_models
from common.datasets.vision import Dataset
from common.utils import str2bool, integer_list
from common.utils import set_seed
from common.utils import save_results_to_file
from common.utils import save_results_to_pickle
from common.utils import append_text_results

CODE_NAME = 'vision_mlp'


def __get_output_mean(activations, ys):
    activation = np.exp(activations[-1])
    return sum(activation[i][ys[i]]
               for i in range(activation.shape[0])) / activation.shape[0]


def _save_total_activations(args, session, models, dataset, file_name):
    total_activations = session.run(
        models['layers'],
        feed_dict={
            models['x']: dataset.total_x,
            models['y']: dataset.total_y,
            models['training']: False,
        })
    save_results_to_pickle(CODE_NAME, args.dataset, 'activations',
                           'total_' + file_name, total_activations)


def _log_training(args, session, models, dataset, results):
    if not args.log_training:
        return

    _save_total_activations(args, session, models, dataset, 'after')

    if args.sparsity_injection:
        sparse_test_accuracy, sparse_test_activations = session.run(
            [models['accuracy'], models['layers']],
            feed_dict={
                models['x']: dataset.sparse_test_x,
                models['y']: dataset.sparse_test_y,
                models['training']: False,
            })
        sparse_test_output_mean = __get_output_mean(sparse_test_activations,
                                                    dataset.sparse_test_y)

        dense_test_accuracy, dense_test_activations = session.run(
            [models['accuracy'], models['layers']],
            feed_dict={
                models['x']: dataset.dense_test_x,
                models['y']: dataset.dense_test_y,
                models['training']: False,
            })
        dense_test_output_mean = __get_output_mean(dense_test_activations,
                                                   dataset.dense_test_y)
        results.extend([
            sparse_test_accuracy, sparse_test_output_mean, dense_test_accuracy,
            dense_test_output_mean
        ])

        label = 'w-sn' if args.sparsity_normalization else 'wo-sn'
        save_results_to_pickle(CODE_NAME, args.dataset, 'activations',
                               'sparse_test_' + label, sparse_test_activations)
        save_results_to_pickle(CODE_NAME, args.dataset, 'activations',
                               'dense_test_' + label, dense_test_activations)

    append_text_results(CODE_NAME, args.dataset, 'epochs', 'default',
                        '\t'.join(str(t) for t in results))


def train(args):
    dataset = Dataset(
        args.dataset,
        dataset_seed=args.dataset_seed,
        n_valid=args.n_valid,
        sparsity_injection=args.sparsity_injection,
        sparsity_normalization=args.sparsity_normalization,
        K_prime=args.K_prime)
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

    if args.log_training:
        save_results_to_pickle('vision_mlp', args.dataset, 'test_ys',
                               str(args.dataset_seed), test_y)
        _save_total_activations(args, session, models, dataset, 'before')

    max_valid_acc = 0.0
    max_valid_epoch_idx = 0
    final_test_acc = 0.0

    max_test_acc = 0.0
    max_test_epoch_idx = 0

    startTime = time.time()
    for epoch_idx in range(args.max_epoch):
        # train_accs = []
        # for i in range(0, train_x.shape[0], 5000):
        #     _, train_loss, train_acc = session.run(
        #         [models['train_op'], models['loss'], models['accuracy']],
        #         feed_dict={
        #             models['x']: train_x[i:i + 5000],
        #             models['y']: train_y[i:i + 5000],
        #             models['training']: True,
        #         })
        #     train_accs.append(train_acc)
        # train_acc = sum(train_accs) / len(train_accs)

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

        if max_test_acc + 1e-5 < test_acc:
            max_test_acc = test_acc
            max_test_epoch_idx = epoch_idx

        _log_training(args, session, models, dataset, [
            epoch_idx, train_loss, train_acc, valid_acc, test_acc,
            max_test_acc, max_test_epoch_idx, max_valid_acc,
            max_valid_epoch_idx, final_test_acc
        ])
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
    save_results_to_file(CODE_NAME, args.dataset, results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Vision MLP')
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
        default=800,
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
    # parser.add_argument(
    #     '--dropconnect_rate',
    #     type=float,
    #     default=0.0,
    #     help='drop rate for dropconnect.')
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
        '--sparsity_injection',
        type=str2bool,
        default=True,
        # default=False,
        help='Whether to inject sparsity')
    parser.add_argument(
        '--K_prime',
        type=float,
        default=None,
        help='K prime in the paper. '
        'K_prime이 None으로 입력되면, 자동으로 K_prime을 선정합니다.')
    parser.add_argument(
        '--log_training',
        type=str2bool,
        default=True,
        # default=False,
        help='Whether log training')
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
