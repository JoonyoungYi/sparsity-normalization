# References
# - https://www.tensorflow.org/versions/r0.10/tutorials/word2vec/index.html
# - https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
# - https://gist.github.com/yxtay/a94d971955d901c4690129580a4eafb9
# - https://stackoverflow.com/questions/37394970/tensorflow-word2vec-cbow-model
# ex.
# python -m word2vec.trainer --dataset=ptb --gpu_memory_faction=0.1 --context_window=3
# python -m word2vec.trainer --dataset=ptb --gpu_memory_faction=0.1 --normalize_input=False --context_window=1
import math

import numpy as np
import tensorflow as tf

from .model import init_models
from .dataset import Dataset
from common.utils import set_seed
from common.utils import str2bool


def ___get_performances(session, models, test_inputs, test_labels, tqdm, args):
    n_valid_batch = args.batch_size
    cross_entropy_loss = 0.0
    n_correct = 0
    for i in tqdm(range(0, test_inputs.shape[0], n_valid_batch)):
        # print(i + n_valid_batch, n_correct)
        # input('')
        _loss, _n_correct = session.run(
            (models['loss'], models['n_correct']),
            feed_dict={
                models['train_inputs']: test_inputs[i:i + n_valid_batch],
                models['train_labels']: test_labels[i:i + n_valid_batch]
            })
        # print(_loss)
        if math.isnan(_loss):
            raise Exception('_loss is NaN! T.T')
        cross_entropy_loss += _loss
        n_correct += _n_correct

    perplexity = math.exp(cross_entropy_loss / test_inputs.shape[0])
    accuracy = n_correct / test_inputs.shape[0]
    return perplexity, accuracy


def __validate(session, models, dataset, tqdm, args):
    valid_perplexity, valid_acc = ___get_performances(
        session, models, dataset.valid_inputs, dataset.valid_labels, tqdm,
        args)
    test_perplexity, test_acc = ___get_performances(
        session, models, dataset.test_inputs, dataset.test_labels, tqdm, args)
    return valid_perplexity, valid_acc, test_perplexity, test_acc


def _train(args):
    set_seed(args.seed)
    dataset = Dataset(args)
    models = init_models(args)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    session.run(tf.global_variables_initializer())

    if args.use_tqdm:
        try:
            from tqdm import tqdm
        except:
            raise Exception('tqdm package required.')
    else:
        tqdm = lambda x: x

    sum_loss = 0
    max_valid_acc = 0.0
    max_valid_iter = -1
    final_test_acc = 0.0
    max_test_acc = 0.0
    max_test_iter = -1
    for iter_idx in range(args.max_iter):
        batch_inputs, batch_labels = dataset.generate_batch()
        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss = session.run(
            [models['train_op'], models['loss']],
            feed_dict={
                models['train_inputs']: batch_inputs,
                models['train_labels']: batch_labels
            })
        sum_loss += loss

        if iter_idx % args.valid_iter == 0 and iter_idx > 0:
            print('')
            # The average loss is an estimate of the loss over the last (args.valid_iter) batches.
            train_loss = float(sum_loss / args.valid_iter)
            sum_loss = 0
            valid_perplexity, valid_acc, test_perplexity, test_acc = __validate(
                session, models, dataset, tqdm, args)

            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                max_valid_iter = iter_idx
                final_test_acc = test_acc

            if test_acc > max_test_acc:
                max_test_acc = test_acc
                max_test_iter = iter_idx

            print(">>", iter_idx, '%4f' % train_loss)
            print(" -", '%4f' % valid_acc, '%4f' % max_valid_acc,
                  max_valid_iter)
            print(" -", '%4f' % test_acc, '%4f' % final_test_acc,
                  '%4f' % max_test_acc, max_test_iter)

        if iter_idx > max_valid_iter + args.iter_early_stop:
            break

    session.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Word2Vec')
    parser.add_argument(
        '--dataset',
        type=str,
        default='text8',
        choices=[
            'text8',  # text8 dataset
            'ptb',  # pentree bank dataset
        ],
        help='')
    parser.add_argument(
        '--seed', type=int, default=1, help='the seed for dataset')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='integer for batch number.')
    parser.add_argument(
        '--context_window',
        type=int,
        default=1,
        help='context_window. How many words to consider left and right.')
    parser.add_argument(
        '--vocabulary_size',
        type=int,
        # default=50000,
        default=10000,
        help='vocabulary_size.')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=640,
        help='embedding_size. Dimension of the embedding vector.')
    parser.add_argument(
        '--num_sampled',
        type=int,
        default=1000,
        help='num_sampled. Number of negative examples to sample.')
    parser.add_argument(
        '--max_iter', type=int, default=10000000, help='Max iter')
    parser.add_argument(
        '--learning_rate',
        type=float,
        # default=1e-2,
        default=1e-4,
        help='learning rate for optimizer.')
    parser.add_argument(
        '--valid_iter',
        type=int,
        default=1000,
        help='iter unit for validation')
    parser.add_argument(
        '--normalize_input',
        type=str2bool,
        default=True,
        help='Whether normalize input.')
    parser.add_argument(
        '--use_tqdm', type=str2bool, default=False, help='Whether use tqdm')
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['sgd', 'adam'],
        # default='sgd',
        default='adam',
        help='Optimizer')
    parser.add_argument(
        '--iter_early_stop',
        type=int,
        default=100000,
        help='the number of iteration for early stop.')
    parser.add_argument(
        '--gpu_memory_fraction',
        type=float,
        default=0.5,
        help='per_process_gpu_memory_fraction')

    # parser.add_argument(
    #     '--n_hidden_layer',
    #     type=int,
    #     default=1,
    #     help='The number of hidden unit')
    # parser.add_argument(
    #     '--use_user_vector',
    #     type=str2bool,
    #     # default=True,
    #     default=False,
    #     help='Whether use user vector.')
    #
    # parser.add_argument(
    #     '--lambda_w1',
    #     type=float,
    #     default=0.0,
    #     help='lambda for weight decay.')
    # parser.add_argument(
    #     '--lambda_w2',
    #     type=float,
    #     default=0.0,
    #     help='lambda for weight decay.')
    # parser.add_argument(
    #     '--dataset',
    #     type=str,
    #     choices=[
    #         Dataset.KIND_MOVIELENS_100k,
    #         Dataset.KIND_MOVIELENS_1M,
    #         Dataset.KIND_MOVIELENS_10M,
    #     ],
    #     default=Dataset.KIND_MOVIELENS_100k,
    #     # default=Dataset.KIND_MOVIELENS_1M,
    #     # default=Dataset.KIND_MOVIELENS_10M,
    #     help='Optimizer')
    # parser.add_argument(
    #     '--sparse_tensor',
    #     type=str2bool,
    #     default=True,
    #     help='whether use sparse tensor.')

    args = parser.parse_args()
    assert args.context_window > 0
    assert args.batch_size and args.batch_size > 0
    args.context_size = args.context_window * 2
    assert args.dataset == 'ptb'
    if args.dataset == 'ptb':
        assert args.vocabulary_size == 10000
    _train(args)


if __name__ == '__main__':
    main()
