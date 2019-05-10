# References
# - https://www.tensorflow.org/versions/r0.10/tutorials/word2vec/index.html
# - https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
# - https://gist.github.com/yxtay/a94d971955d901c4690129580a4eafb9
# - https://stackoverflow.com/questions/37394970/tensorflow-word2vec-cbow-model
import math

import numpy as np
import tensorflow as tf

from .model import init_models
from .dataset import Dataset
from common.utils import set_seed
from common.utils import str2bool


def _train(args):
    set_seed(args.seed)
    dataset = Dataset(args)
    models = init_models(args)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    session.run(tf.global_variables_initializer())

    sum_loss = 0
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
            # The average loss is an estimate of the loss over the last (args.valid_iter) batches.
            print("Average loss at step ", iter_idx, ": ",
                  sum_loss / args.valid_iter)
            sum_loss = 0

            cross_entropy_loss = session.run(
                models['cross_entropy_loss'],
                feed_dict={
                    models['train_inputs']: dataset.test_inputs[:1000],
                    models['train_labels']: dataset.test_labels[:1000]
                })
            print(math.exp(cross_entropy_loss / 1000))

    session.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Word2Vec')
    parser.add_argument(
        '--dataset',
        type=str,
        default='text8',
        choices=['text8', 'ptb'],
        help='')
    parser.add_argument(
        '--seed', type=int, default=1, help='the seed for dataset')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='integer for batch number.')
    parser.add_argument(
        '--context_window',
        type=int,
        default=1,
        help='context_window. How many words to consider left and right.')
    parser.add_argument(
        '--vocabulary_size', type=int, default=50000, help='vocabulary_size.')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=128,
        help='embedding_size. Dimension of the embedding vector.')
    parser.add_argument(
        '--num_sampled',
        type=int,
        default=64,
        help='num_sampled. Number of negative examples to sample.')
    parser.add_argument(
        '--max_iter', type=int, default=100000, help='Max iter')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1.0,
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
        '--optimizer',
        type=str,
        choices=['sgd', 'adam'],
        default='sgd',
        help='Optimizer')

    # parser.add_argument(
    #     '--n_hidden_unit',
    #     type=int,
    #     default=500,
    #     help='The number of hidden unit')
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
    # parser.add_argument(
    #     '--epoch_early_stop',
    #     type=int,
    #     default=1000,
    #     help='the number of epochation for early stop.')
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
    assert args.batch_size and args.batch_size > 0
    args.context_size = args.context_window * 2
    _train(args)


if __name__ == '__main__':
    main()
