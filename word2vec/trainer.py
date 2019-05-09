# References
# - https://www.tensorflow.org/versions/r0.10/tutorials/word2vec/index.html
# - https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
# - https://gist.github.com/yxtay/a94d971955d901c4690129580a4eafb9
# - https://stackoverflow.com/questions/37394970/tensorflow-word2vec-cbow-model
import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from .model import init_models
from common.utils import set_seed
from common.utils import str2bool


def _train(args):
    set_seed(args.seed)

    # Step 1: Download the data.
    url = 'http://mattmahoney.net/dc/'

    def maybe_download(filename, expected_bytes):
        """Download a file if not present, and make sure it's the right size."""
        if not os.path.exists(filename):
            filename, _ = urllib.request.urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified', filename)
        else:
            print(statinfo.st_size)
            raise Exception('Failed to verify ' + filename +
                            '. Can you get to it with a browser?')
        return filename

    filename = maybe_download('text8.zip', 31344016)

    # Read the data into a list of strings.
    def read_data(filename):
        """Extract the first file enclosed in a zip file as a list of words"""
        with zipfile.ZipFile(filename) as f:
            data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return data

    words = read_data(filename)
    print('Data size', len(words))

    # Step 2: Build the dictionary and replace rare words with UNK token.
    vocabulary_size = 50000

    def build_dataset(words):
        count = [['UNK', -1]]
        count.extend(
            collections.Counter(words).most_common(vocabulary_size - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary

    data, count, dictionary, reverse_dictionary = build_dataset(words)
    del words  # Hint to reduce memory.
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    # Step 3: Function to generate a training batch for the skip-gram model.
    def generate_batch(batch_size, context_window, data_index=0):
        # all context tokens should be used, hence no associated num_skips argument
        context_size = 2 * context_window
        batch = np.ndarray(shape=(batch_size, context_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * context_window + 1  # [ context_window target context_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        for i in range(batch_size):
            # context tokens are just all the tokens in buffer except the target
            batch[i, :] = [
                token for idx, token in enumerate(buffer)
                if idx != context_window
            ]
            labels[i, 0] = buffer[context_window]
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        return batch, labels

    batch, labels = generate_batch(batch_size=8, context_window=1)
    for i in range(8):
        print(batch[i, 0], reverse_dictionary[batch[i, 0]], batch[i, 1],
              reverse_dictionary[batch[i, 1]], '->', labels[i, 0],
              reverse_dictionary[labels[i, 0]])

    # Step 4: Build and train a CBOW model.
    models = init_models(args)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    session.run(tf.global_variables_initializer())

    average_loss = 0
    for step in range(args.max_iter):
        batch_inputs, batch_labels = generate_batch(args.batch_size,
                                                    args.context_window)
        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run(
            [models['train_op'], models['loss']],
            feed_dict={
                models['train_inputs']: batch_inputs,
                models['train_labels']: batch_labels
            })
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

    session.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Word2Vec')
    parser.add_argument(
        '--seed', type=int, default=1, help='the seed for dataset')
    parser.add_argument(
        '--n_batch', type=int, default=128, help='integer for batch number.')
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

    # parser.add_argument(
    #     '--epoch_validation',
    #     type=int,
    #     default=1,
    #     help='epoch unit for validation')
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
    #     '--normalize_input',
    #     type=str2bool,
    #     default=False,
    #     help='Whether normalize input.')
    # parser.add_argument(
    #     '--use_user_vector',
    #     type=str2bool,
    #     # default=True,
    #     default=False,
    #     help='Whether use user vector.')
    # parser.add_argument(
    #     '--learning_rate',
    #     type=float,
    #     default=1e-4,
    #     help='learning rate for optimizer.')
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
    #     '--optimizer',
    #     type=str,
    #     choices=['rprop', 'adam'],
    #     default='adam',
    #     help='Optimizer')
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
    assert args.n_batch and args.n_batch > 0
    args.context_size = args.context_window * 2
    _train(args)


if __name__ == '__main__':
    main()
