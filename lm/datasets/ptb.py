# REF: https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb
import collections
import os

import tensorflow as tf

from common.dataset_utils import get_file_path


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def _ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)

    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)

    eos_id = word_to_id['<eos>']
    unk_id = word_to_id['<unk>']
    return train_data, valid_data, test_data, word_to_id


def _init_files():
    file_path = get_file_path(
        'ptb.train.txt',
        'ptb',
        url=
        'https://raw.githubusercontent.com/JoonyoungYi/datasets/master/ptb/ptb.train.txt',
        hash='f26c4b92c5fdc7b3f8c7cdcb991d8420')
    assert file_path

    file_path = get_file_path(
        'ptb.valid.txt',
        'ptb',
        url=
        'https://raw.githubusercontent.com/JoonyoungYi/datasets/master/ptb/ptb.valid.txt',
        hash='aa0affc06ff7c36e977d7cd49e3839bf')
    assert file_path

    file_path = get_file_path(
        'ptb.test.txt',
        'ptb',
        url=
        'https://raw.githubusercontent.com/JoonyoungYi/datasets/master/ptb/ptb.test.txt',
        hash='8b80168b89c18661a38ef683c0dc3721')
    assert file_path

    return os.path.dirname(file_path)


def load_data(args):
    dir_path = _init_files()
    return _ptb_raw_data(dir_path)
