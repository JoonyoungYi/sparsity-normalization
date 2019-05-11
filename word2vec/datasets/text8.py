# http://mattmahoney.net/dc/textdata.html
import os
import collections
import zipfile

from six.moves import urllib
import tensorflow as tf

from common.dataset_utils import get_file_path


def _read_data(filename):
    # Read the data into a list of strings.
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def _build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    dictionary = dict()
    word_idx = 1
    for word, _ in count:
        dictionary[word] = word_idx
        word_idx += 1

    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
            assert index > 0
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    return data


def load_data(args):
    file_path = get_file_path(
        'text8.zip',
        'text8',
        url='http://mattmahoney.net/dc/text8.zip',
        hash='f26f94c5209bc6159618bad4a559ff81')
    words = _read_data(file_path)
    print('>> Data size -> ', len(words))

    # Step 2: Build the dictionary and replace rare words with UNK token.
    data = _build_dataset(words, args.vocabulary_size)
    del words  # Hint to reduce memory.

    return data[:int(len(data) * 0.9)], data[int(len(data) * 0.9):]
