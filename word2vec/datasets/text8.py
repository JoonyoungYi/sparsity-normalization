# http://mattmahoney.net/dc/textdata.html
# TODO: read file desccription.
import os
import collections
import zipfile

from six.moves import urllib
import tensorflow as tf


def _maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(
            'http://mattmahoney.net/dc/' + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename +
                        '. Can you get to it with a browser?')
    return filename


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

    # reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data  #, count, dictionary, reverse_dictionary


def load_data(args):
    # Step 1: Download the data.
    filename = _maybe_download('text8.zip', 31344016)
    words = _read_data(filename)
    print('Data size', len(words))

    # Step 2: Build the dictionary and replace rare words with UNK token.
    data = _build_dataset(words, args.vocabulary_size)
    del words  # Hint to reduce memory.
    return data
