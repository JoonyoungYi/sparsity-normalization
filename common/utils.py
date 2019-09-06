import os
import random
import argparse
import pickle
from datetime import datetime

import numpy as np


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def integer_list(values):
    if values:
        try:
            return [int(v) for v in values.split(',')]
        except:
            raise argparse.ArgumentTypeError(
                'This value should be comma separated integer values.')
    return []


def set_seed(seed, tensorflow=True, torch=False):
    # set seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed=seed)
    if tensorflow:
        import tensorflow as tf
        try:
            tf.random.set_random_seed(seed)
        except:
            print('Error to import tensorflow')

    if torch:
        import torch
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except:
            print('Error to import tensorflow')


def set_torch_seed(seed):
    set_seed(seed, tensorflow=False, torch=True)


def _make_folder_if_not_exist(path):
    if os.path.exists(path):
        assert os.path.isdir(path)
    else:
        os.mkdir(path, 0o777)


def make_folder_if_not_exist(path):
    return _make_folder_if_not_exist(path)


def save_results_to_file(code_name, dataset_name, results):
    line = '\t'.join([str(datetime.now())] + [str(r) for r in results])
    return append_text_results(code_name, dataset_name, 'results', 'default',
                               line)


def save_results_to_pickle(code_name, dataset_name, folder_name, file_name,
                           object):
    # make results folder if not exists.
    _make_folder_if_not_exist('.results')

    # make code_name folder if not exists.
    _make_folder_if_not_exist('.results/{}'.format(code_name))

    # make dataset_name folder if not exists.
    _make_folder_if_not_exist('.results/{}/{}'.format(code_name, dataset_name))

    # make folder_name folder if not exists.
    _make_folder_if_not_exist(
        '.results/{}/{}/{}'.format(code_name, dataset_name, folder_name))

    filepath = '.results/{code_name}/{dataset_name}/{folder_name}/{file_name}.pickle'.format(
        code_name=code_name,
        dataset_name=dataset_name,
        folder_name=folder_name,
        file_name=file_name)
    with open(filepath, 'wb') as f:
        pickle.dump(object, f)


def init_file(code_name, dataset_name, folder_name, file_name):
    filepath = '.results/{code_name}/{dataset_name}/{folder_name}/{file_name}.txt'.format(
        code_name=code_name,
        dataset_name=dataset_name,
        folder_name=folder_name,
        file_name=file_name)
    if os.path.exists(filepath):
        os.remove(filepath)


def append_text_results(code_name, dataset_name, folder_name, file_name, line):
    # make results folder if not exists.
    _make_folder_if_not_exist('.results')

    # make code_name folder if not exists.
    _make_folder_if_not_exist('.results/{}'.format(code_name))

    # make dataset_name folder if not exists.
    _make_folder_if_not_exist('.results/{}/{}'.format(code_name, dataset_name))

    # make folder_name folder if not exists.
    _make_folder_if_not_exist(
        '.results/{}/{}/{}'.format(code_name, dataset_name, folder_name))

    filepath = '.results/{code_name}/{dataset_name}/{folder_name}/{file_name}.txt'.format(
        code_name=code_name,
        dataset_name=dataset_name,
        folder_name=folder_name,
        file_name=file_name)
    os.umask(0)
    with open(os.open(filepath, os.O_CREAT | os.O_WRONLY, 0o777), 'a') as f:
        f.write(line + '\n')
        f.close()
        return True
    return False


def get_activation_fn(activation_fn):
    import tensorflow as tf
    if activation_fn == 'relu':
        return tf.nn.relu
    elif activation_fn == 'leaky_relu':
        return tf.nn.leaky_relu
    elif activation_fn == 'tanh':
        return tf.nn.tanh
    elif activation_fn == 'sigmoid':
        return tf.nn.sigmoid
    elif activation_fn == 'softplus':
        return tf.nn.softplus
    elif activation_fn == 'elu':
        return tf.nn.elu
    else:
        raise NotImplementedError()
