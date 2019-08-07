import os
import random
import argparse
import pickle
from datetime import datetime

try:
    import tensorflow as tf
except:
    print('Error to import tensorflow')
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


def set_seed(seed):
    # set seed
    random.seed(seed)
    np.random.seed(seed=seed)
    try:
        tf.random.set_random_seed(seed)
    except:
        print('Error to import tensorflow')


def _make_folder_if_not_exist(path):
    if os.path.exists(path):
        assert os.path.isdir(path)
    else:
        os.mkdir(path, 0o777)


def save_results_to_file(code_name, dataset_name, results):
<<<<<<< HEAD
    line = '\t'.join([str(datetime.now())] + [str(r) for r in results]) + '\n'
    return append_text_results(code_name, dataset_name, 'results', 'default',
                               line)


def save_results_to_pickle(code_name, dataset_name, folder_name, file_name,
                           object):
=======
>>>>>>> 8b11627036ac569e926c43e9ae35604f2a49f415
    # make results folder if not exists.
    _make_folder_if_not_exist('.results')

    # make code_name folder if not exists.
    _make_folder_if_not_exist('.results/{}'.format(code_name))

<<<<<<< HEAD
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
    if activation_fn == 'relu':
        return tf.nn.relu
    elif args.activation_fn == 'leaky_relu':
        return tf.nn.leaky_relu
    elif activation_fn == 'tanh':
        return tf.nn.tanh
    elif activation_fn == 'sigmoid':
        return tf.nn.sigmoid
    elif args.activation_fn == 'softplus':
        return tf.nn.softplus
    elif args.activation_fn == 'elu':
        return tf.nn.elu
    else:
        raise NotImplementedError()
=======
    # save results
    os.umask(0)
    filepath = '.results/{code_name}/{dataset_name}.txt'.format(
        code_name=code_name, dataset_name=dataset_name)
    with open(os.open(filepath, os.O_CREAT | os.O_WRONLY, 0o777), 'a') as f:
        line = '\t'.join([str(r) for r in results]) + '\n'
        f.write(line)
        f.close()
        return True
    return False
>>>>>>> 8b11627036ac569e926c43e9ae35604f2a49f415
