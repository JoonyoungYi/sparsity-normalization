import os
import random
import argparse

import tensorflow as tf
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
    tf.random.set_random_seed(seed)


def _make_folder_if_not_exist(path):
    if os.path.exists(path):
        assert os.path.isdir(path)
    else:
        os.mkdir(path, 0o777)


def save_results_to_file(code_name, dataset_name, results):
    # make results folder if not exists.
    _make_folder_if_not_exist('results')

    # make code_name folder if not exists.
    _make_folder_if_not_exist('results/{}'.format(code_name))

    # save results
    os.umask(0)
    filepath = 'results/{code_name}/{dataset_name}.txt'.format(
        code_name=code_name, dataset_name=dataset_name)
    with open(os.open(filepath, os.O_CREAT | os.O_WRONLY, 0o777), 'a') as f:
        line = '\t'.join([str(r) for r in results]) + '\n'
        f.write(line)
        f.close()
        return True
    return False
