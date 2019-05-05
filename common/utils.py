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


def set_seed(seed):
    # set seed
    random.seed(seed)
    np.random.seed(seed=seed)
    tf.random.set_random_seed(seed)


def integer_list(values):
    if values:
        try:
            return [int(v) for v in values.split(',')]
        except:
            raise argparse.ArgumentTypeError(
                'This value should be comma separated integer values.')

    return []
