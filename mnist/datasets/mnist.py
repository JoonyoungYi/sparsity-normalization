import numpy as np

from common.dataset_utils import get_file_path


def load_data():
    path = get_file_path(
        'mnist.npz',
        'mnist',
        url='https://s3.amazonaws.com/img-datasets/mnist.npz',
        hash='8a61469f7ea1b51cbae51d4f78837e45')

    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
