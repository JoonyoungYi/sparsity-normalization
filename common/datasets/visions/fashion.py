import gzip

import numpy as np

from ..utils import get_file_path


def load_data():
    paths = []
    for file_name, hash in [
        ('train-labels-idx1-ubyte.gz', '25c81989df183df01b3e8a0aad5dffbe'),
        ('train-images-idx3-ubyte.gz', '8d4fb7e6c68d591d4c3dfef9ec88bf0d'),
        ('t10k-labels-idx1-ubyte.gz', 'bb300cfdad3c16e7a12a480ee83cd310'),
        ('t10k-images-idx3-ubyte.gz', 'bef4ecab320f06d8554ea6380940ec79'),
    ]:
        path = get_file_path(
            'fashion-' + file_name,
            'fashion-mnist',
            url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/' +
            file_name,
            hash=hash)
        assert path
        paths.append(path)

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)
