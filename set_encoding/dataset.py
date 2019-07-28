import numpy as np

from common.utils import set_seed


def _generate_data(batch_size, max_length=10, test=False, pooling='max'):
    # length = np.random.randint(1, max_length + 1)
    x = np.random.randint(1, 100, (batch_size, max_length))
    if pooling == 'max':
        y = np.max(x, axis=1)
    elif pooling == 'mean':
        y = np.mean(x, axis=1)
    elif pooling == 'sum':
        y = np.sum(x, axis=1)
    elif pooling == 'min':
        y = np.min(x, axis=1)
    else:
        raise NotImplementedError()
    x, y = np.expand_dims(x, axis=2), np.expand_dims(y, axis=1)
    return x, y


class Dataset:
    def __init__(self, args):
        set_seed(args.seed)  # for clarity

        self.train_data = _generate_data(
            batch_size=100000, max_length=10, pooling=args.data_pooling)

    def get_train_batch(self, idx=None):
        return self.train_data
