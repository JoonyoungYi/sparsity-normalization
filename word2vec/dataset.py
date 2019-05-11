import collections
import random

import numpy as np

from .datasets import text8
from .datasets import ptb


class Dataset:
    def __init__(self, args):
        if args.dataset == 'text8':
            train_data, test_data = text8.load_data(args)
            raise NotImplementedError()  # TODO: handle_valid_date
        elif args.dataset == 'ptb':
            train_data, valid_data, test_data = ptb.load_data(args)
        else:
            raise NotImplementedError()

        self.batch_size = args.batch_size
        self.context_window = args.context_window
        self.context_size = args.context_size
        self.data_index = 0
        self.train_data = train_data

        span = args.context_size + 1
        test_size = len(test_data) // span
        test_inputs = np.ndarray(
            shape=(test_size, args.context_size), dtype=np.int32)
        test_labels = np.ndarray(shape=(test_size, 1), dtype=np.int32)
        for i in range(test_size):
            test_input = test_data[i * span:i * span + args.
                                   context_window] + test_data[i * span + args.
                                                               context_window +
                                                               1:
                                                               (i + 1) * span]
            test_inputs[i, :] = test_input
            test_labels[i, 0] = test_data[i * span + args.context_window]

        self.test_inputs = test_inputs
        self.test_labels = test_labels

    def generate_batch(self):
        batch_size = self.batch_size
        context_size = self.context_size
        context_window = self.context_window
        data_index = self.data_index
        data = self.train_data

        # all context tokens should be used, hence no associated num_skips argument
        batch = np.ndarray(shape=(batch_size, context_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        span = context_size + 1
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)

        for i in range(batch_size):
            # context tokens are just all the tokens in buffer except the target
            batch[i, :] = [
                token for idx, token in enumerate(buffer)
                if idx != context_window  # remove center.
            ]
            labels[i, 0] = buffer[context_window]
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)

        self.data_index = data_index
        return batch, labels
