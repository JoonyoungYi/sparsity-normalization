import collections
import random

import numpy as np

from .datasets.text8 import load_data


class Dataset:
    def __init__(self, args):
        data = load_data(args)

        self.batch_size = args.batch_size
        self.context_window = args.context_window
        self.context_size = args.context_size
        self.data_index = 0
        self.data = data

    def generate_batch(self):
        batch_size = self.batch_size
        context_size = self.context_size
        context_window = self.context_window
        data_index = self.data_index
        data = self.data

        # all context tokens should be used, hence no associated num_skips argument
        batch = np.ndarray(shape=(batch_size, context_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * context_window + 1
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)

        for i in range(batch_size):
            # context tokens are just all the tokens in buffer except the target
            batch[i, :] = [
                token for idx, token in enumerate(buffer)
                if idx != context_window
            ]
            labels[i, 0] = buffer[context_window]
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)

        self.data_index = data_index
        return batch, labels


# batch, labels = generate_batch(batch_size=8, context_window=1)
# for i in range(8):
#     print(batch[i, 0], reverse_dictionary[batch[i, 0]], batch[i, 1],
#           reverse_dictionary[batch[i, 1]], '->', labels[i, 0],
#           reverse_dictionary[labels[i, 0]])
