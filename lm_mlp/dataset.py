import collections
import random

import numpy as np

# from .datasets import text8
from .datasets import ptb


def _init_buffer(buffer, num_steps):
    for _ in range(num_steps):
        buffer.append(-1)


def _get_inputs_and_labels(data, args, eos_id, unk_id):
    buffer = collections.deque(maxlen=args.num_steps)
    _init_buffer(buffer, args.num_steps)

    test_inputs = []
    test_labels = []
    data_index = 0
    while True:
        token = data[data_index]
        if token == eos_id:
            _init_buffer(buffer, args.num_steps)
        elif token == unk_id:
            buffer.append(token)
        else:
            test_inputs.append([t if t != unk_id else -1 for t in buffer])
            test_labels.append(token)
            buffer.append(token)

        data_index = data_index + 1
        if data_index >= len(data):
            break

    return np.array(test_inputs), np.array(test_labels)


class Dataset:
    def __init__(self, args):
        if args.dataset == 'text8':
            raise NotImplementedError()
        elif args.dataset == 'ptb':
            train_data, valid_data, test_data, word_to_id = ptb.load_data(args)
            eos_id = word_to_id['<eos>']
            unk_id = word_to_id['<unk>']
        else:
            raise NotImplementedError()

        assert train_data
        assert valid_data
        assert test_data

        self.batch_size = args.batch_size
        self.num_steps = args.num_steps
        self.eos_id = eos_id
        self.unk_id = unk_id

        self.train_data = train_data
        self.valid_inputs, self.valid_labels = _get_inputs_and_labels(
            valid_data, args, eos_id, unk_id)
        self.test_inputs, self.test_labels = _get_inputs_and_labels(
            test_data, args, eos_id, unk_id)

        # initialize for batch
        buffer = collections.deque(maxlen=args.num_steps)
        _init_buffer(buffer, args.num_steps)
        self.buffer = buffer
        self.data_index = 0

    def generate_batch(self):
        batch_size = self.batch_size
        num_steps = self.num_steps
        eos_id = self.eos_id
        unk_id = self.unk_id

        data = self.train_data
        buffer = self.buffer
        data_index = self.data_index

        # all context tokens should be used, hence no associated num_skips argument
        batch = np.ndarray(shape=(batch_size, num_steps), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size), dtype=np.int32)
        i = 0
        while i < batch_size:
            token = data[data_index]
            if token == eos_id:
                data_index = (data_index + 1) % len(data)
                _init_buffer(buffer, num_steps)
            elif token == unk_id:
                buffer.append(token)
            else:
                batch[i, :] = [t if t != unk_id else -1 for t in buffer]
                labels[i] = token
                i += 1
                buffer.append(token)

            data_index = (data_index + 1) % len(data)

        self.data_index = data_index
        self.buffer = buffer
        return batch, labels
