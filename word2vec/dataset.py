import collections
import random

import numpy as np

from .datasets import text8
from .datasets import ptb


def _get_inputs_and_labels(data, args, eos_id):
    # # TODO: handle <eos>
    # span = args.context_size + 1
    # test_size = len(test_data) // span
    # test_inputs = np.ndarray(
    #     shape=(test_size, args.context_size), dtype=np.int32)
    # test_labels = np.ndarray(shape=(test_size, 1), dtype=np.int32)
    # for i in range(test_size):
    #     test_input = test_data[i * span:i * span + args.
    #                            context_window] + test_data[i * span + args.
    #                                                        context_window + 1:
    #                                                        (i + 1) * span]
    #     test_inputs[i, :] = test_input
    #     test_labels[i, 0] = test_data[i * span + args.context_window]

    #
    span = args.context_size + 1
    context_window = args.context_window
    eos_handling = False
    buffer = collections.deque(maxlen=span)
    _init_buffer_with_minus_1(buffer, span)
    data_index = _append_initial_data_to_buffer(buffer, data, 0, eos_id,
                                                args.context_window)

    test_inputs = []
    test_labels = []
    while True:
        test_inputs.append([
            token for idx, token in enumerate(buffer)
            if idx != context_window  # remove center.
        ])
        test_labels.append(buffer[context_window])

        if eos_handling:
            token = -1
            if buffer[-context_window] == -1:
                eos_handling = False
                _init_buffer_with_minus_1(buffer, span)
                new_data_index = _append_initial_data_to_buffer(
                    buffer, data, data_index, eos_id, context_window) - 1
                if new_data_index < data_index:
                    break

                data_index = new_data_index
            else:
                buffer.append(token)
        else:
            token = data[data_index]
            buffer.append(token)

        if not eos_handling:
            data_index = data_index + 1
            if data_index >= len(data):
                break
        if token == eos_id:
            eos_handling = True

    return np.array(test_inputs), np.array(test_labels).reshape([-1, 1])


def _init_buffer_with_minus_1(buffer, span):
    for _ in range(span):
        buffer.append(-1)
    return buffer


def _append_initial_data_to_buffer(buffer, data, data_index, eos_id,
                                   context_window):
    meet_eos = False
    while buffer[-context_window - 1] == -1:
        token = data[data_index]
        buffer.append(token)
        if token == eos_id:
            meet_eos = True
            break
        else:
            data_index = (data_index + 1) % len(data)

    if meet_eos:
        while buffer[-context_window - 1] == -1:
            buffer.append(-1)
    return data_index


class Dataset:
    def __init__(self, args):
        if args.dataset == 'text8':
            train_data, valid_data, test_data, eos_id = text8.load_data(args)
            raise NotImplementedError()  # TODO: handle_valid_date
        elif args.dataset == 'ptb':
            train_data, valid_data, test_data, eos_id = ptb.load_data(args)
        else:
            raise NotImplementedError()

        assert train_data
        assert valid_data
        assert test_data

        self.batch_size = args.batch_size
        self.context_window = args.context_window
        self.context_size = args.context_size
        self.eos_id = eos_id

        self.train_data = train_data
        self.valid_inputs, self.valid_labels = _get_inputs_and_labels(
            valid_data, args, eos_id)
        self.test_inputs, self.test_labels = _get_inputs_and_labels(
            test_data, args, eos_id)

        # initialize for batch
        span = args.context_size + 1
        buffer = collections.deque(maxlen=span)
        _init_buffer_with_minus_1(buffer, span)
        data_index = _append_initial_data_to_buffer(
            buffer, train_data, 0, eos_id, args.context_window)
        self.span = span
        self.buffer = buffer
        self.data_index = data_index
        self.eos_handling = False

    def generate_batch(self):
        # print('\n>> start generate_batch')
        batch_size = self.batch_size
        context_size = self.context_size
        context_window = self.context_window
        data_index = self.data_index
        data = self.train_data
        eos_id = self.eos_id
        span = self.span
        buffer = self.buffer
        eos_handling = self.eos_handling

        # all context tokens should be used, hence no associated num_skips argument
        batch = np.ndarray(shape=(batch_size, context_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        for i in range(batch_size):
            # context tokens are just all the tokens in buffer except the target
            batch[i, :] = [
                token for idx, token in enumerate(buffer)
                if idx != context_window  # remove center.
            ]
            labels[i, 0] = buffer[context_window]
            # print(i, data_index, buffer)

            if eos_handling:
                token = -1
                if buffer[-context_window] == -1:
                    eos_handling = False
                    _init_buffer_with_minus_1(buffer, span)
                    data_index = _append_initial_data_to_buffer(
                        buffer, data, data_index, eos_id, context_window) - 1
                else:
                    buffer.append(token)
            else:
                token = data[data_index]
                buffer.append(token)

            if not eos_handling:
                data_index = (data_index + 1) % len(data)
            if token == eos_id:
                eos_handling = True

        self.data_index = data_index
        self.buffer = buffer
        self.eos_handling = eos_handling
        # print('>> finish generate_batch\n')
        return batch, labels
