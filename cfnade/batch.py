import random

import numpy as np


class BatchManager:
    def __init__(self, train_x, args):
        print('BatchManager indexing...')
        self.train_x = train_x
        self.n_batch = args.n_batch
        self.normalize_input = args.normalize_input
        self.parameter_sharing = args.parameter_sharing

    def get_batch(self):
        n_batch = self.n_batch
        train_x = self.train_x
        parameter_sharing = self.parameter_sharing
        normalize_input = self.normalize_input

        batch_x = train_x[np.random.randint(0, train_x.shape[0] - 1,
                                            n_batch), :]
        m = np.zeros(batch_x.shape)
        for i in range(n_batch):
            p = random.random()
            m[i][np.random.random(batch_x.shape[1]) < p] = 1

        x = np.multiply(batch_x, m)
        y = np.multiply(batch_x, 1 - m)

        if normalize_input:
            if parameter_sharing:
                x /= np.reshape(np.clip(np.sum(x, axis=1), 1, None), [-1, 1])
            else:
                m = np.zeros(x.shape)
                m[x > 0.1] = 1
                x /= np.reshape(np.clip(np.sum(m, axis=1), 1, None), [-1, 1])
        return x, y
