import random
import time

import numpy as np


class BatchManager:
    def __init__(self, train_x, args):
        print('BatchManager indexing...')
        self.train_x = train_x
        self.n_batch = args.n_batch

        self.averaging_possible_choices = args.averaging_possible_choices

    def get_batch(self):
        n_batch = self.n_batch
        train_x = self.train_x
        averaging_possible_choices = self.averaging_possible_choices

        batch_x = train_x[np.random.randint(0, train_x.shape[0], n_batch), :]
        if averaging_possible_choices:
            m = np.zeros(batch_x.shape)
            for i in range(n_batch):
                p = random.random()
                m[i][np.random.random(batch_x.shape[1]) < p] = 1

            x = np.multiply(batch_x, m)
            y = np.multiply(batch_x, 1 - m)

            # print(np.mean(np.sum(x > 0.25, axis=1)))
            # input('!')
            return x, y
            # return x, batch_x
        else:
            return batch_x, batch_x
