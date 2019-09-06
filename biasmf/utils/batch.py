import numpy as np

from common.datasets.movielens import Dataset


class BatchManager:
    def __init__(self, kind):
        dataset = Dataset(kind, Dataset.SPLIT_DEFAULT_10_PERCENT, seed=1)
        self.train_data = dataset.train_data
        self.valid_data = dataset.valid_data
        self.test_data = dataset.test_data

        self.n_user = int(
            max(
                np.max(self.train_data[:, 0]),
                np.max(self.valid_data[:, 0]), np.max(self.test_data[:,
                                                                     0]))) + 1
        self.n_item = int(
            max(
                np.max(self.train_data[:, 1]),
                np.max(self.valid_data[:, 1]), np.max(self.test_data[:,
                                                                     1]))) + 1
        self.mu = np.mean(self.train_data[:, 2])
