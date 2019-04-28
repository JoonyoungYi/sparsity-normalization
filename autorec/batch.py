import numpy as np


class Batch:
    def __init__(self, dataset, args):
        dataset.init_full_matrices()
        n_user, n_item = dataset.get_n_user_and_n_item()

        train_x = dataset.train_matrix
        train_y = train_x.copy()
        valid_y = dataset.valid_matrix
        test_x = dataset.train_matrix + dataset.valid_matrix
        test_y = dataset.test_matrix

        if args.use_user_vector:
            n_row = n_user
            n_col = n_item
        else:
            n_row = n_item
            n_col = n_user
            train_x = train_x.T
            train_y = train_y.T
            valid_y = valid_y.T
            test_x = test_x.T
            test_y = test_y.T

        if args.normalize_first_layer:
            train_c = np.clip(
                np.count_nonzero(train_x, axis=1), a_min=1, a_max=None)
            train_x /= np.matmul(
                np.reshape(train_c, [-1, 1]), np.ones((1, n_col)))

            test_c = np.clip(
                np.count_nonzero(test_x, axis=1), a_min=1, a_max=None)
            test_x /= np.matmul(
                np.reshape(test_c, [-1, 1]), np.ones((1, n_col)))

        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = train_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y = test_y
        self.n_row = n_row
        self.n_col = n_col
