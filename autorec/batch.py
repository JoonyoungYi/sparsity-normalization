import numpy as np
from scipy.sparse import csr_matrix


def _sparse__swith_row_col(_x):
    x = _x.copy()
    x[:, 0] = _x[:, 1]
    x[:, 1] = _x[:, 0]
    return x


def _sparse__get_count(x, n_row):
    c = np.zeros(n_row)
    for i in range(x.shape[0]):
        row = x[i]
        c[int(row[0])] += 1
    return c


def _sparse__normalize(x, c):
    for i in range(x.shape[0]):
        x[i][2] /= c[int(x[i][0])]
    return x


def _sparse__tensor(x, n_row, n_col):
    return (x[:, :2], x[:, 2], (n_row, n_col))


def _sparse__csr_matrix(x, n_row, n_col):
    return csr_matrix((x[:, 2], (x[:, 0], x[:, 1])), shape=(n_row, n_col))


class Batch:
    def __init__(self, dataset, args):
        self.n_batch = args.n_batch

        if args.sparse_tensor:
            n_user, n_item = dataset.get_n_user_and_n_item()

            train_x = dataset.train_data
            train_y = train_x.copy()
            valid_y = dataset.valid_data
            test_x = np.concatenate([dataset.train_data, dataset.valid_data])
            test_y = dataset.test_data

            if args.use_user_vector:
                n_row = n_user
                n_col = n_item
            else:
                n_row = n_item
                n_col = n_user

                train_x = _sparse__swith_row_col(train_x)
                train_y = _sparse__swith_row_col(train_y)
                valid_y = _sparse__swith_row_col(valid_y)
                test_x = _sparse__swith_row_col(test_x)
                test_y = _sparse__swith_row_col(test_y)

            if args.normalize_first_layer:
                train_c = np.clip(
                    _sparse__get_count(train_x, n_row), a_min=1, a_max=None)
                train_x = _sparse__normalize(train_x, train_c)

                test_c = np.clip(
                    _sparse__get_count(test_x, n_row), a_min=1, a_max=None)
                test_x = _sparse__normalize(test_x, test_c)

            # self.train_csr_x = _sparse__csr_matrix(train_x, n_row, n_col)
            self.train_csr_x = _sparse__csr_matrix(train_x, n_row, n_col)
            self.train_csr_y = _sparse__csr_matrix(train_y, n_row, n_col)
            self.valid_csr_x = _sparse__csr_matrix(train_x, n_row, n_col)
            self.valid_csr_y = _sparse__csr_matrix(valid_y, n_row, n_col)
            self.test_csr_x = _sparse__csr_matrix(test_x, n_row, n_col)
            self.test_csr_y = _sparse__csr_matrix(test_y, n_row, n_col)

            self.train_x = _sparse__tensor(train_x, n_row, n_col)
            self.train_y = _sparse__tensor(train_y, n_row, n_col)
            self.valid_x = _sparse__tensor(train_x, n_row, n_col)
            self.valid_y = _sparse__tensor(valid_y, n_row, n_col)
            self.test_x = _sparse__tensor(test_x, n_row, n_col)
            self.test_y = _sparse__tensor(test_y, n_row, n_col)

            self.n_row = n_row
            self.n_col = n_col
        else:
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

    def get_batch(self, iter_idx=None):
        if iter_idx is None:
            batch_x = self.train_x
            batch_y = self.train_y
            return batch_x, batch_y
        else:
            assert type(self.n_batch) == int
            start_idx = iter_idx * self.n_batch
            end_idx = min((iter_idx + 1) * self.n_batch, self.n_row)

            batch_x = self.train_csr_x[start_idx:end_idx].tocoo()
            batch_y = self.train_csr_y[start_idx:end_idx].tocoo()

            _x = (np.mat([batch_x.row, batch_x.col]).transpose(), batch_x.data,
                  (end_idx - start_idx, self.n_col))
            _y = (np.mat([batch_y.row, batch_y.col]).transpose(), batch_y.data,
                  (end_idx - start_idx, self.n_col))

            return _x, _y
