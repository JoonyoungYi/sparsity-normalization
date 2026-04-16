import random

import numpy as np
from scipy.sparse import csr_matrix


def _swap_user_item(x):
    out = x.copy()
    out[:, 0] = x[:, 1]
    out[:, 1] = x[:, 0]
    return out


def _row_counts(x, n_row):
    c = np.zeros(n_row)
    for row in x:
        c[int(row[0])] += 1
    return c


def _sn_normalize(x, counts, k_prime):
    out = x.copy()
    for i in range(out.shape[0]):
        out[i][2] = out[i][2] * k_prime / max(counts[int(out[i][0])], 1)
    return out


def _to_sparse_tensor(x, n_row, n_col):
    return (x[:, :2].astype(np.int64), x[:, 2], (n_row, n_col))


def _to_csr(x, n_row, n_col):
    return csr_matrix((x[:, 2], (x[:, 0], x[:, 1])), shape=(n_row, n_col))


class Batch:
    def __init__(self, dataset, args):
        self.n_batch = args.n_batch

        if args.sparse_tensor:
            self._init_sparse(dataset, args)
        else:
            self._init_dense(dataset, args)

    def _init_sparse(self, dataset, args):
        n_user, n_item = dataset.get_n_user_and_n_item()

        train_x = dataset.train_data
        train_y = train_x.copy()
        valid_y = dataset.valid_data
        test_x = np.concatenate([dataset.train_data, dataset.valid_data])
        test_y = dataset.test_data

        if args.use_user_vector:
            n_row, n_col = n_user, n_item
        else:
            n_row, n_col = n_item, n_user
            train_x = _swap_user_item(train_x)
            train_y = _swap_user_item(train_y)
            valid_y = _swap_user_item(valid_y)
            test_x = _swap_user_item(test_x)
            test_y = _swap_user_item(test_y)

        row_indices = list(range(n_row))
        random.shuffle(row_indices)
        self.row_indices = row_indices

        if args.normalize_input:
            k_prime = args.K_denominator or n_col
            train_counts = _row_counts(train_x, n_row)
            test_counts = _row_counts(test_x, n_row)
            train_x = _sn_normalize(train_x, train_counts, k_prime)
            test_x = _sn_normalize(test_x, test_counts, k_prime)

        self.train_x = _to_sparse_tensor(train_x, n_row, n_col)
        self.train_y = _to_sparse_tensor(train_y, n_row, n_col)
        self.valid_x = _to_sparse_tensor(train_x, n_row, n_col)
        self.valid_y = _to_sparse_tensor(valid_y, n_row, n_col)
        self.test_x = _to_sparse_tensor(test_x, n_row, n_col)
        self.test_y = _to_sparse_tensor(test_y, n_row, n_col)

        self.train_csr_x = _to_csr(train_x, n_row, n_col)
        self.train_csr_y = _to_csr(train_y, n_row, n_col)
        self.valid_csr_x = _to_csr(train_x, n_row, n_col)
        self.valid_csr_y = _to_csr(valid_y, n_row, n_col)
        self.test_csr_x = _to_csr(test_x, n_row, n_col)
        self.test_csr_y = _to_csr(test_y, n_row, n_col)

        self.n_row = n_row
        self.n_col = n_col

    def _init_dense(self, dataset, args):
        dataset.init_full_matrices()
        n_user, n_item = dataset.get_n_user_and_n_item()

        train_x = dataset.train_matrix
        train_y = train_x.copy()
        valid_y = dataset.valid_matrix
        test_x = dataset.train_matrix + dataset.valid_matrix
        test_y = dataset.test_matrix

        if args.use_user_vector:
            n_row, n_col = n_user, n_item
        else:
            n_row, n_col = n_item, n_user
            train_x = train_x.T
            train_y = train_y.T
            valid_y = valid_y.T
            test_x = test_x.T
            test_y = test_y.T

        if args.normalize_input:
            assert args.imputation == 'zero'
            k_prime = args.K_denominator or n_col
            train_c = np.maximum(np.count_nonzero(train_x, axis=1, keepdims=True), 1).astype(np.float32)
            test_c = np.maximum(np.count_nonzero(test_x, axis=1, keepdims=True), 1).astype(np.float32)
            train_x = train_x * k_prime / train_c
            test_x = test_x * k_prime / test_c

        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = train_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y = test_y
        self.train_csr_x = self.train_csr_y = None
        self.valid_csr_x = self.valid_csr_y = None
        self.test_csr_x = self.test_csr_y = None
        self.n_row = n_row
        self.n_col = n_col

    def batch_train(self, iter_idx=None):
        return self._batch(self.train_x, self.train_y, self.train_csr_x, self.train_csr_y, iter_idx)

    def batch_valid(self, iter_idx=None):
        return self._batch(self.valid_x, self.valid_y, self.valid_csr_x, self.valid_csr_y, iter_idx)

    def batch_test(self, iter_idx=None):
        return self._batch(self.test_x, self.test_y, self.test_csr_x, self.test_csr_y, iter_idx)

    def _batch(self, x, y, csr_x, csr_y, iter_idx=None):
        if iter_idx is None:
            return x, y

        start = iter_idx * self.n_batch
        end = min((iter_idx + 1) * self.n_batch, self.n_row)

        if csr_x is not None:
            bx = csr_x[self.row_indices[start:end]].tocoo()
            by = csr_y[self.row_indices[start:end]].tocoo()
            size = (end - start, self.n_col)
            return (
                (np.column_stack([bx.row, bx.col]), bx.data, size),
                (np.column_stack([by.row, by.col]), by.data, size),
            )
        else:
            return x[start:end], y[start:end]
