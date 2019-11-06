import random

import numpy as np
from scipy.sparse import csr_matrix


def _sparse__csr_matrix(x, n_row, n_col):
    return csr_matrix((x[:, 2], (x[:, 0], x[:, 1])), shape=(n_row, n_col))


class Batch:
    def __init__(self, dataset, args):
        self.n_batch = args.n_batch

        self.n_user, self.n_item = dataset.get_n_user_and_n_item()

        train_mat = dataset.train_data
        test_mat = np.concatenate([dataset.train_data, dataset.valid_data])
        self.train_csr_umat = _sparse__csr_matrix(train_mat, self.n_user,
                                                  self.n_item)
        self.test_csr_umat = _sparse__csr_matrix(test_mat, self.n_user,
                                                 self.n_item)
        self.train_csr_imat = self.train_csr_umat.T.copy()
        self.test_csr_imat = self.test_csr_umat.T.copy()

        if args.normalize_input:
            user_c = csr_matrix.sum(self.train_csr_umat > 0, axis=1)
            user_c_mean = np.mean(user_c)
            # print(user_c[:10])
            self.train_ucount = np.maximum(user_c, 1.0) / user_c_mean
            # print(self.train_ucount.shape)
            # print(self.train_ucount[:10])
            # print(np.mean(self.train_ucount))
            # input('!')

            item_c = csr_matrix.sum(self.train_csr_imat > 0, axis=1)
            item_c_mean = np.mean(item_c)
            self.train_icount = np.maximum(item_c, 1.0) / item_c_mean

            self.test_ucount = np.maximum(
                csr_matrix.sum(self.test_csr_umat > 0, axis=1),
                1.0) / user_c_mean
            self.test_icount = np.maximum(
                csr_matrix.sum(self.test_csr_imat > 0, axis=1),
                1.0) / item_c_mean

            # print(np.mean(self.test_ucount))
            # print(np.mean(self.test_ucount))
            # input('!')
        else:
            self.train_ucount = np.ones([self.n_user, 1])
            self.train_icount = np.ones([self.n_item, 1])
            self.test_ucount = np.ones([self.n_user, 1])
            self.test_icount = np.ones([self.n_item, 1])

        train_data = dataset.train_data
        self.train_u = train_data[:, 0].astype(np.int64)
        self.train_i = train_data[:, 1].astype(np.int64)
        self.train_y = train_data[:, 2].astype(np.float32)
        self.n_train = train_data.shape[0]

        valid_data = dataset.valid_data
        self.valid_u = valid_data[:, 0].astype(np.int64)
        self.valid_i = valid_data[:, 1].astype(np.int64)
        self.valid_y = valid_data[:, 2].astype(np.float32)
        self.n_valid = valid_data.shape[0]

        test_data = dataset.test_data
        self.test_u = test_data[:, 0].astype(np.int64)
        self.test_i = test_data[:, 1].astype(np.int64)
        self.test_y = test_data[:, 2].astype(np.float32)
        self.n_test = test_data.shape[0]

    def batch_train(self, iter_idx=None):
        return self._batch(
            self.train_u,
            self.train_i,
            self.train_y,
            self.train_csr_umat,
            self.train_csr_imat,
            self.train_ucount,
            self.train_icount,
            self.n_train,
            iter_idx=iter_idx)

    def batch_valid(self, iter_idx=None):
        return self._batch(
            self.valid_u,
            self.valid_i,
            self.valid_y,
            self.train_csr_umat,
            self.train_csr_imat,
            self.train_ucount,
            self.train_icount,
            self.n_valid,
            iter_idx=iter_idx)

    def batch_test(self, iter_idx=None):
        return self._batch(
            self.test_u,
            self.test_i,
            self.test_y,
            self.test_csr_umat,
            self.test_csr_imat,
            self.test_ucount,
            self.test_icount,
            self.n_test,
            iter_idx=iter_idx)

    def _batch(self,
               u,
               i,
               y,
               csr_umat,
               csr_imat,
               ucount,
               icount,
               n_data,
               iter_idx=None):
        assert type(self.n_batch) == int
        assert iter_idx is not None

        start_idx = iter_idx
        end_idx = min(iter_idx + self.n_batch, n_data)

        _u = u[start_idx:end_idx]
        _i = i[start_idx:end_idx]
        _y = y[start_idx:end_idx]

        _ucount = ucount[_u]
        _icount = icount[_i]
        # print(_ucount.shape)
        # print(_icount.shape)
        # input('!')

        user_x = csr_umat[_u]
        # user_x[range(_i.shape[0]), _i] = 0
        user_x = user_x.tocoo()
        _user_x = (np.mat([user_x.row, user_x.col]).transpose(), user_x.data,
                   (end_idx - start_idx, self.n_item))

        item_x = csr_imat[_i]
        # item_x[range(_u.shape[0]), _u] = 0
        item_x = item_x.tocoo()
        _item_x = (np.mat([item_x.row, item_x.col]).transpose(), item_x.data,
                   (end_idx - start_idx, self.n_user))

        return _user_x, _item_x, _ucount, _icount, _y
