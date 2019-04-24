import random

import numpy as np


class BatchManager:
    def __init__(self, train_x):

        # x_filepath = 'cfnade/movielens-1m-x.npy'
        # y_filepath = 'cfnade/movielens-1m-y.npy'

        # try:
        #     x = np.load(x_filepath)
        #     y = np.load(x_filepath)
        # except:
        print('BatchManager indexing...')
        # indices = np.argwhere(train_x > 0)
        # d = {}
        # for i in range(indices.shape[0]):
        #     index = indices[i]
        #     ratings = d.get(index[0], [])
        #     ratings.append((index[1], train_x[index[0]][index[1]]))
        #     d[index[0]] = ratings
        #
        # self.n_row = train_x.shape[0]
        # self.n_col = train_x.shape[1]
        # self.d = d

        # print('BatchManager finding size...')
        # n_row = sum([len(ratings) - 1 for i, ratings in d.items()])

        # print('BatchManager pre-making x and y...', n_row)
        # x = np.zeros((n_row, train_x.shape[1]))
        # y = np.zeros((n_row, train_x.shape[1]))
        # curr = 0
        # for i, ratings in d.items():
        #     random.shuffle(ratings)
        #     _x = np.zeros((train_x.shape[1]))
        #     j, r = ratings[0]
        #     _x[j] = r
        #
        #     for j, r in ratings[1:]:
        #         x[curr] = _x
        #         y[curr][j] = r
        #         curr += 1
        #
        #         _x[j] = r
        # np.save(x_filepath, x)
        # np.save(y_filepath, y)
        #
        # self.x = x
        # self.y = y
        # print('Load Complete!')

        self.train_x = train_x

    def get_batch(self):
        # d = self.d
        # n_row = self.n_row
        # n_col = self.n_col
        #
        # for _ in range(10):
        #     i = random.randint(0, n_row - 1)
        #     ratings = d.get(i, [])
        #     if ratings:
        #         break
        #
        # random.shuffle(ratings)
        #
        # x = np.zeros([len(ratings) - 1, n_col])
        # y = np.zeros([len(ratings) - 1, n_col])
        #
        # _x = np.zeros((n_col))
        # j, r = ratings[0]
        # _x[j] = r
        #
        # curr = 0
        # for j, r in ratings[1:]:
        #     x[curr] = _x
        #     y[curr][j] = r
        #     curr += 1
        #
        #     _x[j] = r
        #
        # return x, y

        train_x = self.train_x

        n_sample = random.randint(1, train_x.shape[1] - 1)
        m_indices = np.random.choice(train_x.shape[1], n_sample)
        # print(m_indices)
        m = np.zeros(train_x.shape)
        # print(train_x.shape)
        m[:, m_indices] = 1

        # m = np.random.randint(2, size=train_x.shape)
        batch_x = np.multiply(train_x, m)
        batch_y = np.multiply(train_x, 1 - m)

        # batch_x = train_x
        # batch_y = train_x
        return batch_x, batch_y
        # return self.x, self.y
