import json
import random
import os
import zipfile

import numpy as np
DATA_PATH = '.datasets/'


def _get_data_from_user_dict(user_dict,
                             user_ids,
                             train=True,
                             train_portion=0.5):
    data = []
    for idx, u in enumerate(user_ids):
        ratings = user_dict.get(u)

        test_start_index = int(len(ratings) * train_portion)
        if train:
            ratings = ratings[:test_start_index]
        else:
            ratings = ratings[test_start_index:]

        for i, r in ratings:
            data.append((idx, i, r))
    return np.array(data)


def _get_data_from_file(original_filepath, user_converter_dict,
                        item_converter_dict, split_token):
    lines = []
    with open(original_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            cols = [float(c) for c in line.split(split_token)]

            user_id = str(int(cols[0]))
            item_id = str(int(cols[1]))

            u = user_converter_dict[user_id]
            i = item_converter_dict[item_id]

            lines.append([u, i, cols[2], cols[3]])

    data = np.array(lines)
    return data


class Dataset:
    KIND_MOVIELENS_100k = 'movielens-100k'
    KIND_MOVIELENS_1M = 'movielens-1m'
    KIND_MOVIELENS_10M = 'movielens-10m'

    # test with random. train:valid:test = 0.855:0.045:0.1
    SPLIT_DEFAULT_10_PERCENT = 'default-10-percent'

    # 5 fold cross validation. train:test = 0.8:0.2
    # movielens 100k를 위한 split. 미리 나눠둔 split이 있어서 그 데이터를 사용한다.
    # dataset seed는 1부터 5까지만 받을 수 있고, 미리 나눠둔 split에 의해 confidence interval도 계산된다.
    SPLIT_5_FOLD = '5-fold'

    # user 기준으로 20%를 test로 자르고, test user에 대해서 50%를 트레이닝 셋으로 줌.
    # train user 중 10%를 validation으로 사용.
    SPLIT_NEW_20_50 = 'new-20-50'

    def __download(self):
        if not os.path.exists(DATA_PATH):
            os.mkdir(DATA_PATH)
            # return False

        if not os.path.isdir(DATA_PATH):
            print("DATA_PATH가 폴더가 아닙니다.")
            return False

        folderpath = os.path.join(DATA_PATH, self.kind)
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)

        filepath = os.path.join(DATA_PATH, self.kind,
                                '{}.zip'.format(self.kind))
        if os.path.exists(filepath):
            return True

        if self.kind == self.KIND_MOVIELENS_100k:
            url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        elif self.kind == self.KIND_MOVIELENS_1M:
            url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        elif self.kind == self.KIND_MOVIELENS_10M:
            url = 'http://files.grouplens.org/datasets/movielens/ml-10m.zip'
        else:
            raise NotImplementedError()

        import requests
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            print("url이 유효하지 않습니다.")
            return False

        with open(filepath, 'wb') as f:
            for i, chunk in enumerate(r.iter_content(1024)):
                f.write(chunk)
                if i % 1024 == 0:
                    print('Download {:4d}MB...'.format(i // 1024))
            f.close()
            return True
        return False

    def __unzip(self):
        if self.kind == self.KIND_MOVIELENS_100k:
            folderpath = os.path.join(DATA_PATH, self.kind, 'ml-100k')
        elif self.kind == self.KIND_MOVIELENS_1M:
            folderpath = os.path.join(DATA_PATH, self.kind, 'ml-1m')
        elif self.kind == self.KIND_MOVIELENS_10M:
            folderpath = os.path.join(DATA_PATH, self.kind, 'ml-10m')
        else:
            raise NotImplementedError()

        if os.path.exists(folderpath):
            return True

        filepath = os.path.join(DATA_PATH, self.kind,
                                '{}.zip'.format(self.kind))
        folderpath = os.path.join(DATA_PATH, self.kind)
        with zipfile.ZipFile(filepath) as f:
            f.extractall(folderpath)
            f.close()
            return True
        return False

    def __init_data_npy(self):
        filepath = os.path.join(DATA_PATH, self.kind, 'data.npy')
        if os.path.exists(filepath):
            return True

        if self.kind == self.KIND_MOVIELENS_100k:
            original_filepath = os.path.join(DATA_PATH, self.kind, 'ml-100k',
                                             'u.data')
            split_token = '\t'
        elif self.kind == self.KIND_MOVIELENS_1M:
            original_filepath = os.path.join(DATA_PATH, self.kind, 'ml-1m',
                                             'ratings.dat')
            split_token = '::'
        elif self.kind == self.KIND_MOVIELENS_10M:
            original_filepath = os.path.join(DATA_PATH, self.kind,
                                             'ml-10M100K', 'ratings.dat')
            split_token = '::'
        else:
            raise NotImplementedError()

        lines = []
        user_dict = {}
        item_dict = {}
        current_u = 0
        current_i = 0

        user_converter_dict = {}
        item_converter_dict = {}
        with open(original_filepath, 'r') as f:
            for line in f:
                line = line.strip()
                cols = [float(c) for c in line.split(split_token)]

                user_id = int(cols[0])
                item_id = int(cols[1])

                u = user_dict.get(user_id, None)
                if u is None:
                    user_dict[user_id] = current_u
                    u = current_u
                    current_u += 1

                i = item_dict.get(item_id, None)
                if i is None:
                    item_dict[item_id] = current_i
                    i = current_i
                    current_i += 1

                user_converter_dict[user_id] = u
                item_converter_dict[item_id] = i

                lines.append([u, i, cols[2], cols[3]])

        data = np.array(lines)
        np.save(filepath, data)

        with open(os.path.join(DATA_PATH, self.kind, 'index-user.json'),
                  'w') as f:
            f.write(json.dumps(user_converter_dict))

        with open(os.path.join(DATA_PATH, self.kind, 'index-item.json'),
                  'w') as f:
            f.write(json.dumps(item_converter_dict))

        print(current_u, current_i)
        # input('should complete get_n_user_and_n_item function.')
        return True

    def __split_npy(self):
        if self.split == self.SPLIT_DEFAULT_10_PERCENT:
            # 10%를 test를 떼고, training 중 5%를 validation으로 뗀다.
            # 이건 CF-NADE 에서 쓴 방법.
            # AutoRec은 validation을 10% 떼는 데... 우리는 그냥 5% 떼는 걸로 통일하자.
            train_filepath = os.path.join(DATA_PATH, self.kind,
                                          'data.{}.{}.train.npy'.format(
                                              self.split, self.seed))
            valid_filepath = os.path.join(DATA_PATH, self.kind,
                                          'data.{}.{}.valid.npy'.format(
                                              self.split, self.seed))
            test_filepath = os.path.join(DATA_PATH, self.kind,
                                         'data.{}.{}.test.npy'.format(
                                             self.split, self.seed))
            if os.path.exists(train_filepath) and os.path.exists(
                    valid_filepath) and os.path.exists(test_filepath):
                return True

            data = np.load(os.path.join(DATA_PATH, self.kind, 'data.npy'))
            np.random.seed(self.seed)
            np.random.shuffle(data)
            # print(data[0])

            n_row = data.shape[0]
            train_data = data[:int(n_row * 0.855)]
            valid_data = data[int(n_row * 0.855):int(n_row * 0.9)]
            test_data = data[int(n_row * 0.9):]
            np.save(train_filepath, train_data)
            np.save(valid_filepath, valid_data)
            np.save(test_filepath, test_data)
            return True

        elif self.split == self.SPLIT_5_FOLD:
            # 10M과 100K만 가능한 방식이나, 100K만 일단 구현해두었습니다.
            assert self.kind in [
                self.KIND_MOVIELENS_100k, self.KIND_MOVIELENS_10M
            ]
            assert self.seed in [1, 2, 3, 4, 5]

            if self.kind == self.KIND_MOVIELENS_100k:
                raw_train_filepath = os.path.join(DATA_PATH, self.kind,
                                                  'ml-100k',
                                                  'u{}.base'.format(self.seed))
                raw_test_filepath = os.path.join(DATA_PATH, self.kind,
                                                 'ml-100k',
                                                 'u{}.test'.format(self.seed))

                train_filepath = os.path.join(DATA_PATH, self.kind,
                                              'data.{}.{}.train.npy'.format(
                                                  self.split, self.seed))
                valid_filepath = os.path.join(DATA_PATH, self.kind,
                                              'data.{}.{}.valid.npy'.format(
                                                  self.split, self.seed))
                test_filepath = os.path.join(DATA_PATH, self.kind,
                                             'data.{}.{}.test.npy'.format(
                                                 self.split, self.seed))

                with open(
                        os.path.join(DATA_PATH, self.kind, 'index-user.json'),
                        'r') as f:
                    user_converter_dict = json.loads(f.read())

                with open(
                        os.path.join(DATA_PATH, self.kind, 'index-item.json'),
                        'r') as f:
                    item_converter_dict = json.loads(f.read())

                train_data = _get_data_from_file(
                    raw_train_filepath,
                    user_converter_dict,
                    item_converter_dict,
                    split_token='\t')
                test_data = _get_data_from_file(
                    raw_test_filepath,
                    user_converter_dict,
                    item_converter_dict,
                    split_token='\t')

                np.random.seed(self.seed)
                np.random.shuffle(train_data)

                n_row = train_data.shape[0]
                valid_data = train_data[int(n_row * 0.8):]
                train_data = train_data[:int(n_row * 0.8)]
                np.save(train_filepath, train_data)
                np.save(valid_filepath, valid_data)
                np.save(test_filepath, test_data)
            else:
                raise NotImplementedError()

        elif self.split == self.SPLIT_NEW_20_50:
            train_filepath = os.path.join(DATA_PATH, self.kind,
                                          'data.{}.{}.train.npy'.format(
                                              self.split, self.seed))
            valid_train_filepath = os.path.join(
                DATA_PATH, self.kind, 'data.{}.{}.valid.train.npy'.format(
                    self.split, self.seed))
            valid_test_filepath = os.path.join(
                DATA_PATH, self.kind, 'data.{}.{}.valid.test.npy'.format(
                    self.split, self.seed))
            test_train_filepath = os.path.join(
                DATA_PATH, self.kind, 'data.{}.{}.test.train.npy'.format(
                    self.split, self.seed))
            test_test_filepath = os.path.join(
                DATA_PATH, self.kind, 'data.{}.{}.test.test.npy'.format(
                    self.split, self.seed))
            if os.path.exists(train_filepath) and os.path.exists(
                    valid_train_filepath) and os.path.exists(
                        valid_test_filepath) and os.path.exists(
                            test_train_filepath) and os.path.exists(
                                test_test_filepath):
                return True

            data = np.load(os.path.join(DATA_PATH, self.kind, 'data.npy'))
            np.random.seed(self.seed)
            np.random.shuffle(data)

            user_dict = {}
            for idx in range(data.shape[0]):
                row = data[idx]
                u = int(row[0])
                i = int(row[1])
                r = float(row[2])

                ratings = user_dict.get(u, [])
                ratings.append((i, r))
                user_dict[u] = ratings

            assert min([len(ratings)
                        for u, ratings in user_dict.items()]) >= 20
            # 그러니까 movielens 100k 데이터에는 한 user가 최소 20개는 rating은 했다.

            assert self.kind == self.KIND_MOVIELENS_100k
            n_user = 943
            user_ids = list(range(n_user))
            random.shuffle(user_ids)

            test_user_ids = user_ids[:int(n_user * 0.2)]
            valid_user_ids = user_ids[int(n_user * 0.2):int(n_user * 0.28)]
            train_user_ids = user_ids[int(n_user * 0.28):]

            train_data = _get_data_from_user_dict(
                user_dict, train_user_ids, train=False, train_portion=0)
            valid_data_train = _get_data_from_user_dict(
                user_dict, valid_user_ids, train=True, train_portion=0.5)
            valid_data_test = _get_data_from_user_dict(
                user_dict, valid_user_ids, train=False, train_portion=0.5)
            test_data_train = _get_data_from_user_dict(
                user_dict, test_user_ids, train=True, train_portion=0.5)
            test_data_test = _get_data_from_user_dict(
                user_dict, test_user_ids, train=False, train_portion=0.5)

            np.save(train_filepath, train_data)
            np.save(valid_train_filepath, valid_data_train)
            np.save(valid_test_filepath, valid_data_test)
            np.save(test_train_filepath, test_data_train)
            np.save(test_test_filepath, test_data_test)
            return True
        else:
            raise NotImplementedError()

    def get_n_user_and_n_item(self):
        if self.kind == self.KIND_MOVIELENS_100k:
            return 943, 1682
        elif self.kind == self.KIND_MOVIELENS_1M:
            return 6040, 3706
        elif self.kind == self.KIND_MOVIELENS_10M:
            return 69878, 10677
        else:
            raise NotImplementedError()

    def _init_npys(self):
        self.__download()
        self.__unzip()
        self.__init_data_npy()
        self.__split_npy()

        if self.split == self.SPLIT_DEFAULT_10_PERCENT or self.split == self.SPLIT_5_FOLD:
            train_filepath = os.path.join(DATA_PATH, self.kind,
                                          'data.{}.{}.train.npy'.format(
                                              self.split, self.seed))
            valid_filepath = os.path.join(DATA_PATH, self.kind,
                                          'data.{}.{}.valid.npy'.format(
                                              self.split, self.seed))
            test_filepath = os.path.join(DATA_PATH, self.kind,
                                         'data.{}.{}.test.npy'.format(
                                             self.split, self.seed))
            self.train_data = np.load(train_filepath)
            self.valid_data = np.load(valid_filepath)
            self.test_data = np.load(test_filepath)

        elif self.split == self.SPLIT_NEW_20_50:
            train_filepath = os.path.join(DATA_PATH, self.kind,
                                          'data.{}.{}.train.npy'.format(
                                              self.split, self.seed))
            valid_train_filepath = os.path.join(
                DATA_PATH, self.kind, 'data.{}.{}.valid.train.npy'.format(
                    self.split, self.seed))
            valid_test_filepath = os.path.join(
                DATA_PATH, self.kind, 'data.{}.{}.valid.test.npy'.format(
                    self.split, self.seed))
            test_train_filepath = os.path.join(
                DATA_PATH, self.kind, 'data.{}.{}.test.train.npy'.format(
                    self.split, self.seed))
            test_test_filepath = os.path.join(
                DATA_PATH, self.kind, 'data.{}.{}.test.test.npy'.format(
                    self.split, self.seed))
            self.train_data = np.load(train_filepath)
            self.valid_data_train = np.load(valid_train_filepath)
            self.valid_data_test = np.load(valid_test_filepath)
            self.test_data_train = np.load(test_train_filepath)
            self.test_data_test = np.load(test_test_filepath)

        else:
            raise NotImplementedError()

    def __init_full_matrix(self, data):
        n_user, n_item = self.get_n_user_and_n_item()
        matrix = np.zeros((n_user, n_item), dtype=np.float32)
        for i in range(data.shape[0]):
            row = data[i]
            # print(row)
            u = int(row[0])
            i = int(row[1])
            r = float(row[2])
            matrix[u][i] = r
        return matrix

    def init_full_matrices(self):
        self.train_matrix = self.__init_full_matrix(self.train_data)
        self.valid_matrix = self.__init_full_matrix(self.valid_data)
        self.test_matrix = self.__init_full_matrix(self.test_data)

    def __init_full_vector(self, data):
        _, n_item = self.get_n_user_and_n_item()
        n_user = int(data[-1][0]) + 1
        matrix = np.zeros([n_user, n_item], dtype=np.float32)
        for idx in range(data.shape[0]):
            row = data[idx]
            u = int(row[0])
            i = int(row[1])
            r = float(row[2])
            matrix[u][i] = r
        return matrix

    def _init_full_vectors(self):
        self.train_x = self.__init_full_vector(self.train_data)
        self.valid_x = self.__init_full_vector(self.valid_data_train)
        self.valid_y = self.__init_full_vector(self.valid_data_test)
        self.test_x = self.__init_full_vector(self.test_data_train)
        self.test_y = self.__init_full_vector(self.test_data_test)

    def __init__(self, kind, split, seed=0):
        self.kind = kind
        self.split = split
        self.seed = seed

        if self.kind == self.KIND_MOVIELENS_100k:
            self.n_class = 5
        elif self.kind == self.KIND_MOVIELENS_1M:
            self.n_class = 5
        elif self.kind == self.KIND_MOVIELENS_10M:
            self.n_class = 10
        else:
            raise NotImplementedError()

        self._init_npys()
        if self.split == self.SPLIT_NEW_20_50:
            self._init_full_vectors()


if __name__ == '__main__':
    dataset = Dataset(
        Dataset.KIND_MOVIELENS_100k, Dataset.SPLIT_DEFAULT_10_PERCENT, seed=0)
