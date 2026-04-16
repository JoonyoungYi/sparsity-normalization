import json
import random
import os
import zipfile

import numpy as np

DATA_PATH = '.datasets/'


def _parse_ratings_file(filepath, user_dict, item_dict, sep):
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            cols = [float(c) for c in line.strip().split(sep)]
            u = user_dict[int(cols[0])]
            i = item_dict[int(cols[1])]
            rows.append([u, i, cols[2], cols[3]])
    return np.array(rows)


class Dataset:
    KIND_MOVIELENS_100k = 'movielens-100k'
    KIND_MOVIELENS_1M = 'movielens-1m'
    KIND_MOVIELENS_10M = 'movielens-10m'

    SPLIT_DEFAULT_10_PERCENT = 'default-10-percent'
    SPLIT_5_FOLD = '5-fold'

    _URLS = {
        KIND_MOVIELENS_100k: 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
        KIND_MOVIELENS_1M:   'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
        KIND_MOVIELENS_10M:  'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
    }
    _N_USERS = {KIND_MOVIELENS_100k: 943, KIND_MOVIELENS_1M: 6040, KIND_MOVIELENS_10M: 69878}
    _N_ITEMS = {KIND_MOVIELENS_100k: 1682, KIND_MOVIELENS_1M: 3706, KIND_MOVIELENS_10M: 10677}
    _N_CLASS = {KIND_MOVIELENS_100k: 5, KIND_MOVIELENS_1M: 5, KIND_MOVIELENS_10M: 10}

    def __init__(self, kind, split, seed=0):
        self.kind = kind
        self.split = split
        self.seed = seed
        self.n_class = self._N_CLASS[kind]
        self._init()

    def get_n_user_and_n_item(self):
        return self._N_USERS[self.kind], self._N_ITEMS[self.kind]

    def init_full_matrices(self):
        n_user, n_item = self.get_n_user_and_n_item()
        self.train_matrix = self._to_matrix(self.train_data, n_user, n_item)
        self.valid_matrix = self._to_matrix(self.valid_data, n_user, n_item)
        self.test_matrix = self._to_matrix(self.test_data, n_user, n_item)

    def _to_matrix(self, data, n_user, n_item):
        m = np.zeros((n_user, n_item), dtype=np.float32)
        for row in data:
            m[int(row[0])][int(row[1])] = float(row[2])
        return m

    def _init(self):
        self._download()
        self._unzip()
        self._build_data_npy()
        self._split()
        self._load_splits()

    def _download(self):
        os.makedirs(os.path.join(DATA_PATH, self.kind), exist_ok=True)
        zip_path = os.path.join(DATA_PATH, self.kind, f'{self.kind}.zip')
        if os.path.exists(zip_path):
            return

        import requests
        r = requests.get(self._URLS[self.kind], stream=True)
        with open(zip_path, 'wb') as f:
            for i, chunk in enumerate(r.iter_content(1024)):
                f.write(chunk)
                if i % 1024 == 0:
                    print(f'Downloading {self.kind}: {i // 1024}MB...')

    def _unzip(self):
        _dirs = {
            self.KIND_MOVIELENS_100k: 'ml-100k',
            self.KIND_MOVIELENS_1M:   'ml-1m',
            self.KIND_MOVIELENS_10M:  'ml-10M100K',
        }
        folder = os.path.join(DATA_PATH, self.kind, _dirs[self.kind])
        if os.path.exists(folder):
            return
        zip_path = os.path.join(DATA_PATH, self.kind, f'{self.kind}.zip')
        with zipfile.ZipFile(zip_path) as f:
            f.extractall(os.path.join(DATA_PATH, self.kind))

    def _build_data_npy(self):
        out = os.path.join(DATA_PATH, self.kind, 'data.npy')
        if os.path.exists(out):
            return

        _files = {
            self.KIND_MOVIELENS_100k: ('ml-100k/u.data', '\t'),
            self.KIND_MOVIELENS_1M:   ('ml-1m/ratings.dat', '::'),
            self.KIND_MOVIELENS_10M:  ('ml-10M100K/ratings.dat', '::'),
        }
        rel_path, sep = _files[self.kind]
        filepath = os.path.join(DATA_PATH, self.kind, rel_path)

        user_map, item_map = {}, {}
        rows = []
        with open(filepath, 'r') as f:
            for line in f:
                cols = [float(c) for c in line.strip().split(sep)]
                uid, iid = int(cols[0]), int(cols[1])
                if uid not in user_map:
                    user_map[uid] = len(user_map)
                if iid not in item_map:
                    item_map[iid] = len(item_map)
                rows.append([user_map[uid], item_map[iid], cols[2], cols[3]])

        np.save(out, np.array(rows))
        with open(os.path.join(DATA_PATH, self.kind, 'index-user.json'), 'w') as f:
            json.dump(user_map, f)
        with open(os.path.join(DATA_PATH, self.kind, 'index-item.json'), 'w') as f:
            json.dump(item_map, f)

    def _split(self):
        base = os.path.join(DATA_PATH, self.kind)

        if self.split == self.SPLIT_DEFAULT_10_PERCENT:
            train_p = os.path.join(base, f'data.{self.split}.{self.seed}.train.npy')
            if os.path.exists(train_p):
                return
            data = np.load(os.path.join(base, 'data.npy'))
            np.random.seed(self.seed)
            np.random.shuffle(data)
            n = data.shape[0]
            train, valid, test = data[:int(n*0.882)], data[int(n*0.882):int(n*0.9)], data[int(n*0.9):]
            for split_name, arr in [('train', train), ('valid', valid), ('test', test)]:
                np.save(os.path.join(base, f'data.{self.split}.{self.seed}.{split_name}.npy'), arr)

        elif self.split == self.SPLIT_5_FOLD:
            assert self.kind == self.KIND_MOVIELENS_100k
            assert 1 <= self.seed <= 5
            train_p = os.path.join(base, f'data.{self.split}.{self.seed}.train.npy')
            if os.path.exists(train_p):
                return

            with open(os.path.join(base, 'index-user.json')) as f:
                user_map = {int(k): v for k, v in json.load(f).items()}
            with open(os.path.join(base, 'index-item.json')) as f:
                item_map = {int(k): v for k, v in json.load(f).items()}

            ml_dir = os.path.join(base, 'ml-100k')
            train_raw = _parse_ratings_file(os.path.join(ml_dir, f'u{self.seed}.base'), user_map, item_map, '\t')
            test_raw  = _parse_ratings_file(os.path.join(ml_dir, f'u{self.seed}.test'), user_map, item_map, '\t')

            np.random.seed(self.seed)
            np.random.shuffle(train_raw)
            n = train_raw.shape[0]
            valid = train_raw[int(n*0.9):]
            train = train_raw[:int(n*0.9)]

            for split_name, arr in [('train', train), ('valid', valid), ('test', test_raw)]:
                np.save(os.path.join(base, f'data.{self.split}.{self.seed}.{split_name}.npy'), arr)

    def _load_splits(self):
        base = os.path.join(DATA_PATH, self.kind)
        for split_name in ['train', 'valid', 'test']:
            path = os.path.join(base, f'data.{self.split}.{self.seed}.{split_name}.npy')
            setattr(self, f'{split_name}_data', np.load(path))


if __name__ == '__main__':
    dataset = Dataset(Dataset.KIND_MOVIELENS_100k, Dataset.SPLIT_DEFAULT_10_PERCENT, seed=0)
    dataset.init_full_matrices()
    print(dataset.train_matrix.shape)
