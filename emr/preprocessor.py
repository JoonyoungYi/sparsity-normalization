import re
import glob

import numpy as np
from tqdm import tqdm
from pprint import pprint

from common.utils import set_torch_seed

FEATURES = [
    'DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH',
    'Albumin', 'ALT', 'Glucose', 'SaO2', 'Temp', 'AST', 'Bilirubin', 'HCO3',
    'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
    'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2',
    'Platelets', 'Urine', 'NIMAP', 'Creatinine', 'ALP', 'Weight'
]
# {'ICUType', 'Age', 'Height', 'Gender'}

# for NORMALIZATION in ['minmax', 'standardization']:
NORMALIZATION = 'standardization'
TIME_UNIT = 60

# for TIME_UNIT in [10, 12, 15, 18, 20, 30, 40, 60, 80, 90]:
# for TIME_UNIT in [80, 90]:


def _parse_filename(filename):
    assert 48 * 60 % TIME_UNIT == 0
    n_time = (48 * 60) // TIME_UNIT

    x = np.empty([n_time, len(FEATURES)])
    x[:] = np.nan

    abandonded_features = []
    icu_type = None
    with open(filename, 'r') as f:
        f.readline()

        for line in f:
            cols = line.strip().split(',')
            assert len(cols) == 3

            times = [int(s) for s in cols[0].split(':')]
            time = times[0] * 60 + times[1]
            time_idx = min(time // TIME_UNIT, n_time - 1)

            cols[1] = cols[1].strip()
            if cols[1] not in FEATURES:
                if cols[1] and cols[1] not in ['MechVent']:
                    abandonded_features.append(cols[1])
                    if 'ICUType' == cols[1]:
                        icu_type = cols[2]
                continue

            feature_idx = FEATURES.index(cols[1])
            value = float(cols[2])

            x[time_idx][feature_idx] = value
            # print(time, feature_index, value)

    if len(set(abandonded_features)) != len(abandonded_features):
        print(abandonded_features)
        assert len(set(abandonded_features)) == len(abandonded_features)

    assert icu_type
    return x, icu_type


def _load_y_dict(set_label):
    y_dict = {}
    with open(
            'emr/datasets/physionet2012/Outcomes-{set_label}.txt'.format(
                set_label=set_label),
            'r') as f:
        line = f.readline()
        names = line.strip().split(',')
        print(names)

        for line in f:
            line = line.strip()
            cols = [int(c) for c in line.split(',')]

            user_id = cols[0]
            y_dict[user_id] = {names[j]: cols[j] for j in range(1, 6)}
    return y_dict


def _load_x_and_y_of_set(set_label):
    y_dict = _load_y_dict(set_label)
    xs, ys = [], []
    filenames = glob.glob('emr/datasets/physionet2012/set-{set_label}/*'.
                          format(set_label=set_label))
    assert len(filenames) == 4000

    for filename in tqdm(filenames, total=4000):
        m = re.search(
            'emr/datasets/physionet2012/set-{set_label}'.format(
                set_label=set_label) + '/(?P<user_id>\d{6})',
            filename)
        assert m
        user_id = int(m.group('user_id'))

        x, icu_type = _parse_filename(filename)
        user = y_dict[user_id]
        y = [
            user['In-hospital_death'],  # mortality
            1 if user['Length_of_stay'] < 3 else 0,  # Length of stay < 3
            1 if icu_type == '2' else 0,  # Cardiac condition
            1 if icu_type == '4' else 0,  # Recovery from surgery
            user['Length_of_stay'],  # regression
        ]
        xs.append(x)
        ys.append(y)
    x = np.stack(xs)
    y = np.stack(ys)
    return x, y


def _load_x_and_y():
    xs, ys = [], []
    for set_label in ['a', 'b', 'c']:
        # for set_label in ['a']:
        x, y = _load_x_and_y_of_set(set_label)
        xs.append(x)
        ys.append(y)
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    perm = np.random.permutation(x.shape[0])
    x = x[perm]
    y = y[perm]

    return x, y


# perm = np.random.permutation(test_x.shape[0])
# test_x = test_x[perm]
# test_y = test_y[perm]
#
# valid_x = test_x[:2000]
# valid_y = test_y[:2000]
# test_x = test_x[2000:]
# test_y = test_y[2000:]
#
# print(test_x.shape)
# print(valid_x.shape)
# train_x = x[:8000]
# train_y = y[:8000]
# y = y[:, 0]
# print(index_1.shape)
# print(index_0.shape)
# input('!')


def main():
    dataset_seed = 1
    set_torch_seed(dataset_seed)
    x, y = _load_x_and_y()

    # train_x = x[:8000]
    # train_y = y[:8000]
    # valid_x = x[8000:10000]
    # valid_y = y[8000:10000]
    # test_x = x[10000:]
    # test_y = y[10000:]

    y = y[:, 0]
    index_1 = np.where(y == 1)[0]
    index_0 = np.where(y == 0)[0]

    # print(index_1.shape)
    # print(index_0.shape)

    n_test_per_label = 250

    _index_1 = index_1[:n_test_per_label]
    _index_0 = index_0[:n_test_per_label]
    test_x = np.concatenate([x[_index_1], x[_index_0]], axis=0)
    test_y = np.concatenate([y[_index_1], y[_index_0]], axis=0)

    _index_1 = index_1[n_test_per_label:2 * n_test_per_label]
    _index_0 = index_0[n_test_per_label:2 * n_test_per_label]
    valid_x = np.concatenate([x[_index_1], x[_index_0]], axis=0)
    valid_y = np.concatenate([y[_index_1], y[_index_0]], axis=0)

    _index_1 = index_1[2 * n_test_per_label:]
    _index_0 = index_0[2 * n_test_per_label:]
    # print(_index_0.shape)
    # print(_index_1.shape)
    train_x = np.concatenate([x[_index_1], x[_index_0]], axis=0)
    train_y = np.concatenate([y[_index_1], y[_index_0]], axis=0)

    perm = np.random.permutation(train_x.shape[0])
    train_x = train_x[perm]
    train_y = train_y[perm]

    perm = np.random.permutation(valid_x.shape[0])
    valid_x = valid_x[perm]
    valid_y = valid_y[perm]

    perm = np.random.permutation(test_x.shape[0])
    test_x = test_x[perm]
    test_y = test_y[perm]

    print(train_x.shape)
    print(train_y.shape)
    print(valid_x.shape)
    print(valid_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    input('!')

    for feature_idx in range(len(FEATURES)):
        x_feature = train_x[:, :, feature_idx]
        x_feature = x_feature[~np.isnan(x_feature)]

        if NORMALIZATION == 'minmax':
            x_feature_min = np.min(x_feature)
            x_feature_max = np.max(x_feature)
            print(feature_idx, FEATURES[feature_idx],
                  x_feature_max - x_feature_min)
            train_x[:, :, feature_idx] = (
                train_x[:, :, feature_idx] - x_feature_min) / (
                    x_feature_max - x_feature_min)
            valid_x[:, :, feature_idx] = (
                valid_x[:, :, feature_idx] - x_feature_min) / (
                    x_feature_max - x_feature_min)
            test_x[:, :, feature_idx] = (
                test_x[:, :, feature_idx] - x_feature_min) / (
                    x_feature_max - x_feature_min)

        elif NORMALIZATION == 'standardization':
            x_feature_mean = np.mean(x_feature)
            x_feature_std = np.std(x_feature)
            train_x[:, :, feature_idx] = (
                train_x[:, :, feature_idx] - x_feature_mean) / (x_feature_std)
            valid_x[:, :, feature_idx] = (
                valid_x[:, :, feature_idx] - x_feature_mean) / (x_feature_std)
            test_x[:, :, feature_idx] = (
                test_x[:, :, feature_idx] - x_feature_mean) / (x_feature_std)

        else:
            raise NotImplementedError()

    n_timestep = 2880 // TIME_UNIT
    base_path = 'emr/datasets/physionet2012/processed/balanaced-t{n_timestep}-{normalization}-dseed{dataset_seed}-'.format(
        normalization=NORMALIZATION,
        n_timestep=n_timestep,
        dataset_seed=dataset_seed)
    np.save(base_path + 'train_x.npy', train_x)
    np.save(base_path + 'train_y.npy', train_y)
    np.save(base_path + 'valid_x.npy', valid_x)
    np.save(base_path + 'valid_y.npy', valid_y)
    np.save(base_path + 'test_x.npy', test_x)
    np.save(base_path + 'test_y.npy', test_y)

    print(
        np.min(valid_x[~np.isnan(valid_x)]),
        np.max(valid_x[~np.isnan(valid_x)]))
    print(np.min(test_x[~np.isnan(test_x)]), np.max(test_x[~np.isnan(test_x)]))

    # print(list(m_time))

    # print(np.sum(np.isnan(x), axis=1))
    # print(x.shape)


main()

# python3 -m emr.preprocesser
