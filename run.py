import random

import tensorflow as tf
import numpy as np

from model import init_models


def _prepare_x_y_m():
    f = open('data.csv', 'r')
    x = []
    m = []
    y = []
    for line in f:
        tokens = line.strip().split(',')
        assert len(tokens) == 17
        y.append([1, 0] if tokens[0] == 'republican' else [0, 1])

        tokens = tokens[1:]
        _m = [0 if t == '?' else 1 for t in tokens]
        _x = [1 if t == 'y' else 0 for t in tokens]

        if sum(_m) == 0:
            # print(tokens)
            # input('')
            continue

        prob = 0.2 if random.random() < 0.5 else 0.8
        for j in range(16):
            if random.random() < prob:
                _m[j] = 0
                _x[j] = 0

        if sum(_m) == 0:
            # print(tokens)
            # input('')
            continue

        m.append(_m)
        x.append(_x)
    f.close()
    return x, y, m


def _split_train_test(xs, ys, ms):
    random.seed(1)
    train_x, test_x = [], []
    train_y, test_y = [], []
    train_m, test_m = [], []
    for _x, _y, _m in zip(xs, ys, ms):
        if random.random() < 0.1:
            train_x.append(_x)
            train_y.append(_y)
            train_m.append(_m)
        else:
            test_x.append(_x)
            test_y.append(_y)
            test_m.append(_m)

    return train_x, train_y, train_m, test_x, test_y, test_m


def main():
    x, y, m = _prepare_x_y_m()
    train_x, train_y, train_m, test_x, test_y, test_m = _split_train_test(
        x, y, m)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models()
    session.run(tf.global_variables_initializer())

    max_test_acc = 0.0
    max_test_iter_idx = 0
    for iter_idx in range(1000):
        _, train_loss, train_acc = session.run(
            [models['train_op'], models['loss'], models['accuracy']],
            feed_dict={
                models['x']: train_x,
                models['y']: train_y,
                models['m']: train_m,
            })

        test_loss, test_acc = session.run(
            [models['loss'], models['accuracy']],
            feed_dict={
                models['x']: test_x,
                models['y']: test_y,
                models['m']: test_m,
            })
        print(train_loss, train_acc, test_loss, test_acc, max_test_acc)

        if max_test_acc < test_acc:
            max_test_acc = test_acc
            max_test_iter_idx = iter_idx

        if iter_idx > max_test_iter_idx + 100:
            break


if __name__ == '__main__':
    main()
