import os
import time
import math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from .utils.batch import BatchManager
from .models import init_models


def _train(kind, K, lambda_value):
    batch_manager = BatchManager(kind)
    models = init_models(K, lambda_value, batch_manager)

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    session.run(tf.global_variables_initializer())

    min_valid_rmse = float("Inf")
    min_valid_epoch_idx = -1
    final_test_rmse = float("Inf")
    for epoch_idx in range(100000):
        _, train_rmse = session.run(
            [models['train_op'], models['rmse']],
            feed_dict={
                models['u']: batch_manager.train_data[:, 0],
                models['i']: batch_manager.train_data[:, 1],
                models['r']: batch_manager.train_data[:, 2],
            })

        _, valid_rmse = session.run(
            [models['loss'], models['rmse']],
            feed_dict={
                models['u']: batch_manager.valid_data[:, 0],
                models['i']: batch_manager.valid_data[:, 1],
                models['r']: batch_manager.valid_data[:, 2],
            })

        _, test_rmse = session.run(
            [models['loss'], models['rmse']],
            feed_dict={
                models['u']: batch_manager.test_data[:, 0],
                models['i']: batch_manager.test_data[:, 1],
                models['r']: batch_manager.test_data[:, 2],
            })

        if valid_rmse < min_valid_rmse:
            min_valid_rmse = valid_rmse
            min_valid_epoch_idx = epoch_idx
            final_test_rmse = test_rmse

            p, q, b_u, b_i = session.run(
                [models['p'], models['q'], models['b_u'], models['b_i']])

        print('>> EPOCH:',
              "{:3d}".format(epoch_idx), "{:5f}, {:5f}, {:5f}, {:5f}".format(
                  train_rmse, valid_rmse, test_rmse, final_test_rmse))

        if min_valid_epoch_idx + 100 < epoch_idx:
            break

    np.save('biasmf/weights/{}/p.npy'.format(kind), p)
    np.save('biasmf/weights/{}/q.npy'.format(kind), q)
    np.save('biasmf/weights/{}/b_u.npy'.format(kind), b_u)
    np.save('biasmf/weights/{}/b_i.npy'.format(kind), b_i)
    session.close()


def main():
    _train('movielens-100k', 7, 10)


if __name__ == '__main__':
    main()
