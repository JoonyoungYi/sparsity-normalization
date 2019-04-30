import random

import tensorflow as tf
from tensorflow import keras
import numpy as np

from .model import init_models


def main():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    train_x = train_x.reshape([-1, 28 * 28])
    test_x = test_x.reshape([-1, 28 * 28])

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
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
            })

        test_loss, test_acc = session.run(
            [models['loss'], models['accuracy']],
            feed_dict={
                models['x']: test_x,
                models['y']: test_y,
            })
        print(train_loss, train_acc, test_loss, test_acc, max_test_acc)

        if max_test_acc < test_acc:
            max_test_acc = test_acc
            max_test_iter_idx = iter_idx

        if iter_idx > max_test_iter_idx + 100:
            break


if __name__ == '__main__':
    main()
