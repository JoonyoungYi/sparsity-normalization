import random
import math

import tensorflow as tf

from .configs import *

ACTIVATION_FUNCTION = tf.nn.sigmoid


def _init_weights(n_col, args):
    weights, biases = [], []

    n_units = [n_col, args.n_hidden_unit, n_col]

    for n_in_unit, n_out_unit in zip(n_units[:-1], n_units[1:]):
        # if i == 1 and hidden_unit_numbers[0] == column_number:
        #     # kaiming init 이다. (first layer는 xavier init과 동일.)
        #     # 첫번째 레이어는 sparsity 고려해 줘야 한다.
        #     limit = math.sqrt(3 / input_unit_number / 0.063)
        # else:
        # kaiming init 이다.
        limit = math.sqrt(6 / n_in_unit)

        weight = tf.Variable(
            tf.random_uniform([n_in_unit, n_out_unit], -limit, limit))
        bias = tf.Variable(tf.zeros([n_out_unit]))
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def _forward(train_x, weights, biases):
    assert len(weights) == len(biases)

    layer = train_x
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.matmul(layer, weight)

        if i == 0:
            layer /= tf.matmul(
                tf.maximum(tf.reduce_sum(train_x, axis=1, keepdims=True), 1),
                tf.ones([1, layer.shape[1]]))

        layer = layer + bias

        if i < len(biases) - 1:
            layer = tf.nn.sigmoid(layer)
            # layer = tf.layers.dropout(
            #     layer, rate=dropout_rate, training=training)

    return layer


def _get_mask(y):
    zero = tf.constant(0.1, dtype=tf.float32)
    where = tf.greater(y, zero)
    m = tf.cast(where, tf.float32)
    return m


def _get_loss(y, y_hat, weights, args):
    m = _get_mask(y)
    rec_loss = tf.reduce_sum(tf.square(m * (y - y_hat)))
    return rec_loss + 0.5 * args.lambda_wd * tf.reduce_sum(
        [tf.reduce_sum(tf.square(w)) for w in weights])


def _get_rmse(y, y_hat, epsilon=0):
    m = _get_mask(y)
    return tf.sqrt(
        (tf.reduce_sum(tf.multiply(tf.square(y - y_hat), m)) + epsilon) /
        (tf.reduce_sum(m) + epsilon))


def init_models(n_col, args):
    train_x = tf.placeholder(tf.float32, [None, n_col], name="train_x")
    test_y = tf.placeholder(tf.float32, [None, n_col], name="test_y")

    # init weights
    weights, biases = _init_weights(n_col, args)
    y_hat = _forward(train_x, weights, biases)
    loss = _get_loss(train_x, y_hat, weights, args)
    train_rmse = _get_rmse(train_x, y_hat)
    test_rmse = _get_rmse(test_y, y_hat)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss, var_list=weights + biases)

    return {
        'train_op': train_op,
        'train_x': train_x,
        'train_rmse': train_rmse,
        'loss': loss,
        'test_y': test_y,
        'test_rmse': test_rmse
    }
