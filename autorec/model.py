import random
import math

import tensorflow as tf

from .configs import *
from .rprop import RPropOptimizer

ACTIVATION_FUNCTION = tf.nn.sigmoid

# ACTIVATION_FUNCTION = tf.nn.tanh
# ACTIVATION_FUNCTION = tf.nn.relu


def _init_weights(n_col, args):
    weights, biases = [], []

    assert args.n_hidden_layer == 1
    n_units = [n_col] + [args.n_hidden_unit] * args.n_hidden_layer + [n_col]

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


def _forward(train_x, weights, biases, training, args):
    assert len(weights) == len(biases)

    layer = train_x
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.matmul(layer, weight)
        layer = layer + bias

        if i < len(biases) - 1:
            layer = ACTIVATION_FUNCTION(layer)
            # layer = tf.layers.dropout(layer, rate=0.5, training=training)
    return layer


def _get_mask(y):
    zero = tf.constant(0.1, dtype=tf.float32)
    where = tf.greater(y, zero)
    m = tf.cast(where, tf.float32)
    return m


def _get_loss(y, y_hat, weights, args):
    m = _get_mask(y)
    m_user = _get_mask(tf.reduce_sum(m, axis=1))
    error = tf.square(m * (y - y_hat))
    user_error = tf.reduce_sum(error, axis=1)
    rec_loss = tf.reduce_sum(user_error * m_user)

    reg_loss = tf.reduce_sum([
        tf.reduce_sum(tf.square(w)) * l
        for l, w in zip([args.lambda_w1, args.lambda_w2], weights)
    ])
    return rec_loss + 0.5 * reg_loss


def _get_rmse(y, y_hat, epsilon=0.0):
    # y_hat = tf.clip_by_value(y_hat, 1, 5)
    m = _get_mask(y)
    return tf.sqrt(
        (tf.reduce_sum(tf.multiply(tf.square(y - y_hat), m)) + epsilon) /
        (tf.reduce_sum(m) + epsilon))


def init_models(n_col, args):
    x = tf.placeholder(tf.float32, [None, n_col], name="x")
    y = tf.placeholder(tf.float32, [None, n_col], name="y")
    training = tf.placeholder(tf.bool, name="training")

    # init weights
    weights, biases = _init_weights(n_col, args)
    y_hat = _forward(x, weights, biases, training, args)
    loss = _get_loss(y, y_hat, weights, args)
    rmse = _get_rmse(y, y_hat)

    if args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
    elif args.optimizer == 'rprop':
        optimizer = RPropOptimizer()
    else:
        raise NotImplementedError()

    train_op = optimizer.minimize(loss, var_list=weights + biases)

    return {
        'x': x,
        'y': y,
        'training': training,
        'train_op': train_op,
        'rmse': rmse,
        'loss': loss,
    }
