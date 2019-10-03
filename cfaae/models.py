import random
import math

import tensorflow as tf


def _init_weights(n_col, args):
    weights, biases = [], []

    n_units = [n_col] + [500] + [n_col]

    for n_in_unit, n_out_unit in zip(n_units[:-1], n_units[1:]):
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
        if i == 0:
            layer = tf.sparse.matmul(layer, weight)
            layer = tf.where(tf.is_nan(layer), tf.zeros_like(layer), layer)
        else:
            layer = tf.matmul(layer, weight)
        layer = layer + bias

        if i < len(biases) - 1:
            layer = tf.nn.relu(layer)

    return layer


def _get_mask(y):
    zero = tf.constant(0.1, dtype=tf.float32)
    where = tf.greater(y, zero)
    m = tf.cast(where, tf.float32)
    return m


def _get_rec_loss(y, y_hat):
    _y_hat = tf.gather_nd(y_hat, y.indices)
    rec_loss = tf.reduce_sum(tf.square(y.values - _y_hat))
    return rec_loss


def _get_loss(y, y_hat, weights, args):
    _y_hat = tf.gather_nd(y_hat, y.indices)
    rec_loss = tf.reduce_sum(tf.square(y.values - _y_hat))

    reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in weights])
    return rec_loss + 0.5 * args.lambda_w * reg_loss


def _get_rmse(y, y_hat, args, epsilon=0.0):
    _y_hat = tf.gather_nd(y_hat, y.indices)
    if args.clipping:
        _y_hat = tf.clip_by_value(_y_hat, 1, 5)
    return tf.sqrt(tf.reduce_mean(tf.square(y.values - _y_hat)))


def init_models(n_col, args):
    x = tf.sparse.placeholder(tf.float32, name="x")
    y = tf.sparse.placeholder(tf.float32, name="y")
    training = tf.placeholder(tf.bool, name="training")

    # init weights
    weights, biases = _init_weights(n_col, args)
    y_hat = _forward(x, weights, biases, training, args)
    loss = _get_loss(y, y_hat, weights, args)
    rmse = _get_rmse(y, y_hat, args)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss, var_list=weights + biases)

    return {
        'x': x,
        'y': y,
        'weights': weights,
        'biases': biases,
        'y_hat': y_hat,
        'training': training,
        'loss': loss,
        'rmse': rmse,
        'train_op': train_op,
    }
