import random
import math

import tensorflow as tf

from .configs import *

# projection matrix for sharing
PMS = [
    [1., 0., 0., 0., 0.],  # for rating 1
    [1., 1., 0., 0., 0.],  # for rating 2
    [1., 1., 1., 0., 0.],  # for rating 3
    [1., 1., 1., 1., 0.],  # for rating 4
    [1., 1., 1., 1., 1.],  # for rating 5
]


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
        # weight = tf.Variable(
        #     tf.random.truncated_normal(
        #         [n_in_unit, n_out_unit], mean=0, stddev=2 / n_in_unit))
        bias = tf.Variable(tf.zeros([n_out_unit]))
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def _forward(x_hot, weights, biases, args, training):
    assert len(weights) == len(biases)

    if args.parameter_sharing:
        _x_hot = tf.tensordot(x_hot, PMS, axes=[[2], [0]])
    else:
        _x_hot = x_hot

    layer = tf.reshape(_x_hot, [-1, weights[0].shape[0]])
    input_layer = layer

    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.matmul(layer, weight) + bias

        if i < len(biases) - 1:
            layer = tf.nn.tanh(layer)

    layer = tf.reshape(layer, tf.shape(x_hot))
    if args.parameter_sharing:
        layer = tf.tensordot(layer, PMS, axes=[[2], [0]])
    return layer


def _get_loss(x_hot, y_hot, y_hat, v, weights, args):
    _y = tf.reduce_sum(y_hot, axis=2)
    zero = tf.constant(0.1, dtype=tf.float32)
    _where = tf.greater(_y, zero)
    m = tf.cast(_where, tf.float32)

    if args.use_ordinal_cost:
        # FIRST_IMPLEMENTATION = True
        FIRST_IMPLEMENTATION = False

        if FIRST_IMPLEMENTATION:
            mask_1k = tf.clip_by_value(
                tf.cumsum(y_hot, reverse=True, axis=2), 0, 1)
            mask_k1 = tf.clip_by_value(tf.cumsum(y_hot, axis=2), 0, 1)
            y_hat_1k = y_hat * mask_1k
            y_hat_k1 = y_hat * mask_k1
            costs = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y_hot,
                logits=y_hat_1k) + tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=y_hot, logits=y_hat_k1)
            costs = m * costs
            costs = tf.reduce_sum(costs, axis=1) * v
            rec_loss = tf.reduce_sum(costs)

        else:
            y_prob = tf.nn.softmax(y_hat)

            mask_1k = tf.clip_by_value(
                tf.cumsum(y_hot, reverse=True, axis=2), 0, 1)
            mask_k1 = tf.clip_by_value(tf.cumsum(y_hot, axis=2), 0, 1)
            y_prob_1k = tf.cumsum(y_prob, axis=2)
            y_prob_k1 = tf.cumsum(y_prob, reverse=True, axis=2)

            ordinal_costs_1k = -tf.reduce_sum(
                (tf.log(y_prob) - tf.log(y_prob_1k)) * mask_1k, axis=2)
            ordinal_costs_k1 = -tf.reduce_sum(
                (tf.log(y_prob) - tf.log(y_prob_k1)) * mask_k1, axis=2)
            ordinal_costs = ordinal_costs_1k + ordinal_costs_k1

            d = tf.reduce_sum(tf.reduce_sum(x_hot, axis=2), axis=1)
            D = tf.reduce_sum(tf.reduce_sum(x_hot + y_hot, axis=2), axis=1)

            costs = tf.reduce_sum(ordinal_costs, axis=1) * D / (D - d + 1e-6)
            rec_loss = tf.reduce_mean(costs)
    else:
        costs = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_hot, logits=y_hat)
        costs = m * costs
        costs = tf.reduce_sum(costs, axis=1) * v
        rec_loss = tf.reduce_sum(costs)

    return rec_loss + tf.add_n([
        l * tf.reduce_sum(tf.square(w))
        for l, w in zip([args.lambda_1, args.lambda_2], weights)
    ])


def _get_rmse(y, y_hat, v, epsilon=0):
    y_hat = tf.where(v > 0.1, y_hat, tf.ones_like(y_hat) * 3)

    y = tf.cast(y, tf.float32)
    zero = tf.constant(0.1, dtype=tf.float32)
    where = tf.greater(y, zero)
    m = tf.cast(where, tf.float32)

    return tf.sqrt(
        (tf.reduce_sum(tf.multiply(tf.square(y - y_hat), m)) + epsilon) /
        (tf.reduce_sum(m) + epsilon))


def _get_y_hat_from_y_hat_hot(y_hat_hot):
    y_hat = tf.reduce_sum(
        tf.multiply(tf.nn.softmax(y_hat_hot), [1., 2., 3., 4., 5.]), axis=2)
    return y_hat


def init_models(n_col, n_class, args):
    x = tf.placeholder(tf.int64, [None, n_col], name="x")
    y = tf.placeholder(tf.int64, [None, n_col], name="y")
    training = tf.placeholder(tf.bool, name="training")

    x_hot = tf.one_hot(x - 1, 5)
    y_hot = tf.one_hot(y - 1, 5)
    v = tf.cast(tf.greater(tf.reduce_sum(x, axis=1), 0), tf.float32)  # valid

    # init weights
    weights, biases = _init_weights(n_col * n_class, args)
    y_hat_hot = _forward(x_hot, weights, biases, args, training)
    loss = _get_loss(x_hot, y_hot, y_hat_hot, v, weights, args)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss, var_list=weights + biases)

    y_hat = _get_y_hat_from_y_hat_hot(y_hat_hot)
    rmse = _get_rmse(y, y_hat, v)

    return {
        'x': x,
        'y': y,
        'training': training,
        'loss': loss,
        'train_op': train_op,
        'rmse': rmse
    }
