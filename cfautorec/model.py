import random
import math

import tensorflow as tf

from .configs import *
from .rprop import RPropOptimizer


def _init_weights(n_col, args):
    weights, biases = [], []

    n_units = [n_col] + [args.n_hidden_unit] * args.n_hidden_layer + [n_col]

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

    if args.activation_fn == 'sigmoid':
        activation_fn = tf.nn.sigmoid
    elif args.activation_fn == 'relu':
        activation_fn = tf.nn.relu
    elif args.activation_fn == 'relu6':
        activation_fn = tf.nn.relu6
    elif args.activation_fn == 'elu':
        activation_fn = tf.nn.elu
    elif args.activation_fn == 'leaky_relu':
        activation_fn = tf.nn.leaky_relu
    elif args.activation_fn == 'identity':
        activation_fn = lambda x: x
    elif args.activation_fn == 'tanh':
        activation_fn = tf.nn.tanh
    elif args.activation_fn == 'softplus':
        activation_fn = tf.math.softplus
    else:
        raise NotImplementedError()

    layer = train_x
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        if i == 0 and args.sparse_tensor:
            layer = tf.sparse.matmul(layer, weight)
            layer = tf.where(tf.is_nan(layer), tf.zeros_like(layer), layer)
        else:
            layer = tf.matmul(layer, weight)
            if args.imputation == 'batch-norm':
                layer = tf.layers.batch_normalization(layer, training=training)
            elif args.imputation == 'layer-norm':
                layer = tf.contrib.layers.layer_norm(layer)
        layer = layer + bias

        if i < len(biases) - 1:
            layer = activation_fn(layer)
            if args.dropout_rate > 1e-8:
                layer = tf.layers.dropout(
                    layer, rate=args.dropout_rate, training=training)
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
    if args.sparse_tensor:
        _y_hat = tf.gather_nd(y_hat, y.indices)
        rec_loss = tf.reduce_sum(tf.square(y.values - _y_hat))
    else:
        m = _get_mask(y)
        m_user = _get_mask(tf.reduce_sum(m, axis=1))
        error = tf.square(m * (y - y_hat))
        user_error = tf.reduce_sum(error, axis=1)
        rec_loss = tf.reduce_sum(user_error * m_user)

    reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in weights])
    return rec_loss + 0.5 * args.lambda_w * reg_loss


def _get_rmse(y, y_hat, args, epsilon=0.0):
    # y_hat = tf.clip_by_value(y_hat, 1, 5)
    if args.sparse_tensor:
        _y_hat = tf.gather_nd(y_hat, y.indices)
        if args.clipping:
            _y_hat = tf.clip_by_value(_y_hat, 1, 5)
        return tf.sqrt(tf.reduce_mean(tf.square(y.values - _y_hat)))
    else:
        m = _get_mask(y)
        return tf.sqrt(
            (tf.reduce_sum(tf.multiply(tf.square(y - y_hat), m)) + epsilon) /
            (tf.reduce_sum(m) + epsilon))


def init_models(n_col, args):
    if args.sparse_tensor:
        x = tf.sparse.placeholder(tf.float32, name="x")
        y = tf.sparse.placeholder(tf.float32, name="y")
    else:
        x = tf.placeholder(tf.float32, [None, n_col], name="x")
        y = tf.placeholder(tf.float32, [None, n_col], name="y")
    training = tf.placeholder(tf.bool, name="training")

    # init weights
    weights, biases = _init_weights(n_col, args)
    y_hat = _forward(x, weights, biases, training, args)
    loss = _get_loss(y, y_hat, weights, args)
    rmse = _get_rmse(y, y_hat, args)

    if args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
    elif args.optimizer == 'rprop':
        optimizer = RPropOptimizer()
    else:
        raise NotImplementedError()

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        # train_op = optimizer.minimize(loss, var_list=weights + biases)
        train_op = optimizer.minimize(loss)

    return {
        'x':
        x,
        'y':
        y,
        'weights':
        weights,
        'biases':
        biases,
        'y_hat':
        y_hat,
        'training':
        training,
        'loss':
        loss,
        'rmse':
        rmse,
        'train_op':
        train_op,
        'ln-variables':
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='LayerNorm'),
        'bn-variables':
        tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='batch_normalization')
    }
