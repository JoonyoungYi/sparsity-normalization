import random
import math

import tensorflow as tf


def __init_weights(n_units):
    weights, biases = [], []
    for n_in_unit, n_out_unit in zip(n_units[:-1], n_units[1:]):
        limit = math.sqrt(3 / n_in_unit)
        weight = tf.Variable(
            tf.random_uniform([n_in_unit, n_out_unit], -limit, limit))
        bias = tf.Variable(tf.zeros([n_out_unit]))
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def _init_uae_weights(n_item, args):
    if args.dataset == 'movielens-100k':
        return __init_weights([n_item, 100, 50])
    elif args.dataset == 'movielens-1m':
        return __init_weights([n_item, 600, 400, 500])
    else:
        raise NotImplementedError()


def _init_iae_weights(n_user, args):
    if args.dataset == 'movielens-100k':
        return __init_weights([n_user, 100, 50])
    elif args.dataset == 'movielens-1m':
        return __init_weights([n_user, 400, 500])
    else:
        raise NotImplementedError()


def _forward(x, c, weights, biases):
    assert len(weights) == len(biases)

    layer = x
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        if i == 0:
            layer = tf.sparse.matmul(layer, weight)
            layer = tf.where(tf.is_nan(layer), tf.zeros_like(layer), layer)
            layer /= c
        else:
            layer = tf.matmul(layer, weight)
        layer = layer + bias

        if i < len(biases) - 1:
            layer = tf.nn.relu(layer)

    layer = tf.nn.sigmoid(layer)
    # layer = tf.nn.relu(layer)
    # layer = tf.nn.tanh(layer)
    return layer


def _get_mask(y):
    zero = tf.constant(0.1, dtype=tf.float32)
    where = tf.greater(y, zero)
    m = tf.cast(where, tf.float32)
    return m


def _get_rec_loss(y, y_hat):
    rec_loss = tf.reduce_sum(tf.square(y - y_hat))
    return rec_loss


def _get_loss(y, y_hat, weights, args):
    rec_loss = tf.reduce_mean(tf.square(y - y_hat))
    reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in weights])
    return rec_loss + 0.5 * args.lambda_w * reg_loss


def _get_rmse(y, y_hat, args, epsilon=0.0):
    if args.clipping:
        y_hat = tf.clip_by_value(y_hat, 1, 5)
    return tf.sqrt(tf.reduce_mean(tf.square(y - y_hat)))


def init_models(n_user, n_item, args):
    # x matrix는 n_user * n_item으로 생각한다.
    user_x = tf.sparse.placeholder(tf.float32, name="user_x")
    user_c = tf.placeholder(tf.float32, [None, 1], name="user_c")
    item_x = tf.sparse.placeholder(tf.float32, name="item_x")
    item_c = tf.placeholder(tf.float32, [None, 1], name="item_c")
    y = tf.placeholder(tf.float32, [None], name="y")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    # init weights
    u_weights, u_biases = _init_uae_weights(n_item, args)
    i_weights, i_biases = _init_iae_weights(n_user, args)

    user_embed = _forward(user_x, user_c, u_weights, u_biases)
    item_embed = _forward(item_x, item_c, i_weights, i_biases)
    y_hat = tf.reduce_sum(tf.multiply(user_embed, item_embed), axis=1)

    loss = _get_loss(y, y_hat, u_weights + i_weights, args)
    rmse = _get_rmse(y, y_hat, args)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    u_train_op = optimizer.minimize(loss, var_list=u_weights + u_biases)
    i_train_op = optimizer.minimize(loss, var_list=i_weights + i_biases)

    return {
        'user_x': user_x,
        'item_x': item_x,
        'user_c': user_c,
        'item_c': item_c,
        'y': y,
        'y_hat': y_hat,
        'loss': loss,
        'rmse': rmse,
        'u_train_op': u_train_op,
        'i_train_op': i_train_op,
        'learning_rate': learning_rate,
    }
