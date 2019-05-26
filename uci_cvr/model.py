import math

import tensorflow as tf

from .configs import *


def _forward(x, m, weights, biases, args, activation_fn=tf.nn.relu):
    if args.kind == KIND_CONCAT_NEG:
        layer = tf.concat([x, 1 - m], 1)
    elif args.kind == KIND_CONCAT_POS:
        layer = tf.concat([x, m], 1)
    else:
        layer = x

    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.matmul(layer, weight) + bias
        if i < len(biases) - 1:
            layer = activation_fn(layer)
            # layer = tf_layers.batch_norm(layer, activation_fn=activation_fn)
    return layer


def _init_weights(args):
    if args.kind == KIND_CONCAT_POS or args.kind == KIND_CONCAT_NEG:
        n_units = [32 + 16] + [args.n_hidden_unit] * args.n_hidden_layer + [2]
    else:
        n_units = [32] + [args.n_hidden_unit] * args.n_hidden_layer + [2]

    weights, biases = [], []
    for n_in, n_out in zip(n_units[:-1], n_units[1:]):
        limit = math.sqrt(6 / n_in)

        weight = tf.Variable(tf.random_uniform([n_in, n_out], -limit, limit))
        bias = tf.Variable(tf.zeros([n_out]))
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def _get_accuracy(y, y_hat):
    return tf.count_nonzero(
        tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1)),
                tf.int64)) / tf.cast(tf.shape(y)[0], tf.int64)


def init_models(args):
    x = tf.placeholder(tf.float32, [None, 32], name="x")

    y = tf.placeholder(tf.float32, [None, 2], name="y")
    m = tf.placeholder(tf.float32, [None, 16], name="m")
    weights, biases = _init_weights(args)

    y_hat = _forward(x, m, weights, biases, args)

    costs = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_hat)
    loss = tf.reduce_sum(costs)
    accuracy = _get_accuracy(y, y_hat)
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss)

    return {
        'loss': loss,
        'accuracy': accuracy,
        'x': x,
        'y': y,
        'm': m,
        'train_op': train_op,
    }
