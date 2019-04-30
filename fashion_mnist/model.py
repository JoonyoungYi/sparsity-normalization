import math

import tensorflow as tf


def _forward(x, weights, biases, activation_fn=tf.nn.relu):
    layer = x
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.matmul(layer, weight) + bias
        if i < len(biases) - 1:
            layer = activation_fn(layer)
    return layer


def _init_weights():
    n_units = [784, 100, 100, 100, 10]

    weights, biases = [], []
    for n_in, n_out in zip(n_units[:-1], n_units[1:]):
        limit = math.sqrt(6 / n_in)

        weight = tf.Variable(tf.random_uniform([n_in, n_out], -limit, limit))
        bias = tf.Variable(tf.zeros([n_out]))
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def _get_accuracy(y_hot, y_hat):
    return tf.count_nonzero(
        tf.cast(tf.equal(tf.argmax(y_hot, 1), tf.argmax(y_hat, 1)),
                tf.int64)) / tf.cast(tf.shape(y_hot)[0], tf.int64)


def init_models():
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y = tf.placeholder(tf.int32, [None], name="y")
    y_hot = tf.one_hot(y, 10)
    weights, biases = _init_weights()

    y_hat = _forward(x, weights, biases)

    costs = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_hot, logits=y_hat)
    loss = tf.reduce_sum(costs)
    accuracy = _get_accuracy(y_hot, y_hat)
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(loss)

    return {
        'loss': loss,
        'accuracy': accuracy,
        'x': x,
        'y': y,
        'train_op': train_op,
    }
