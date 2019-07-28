import random
import math

import tensorflow as tf


def _init_weights(n_units):
    weights, biases = [], []

    for n_in_unit, n_out_unit in zip(n_units[:-1], n_units[1:]):
        limit = math.sqrt(6 / n_in_unit)

        weight = tf.Variable(
            tf.random_uniform([n_in_unit, n_out_unit], -limit, limit))
        bias = tf.Variable(tf.zeros([n_out_unit]))
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def _forward(x, weights, biases, tensordot=True):
    assert len(weights) == len(biases)

    layer = x
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        if tensordot:
            layer = tf.tensordot(layer, weight, [[2], [0]])
        else:
            layer = tf.matmul(layer, weight)
        layer += bias

        if i < len(biases) - 1:
            layer = tf.nn.tanh(layer)
    return layer


def init_models():
    x_index = tf.placeholder(tf.int32, [None, None], name="x")
    y = tf.placeholder(tf.float32, [None], name="y")
    image = tf.placeholder(tf.float32, [None, 784], name="image")
    x = tf.nn.embedding_lookup(image, x_index)
    m = tf.placeholder(tf.float32, [None, None, 1], name="m")

    # init weights
    encoder_weights, encoder_biases = _init_weights([784, 300, 100, 30])
    decoder_weights, decoder_biases = _init_weights([30, 1])

    # build graph
    embeddings = tf.nn.tanh(
        _forward(x, encoder_weights, encoder_biases, tensordot=True))

    embeddings = embeddings * m
    embedding = tf.reduce_sum(
        embeddings, axis=1) / tf.reduce_sum(
            m, axis=1)
    y_hat = _forward(
        embedding, decoder_weights, decoder_biases, tensordot=False)
    loss = tf.reduce_sum(tf.abs(y - y_hat))
    mae = tf.reduce_mean(tf.abs(y - y_hat))

    optimizer = tf.train.AdamOptimizer(1e-1)
    train_op = optimizer.minimize(loss)

    return {
        'x_index': x_index,
        'm': m,
        'image': image,
        'y': y,
        'loss': loss,
        'mae': mae,
        'train_op': train_op,
    }
