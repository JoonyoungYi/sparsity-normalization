import math

import tensorflow as tf

KIND_DEFAULT = 'default'
KIND_NORMALIZATION = 'normalization'
KIND_CONCATENATION = 'concatenation'
KIND_AUGMENT_NORMAL = 'augment-normal'  # normal value로 augment하는 기법.
KIND = KIND_AUGMENT_NORMAL

if KIND == KIND_CONCATENATION:
    n_units = [16 * 2, 32, 32, 2]
else:
    n_units = [16, 32, 32, 2]


def _forward(x, m, weights, biases, activation_fn=tf.nn.relu):
    if KIND == KIND_CONCATENATION:
        layer = tf.concat([x, 1 - m], 1)
    elif KIND == KIND_AUGMENT_NORMAL:
        layer = x + 0.5 * (1 - m)
    else:
        layer = x

    for i, (weight, bias) in enumerate(zip(weights, biases)):

        layer = tf.matmul(layer, weight)
        if i == 0 and KIND == KIND_NORMALIZATION:
            c = tf.clip_by_value(
                tf.matmul(
                    tf.reshape(tf.reduce_sum(x, axis=1), [-1, 1]),
                    tf.ones([1, 32])), 1, 16)
            layer = layer / c
        layer = layer + bias

        if i < len(biases) - 1:
            layer = activation_fn(layer)
            # layer = tf_layers.batch_norm(layer, activation_fn=activation_fn)
    return layer


def _init_weights():
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


def init_models():
    x = tf.placeholder(tf.float32, [None, 16], name="x")
    y = tf.placeholder(tf.float32, [None, 2], name="y")
    m = tf.placeholder(tf.float32, [None, 16], name="m")
    weights, biases = _init_weights()

    y_hat = _forward(x, m, weights, biases)

    costs = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
    loss = tf.reduce_sum(costs)
    accuracy = _get_accuracy(y, y_hat)
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(loss)

    return {
        'loss': loss,
        'accuracy': accuracy,
        'x': x,
        'y': y,
        'm': m,
        'train_op': train_op,
    }
