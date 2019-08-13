import math

import tensorflow as tf

from common.utils import get_activation_fn


def _forward(x, weights, biases, training, args):
    activation_fn = get_activation_fn(args.activation_fn)

    layer = x
    assert len(weights) == len(biases)
    n_layer = len(weights)
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.matmul(layer, weight) + bias
        if i < n_layer - 1:
            layer = activation_fn(layer)

            if args.mode == 'none':
                pass
            elif args.mode == 'default' or args.mode == 'default-with-sn':
                layer = tf.layers.dropout(
                    layer, args.drop_rate, training=training)
            elif args.mode == 'expectation':
                r = tf.random.uniform(
                    (1, tf.shape(layer)[1]), minval=0., maxval=1.)
                m = tf.where(r < args.drop_rate,
                             tf.zeros_like(r), tf.ones_like(r))
                train_layer = tf.multiply(layer, m) / (1 - args.drop_rate)
                layer = tf.cond(training, lambda: train_layer, lambda: layer)
            elif args.mode == 'exact':
                r = tf.random.uniform(
                    (1, tf.shape(layer)[1]), minval=0., maxval=1.)
                m = tf.where(r < args.drop_rate,
                             tf.zeros_like(r), tf.ones_like(r))
                train_layer = tf.multiply(layer, m) * tf.cast(
                    tf.shape(layer)[1], tf.float32) / tf.reduce_sum(
                        m, keepdims=True)
                layer = tf.cond(training, lambda: train_layer, lambda: layer)
            else:
                raise NotImplementedError()
    return layer


def _init_weights(args, n_in, n_out):
    n_units = [n_in] + [args.n_hidden_unit] * args.n_hidden_layer + [n_out]

    weights, biases = [], []
    for i, (n_in, n_out) in enumerate(zip(n_units[:-1], n_units[1:])):
        weight = tf.Variable(
            tf.truncated_normal(
                [n_in, n_out], mean=0, stddev=math.sqrt(3 / n_in)))
        bias = tf.Variable(tf.zeros([n_out]))
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def _get_accuracy(y_hot, y_hat):
    return tf.count_nonzero(
        tf.cast(tf.equal(tf.argmax(y_hot, 1), tf.argmax(y_hat, 1)),
                tf.int64)) / tf.cast(tf.shape(y_hot)[0], tf.int64)


def init_models(args, n_in, n_out):
    x = tf.placeholder(tf.float32, [None, n_in], name="x")
    y = tf.placeholder(tf.int32, [None], name="y")
    training = tf.placeholder(tf.bool, name="training")

    y_hot = tf.one_hot(y, n_out)
    weights, biases = _init_weights(args, n_in, n_out)

    y_hat = _forward(x, weights, biases, training, args)
    y_prob = tf.nn.softmax(y_hat)

    costs = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_hot, logits=y_hat)
    loss = tf.reduce_sum(costs)
    accuracy = _get_accuracy(y_hot, y_hat)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss, var_list=weights + biases)

    return {
        'x': x,
        'y': y,
        'y_prob': y_prob,
        'training': training,
        'loss': loss,
        'accuracy': accuracy,
        'train_op': train_op,
    }
