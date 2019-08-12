import math

import tensorflow as tf

from .modules import DropConnect
from common.utils import get_activation_fn


def _forward(x, weights, biases, training, args):
    activation_fn = get_activation_fn(args.activation_fn)

    layer = x
    assert len(weights) == len(biases)
    n_layer = len(weights)
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        if i < n_layer - 1:
            if i == 0:
                layer = tf.matmul(layer, weight) + bias
            else:
                drop_connect = DropConnect(
                    drop_rate=args.dropconnect_rate,
                    activation_fn=activation_fn,
                    Z=args.Z,
                    mode=args.dropconnect_mode)
                layer = drop_connect(layer, weight, bias, training=training)
                # weight_m_number = drop_connect.weight_m_number

            if args.dropout_rate > 1e-8:
                layer = tf.layers.dropout(
                    layer, args.dropout_rate, training=training)
        else:
            layer = tf.matmul(layer, weight) + bias
    return layer  # , weight_m_number


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

    # y_hat, weight_m_number = _forward(x, weights, biases, training, args)
    y_hat = _forward(x, weights, biases, training, args)

    costs = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_hot, logits=y_hat)
    loss = tf.reduce_sum(costs)
    accuracy = _get_accuracy(y_hot, y_hat)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss, var_list=weights + biases)

    return {
        'x': x,
        'y': y,
        'training': training,
        'loss': loss,
        'accuracy': accuracy,
        'train_op': train_op,
        # 'weight_m_number': weight_m_number,
    }
