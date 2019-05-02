import math

import tensorflow as tf


def _forward(x, weights, biases, training, args):
    if args.activation_fn == 'relu':
        activation_fn = tf.nn.relu
    elif args.activation_fn == 'tanh':
        activation_fn = tf.nn.tanh
    else:
        assert False

    layer = x
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.matmul(layer, weight) + bias
        if i < len(biases) - 1:
            layer = activation_fn(layer)
            layer = tf.layers.dropout(
                layer, rate=args.drop_rate, training=training)

            if args.normalize:
                divider = tf.maximum(
                    tf.cast(tf.math.count_nonzero(layer, axis=1), tf.float32),
                    1)
                divider = tf.matmul(
                    tf.expand_dims(divider, axis=1),
                    tf.ones([1, layer.shape[1]]))
                if args.drop_rate and args.drop_rate > 1e-8:
                    divider /= tf.where(training, args.drop_rate, 1.0)
                layer /= divider

    return layer


def _init_weights(args):
    n_units = [784] + [args.n_hidden_unit] * args.n_hidden_layer + [10]

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


def init_models(args):
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y = tf.placeholder(tf.int32, [None], name="y")
    training = tf.placeholder(tf.bool, name="training")

    y_hot = tf.one_hot(y, 10)
    weights, biases = _init_weights(args)

    y_hat = _forward(x, weights, biases, training, args)

    costs = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_hot, logits=y_hat)
    loss = tf.reduce_sum(costs)
    accuracy = _get_accuracy(y_hot, y_hat)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss)

    return {
        'x': x,
        'y': y,
        'training': training,
        'loss': loss,
        'accuracy': accuracy,
        'train_op': train_op,
    }
