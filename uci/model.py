import math

import tensorflow as tf


def _forward(x, weights, biases, training, args):
    if args.activation_fn == 'relu':
        activation_fn = tf.nn.relu
    elif args.activation_fn == 'tanh':
        activation_fn = tf.nn.tanh
    elif args.activation_fn == 'sigmoid':
        activation_fn = tf.nn.sigmoid
    else:
        raise NotImplementedError()

    layer = x
    layers = []
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.matmul(layer, weight) + bias
        if i < len(biases) - 1:
            layer = activation_fn(layer)
            layers.append(layer)
            layer = tf.layers.dropout(
                layer, rate=args.drop_rate, training=training)
        else:
            layers.append(layer)

    return layers


def _init_weights(args, n_feature, n_class):
    n_units = [n_feature] + [args.n_hidden_unit] * args.n_hidden_layer + [
        n_class if n_class else 1
    ]

    weights, biases = [], []
    for n_in, n_out in zip(n_units[:-1], n_units[1:]):
        limit = math.sqrt(6 / n_in)
        # limit = 1.0

        weight = tf.Variable(tf.random_uniform([n_in, n_out], -limit, limit))
        bias = tf.Variable(tf.zeros([n_out]))
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def _get_accuracy(y, y_hat):
    return tf.count_nonzero(
        tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1)),
                tf.int64)) / tf.cast(tf.shape(y)[0], tf.int64)


def _get_rmse(y, y_hat):
    return tf.sqrt(tf.reduce_mean(tf.square(y - y_hat)))


def init_models(args, n_feature, n_class=None):
    # 'n_class == None' means it is regression problem.
    x = tf.placeholder(tf.float32, [None, n_feature], name="x")
    y = tf.placeholder(tf.float32, [None], name="y")
    training = tf.placeholder(tf.bool, name="training")

    weights, biases = _init_weights(args, n_feature, n_class)
    layers = _forward(x, weights, biases, training, args)
    activations = [tf.reduce_mean(layer) for layer in layers]
    y_hat = layers[-1]

    if n_class is not None:
        assert n_class > 0
        _y_hot = tf.one_hot(tf.cast(y, tf.int64), n_class)
        _costs = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=_y_hot, logits=y_hat)
        loss = tf.reduce_sum(_costs)
        error = 1.0 - _get_accuracy(_y_hot, y_hat)
    else:
        _y_hat= tf.reshape(y_hat, [-1])
        loss = tf.reduce_sum(tf.square(y - _y_hat))
        error = _get_rmse(y, _y_hat)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss, var_list=weights + biases)

    return {
        'x': x,
        'y': y,
        'training': training,
        'loss': loss,
        'error': error,
        'activations': activations,
        'train_op': train_op,
    }
