import math

import tensorflow as tf

from common.utils import get_activation_fn


def _forward(x, weights, biases, training, args):
    activation_fn = get_activation_fn(args.activation_fn)

    layer = x
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.matmul(layer, weight) + bias
        if i < len(biases) - 1:
            layer = activation_fn(layer)
<<<<<<< HEAD
            if args.dropout_rate > 1e-8:
                layer = tf.layers.dropout(
                    layer, rate=args.dropout_rate, training=training)
    return layer
=======
            layers.append(layer)
            layer = tf.layers.dropout(
                layer, rate=args.drop_rate, training=training)
        else:
            layers.append(layer)

    return layers
>>>>>>> 8b11627036ac569e926c43e9ae35604f2a49f415


def _init_weights(args, n_feature, n_class):
    n_units = [n_feature] + [args.n_hidden_unit] * args.n_hidden_layer + [
        n_class if n_class else 1
    ]

    weights, biases = [], []
    for n_in, n_out in zip(n_units[:-1], n_units[1:]):
        weight = tf.Variable(
            tf.random.truncated_normal(
                [n_in, n_out], mean=0, stddev=math.sqrt(1 / n_in)))
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
    y_hat = _forward(x, weights, biases, training, args)

    if n_class is not None:
        assert n_class > 0
        _y = tf.cast(y, tf.int64)
        _y_hot = tf.one_hot(_y, n_class)
        # activation = [tf.reduce_mean(layer) for layer in layers]
        _costs = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=_y_hot, logits=y_hat)
        loss = tf.reduce_sum(_costs)
        error = 1.0 - _get_accuracy(_y_hot, y_hat)

        y_hat = tf.exp(y_hat)
        # y_hat = tf.nn.softmax(y_hat)
    else:
<<<<<<< HEAD
        _y_hat = tf.reshape(y_hat, [-1])
=======
        _y_hat= tf.reshape(y_hat, [-1])
>>>>>>> 8b11627036ac569e926c43e9ae35604f2a49f415
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
        'y_hat': y_hat,
        'train_op': train_op,
    }
