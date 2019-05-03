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

            # if args.normalize:
            #     divider = tf.maximum(
            #         tf.cast(tf.math.count_nonzero(layer, axis=1), tf.float32),
            #         1)
            #     divider = tf.matmul(
            #         tf.expand_dims(divider, axis=1),
            #         tf.ones([1, layer.shape[1]]))
            #     if args.drop_rate and args.drop_rate > 1e-8:
            #         divider /= tf.where(training, args.drop_rate, 1.0)
            #     layer /= divider
        else:
            layers.append(layer)
    return layers


def _init_weights(args, n_feature):
    n_units = [n_feature] + [args.n_hidden_unit] * args.n_hidden_layer + [1]

    weights, biases = [], []
    for n_in, n_out in zip(n_units[:-1], n_units[1:]):
        limit = math.sqrt(6 / n_in)
        # limit = 1.0

        weight = tf.Variable(tf.random_uniform([n_in, n_out], -limit, limit))
        bias = tf.Variable(tf.zeros([n_out]))
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def init_models(args, n_feature):
    x = tf.placeholder(tf.float32, [None, n_feature], name="x")
    y = tf.placeholder(tf.float32, [None], name="y")
    training = tf.placeholder(tf.bool, name="training")

    weights, biases = _init_weights(args, n_feature)
    layers = _forward(x, weights, biases, training, args)
    mean_activations = [tf.reduce_mean(layer) for layer in layers]
    y_hat = tf.reshape(layers[-1], [-1])

    loss = tf.reduce_sum(tf.square(y - y_hat))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_hat)))

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss)

    return {
        'x': x,
        'y': y,
        'training': training,
        'loss': loss,
        'rmse': rmse,
        'mean_activations': mean_activations,
        'train_op': train_op,
    }
