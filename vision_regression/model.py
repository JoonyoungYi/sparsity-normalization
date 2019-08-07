import math

import tensorflow as tf


def _forward(x, weights, biases, training, args):
    if args.activation_fn == 'relu':
        activation_fn = tf.nn.relu
    elif args.activation_fn == 'elu':
        activation_fn = tf.nn.elu
    elif args.activation_fn == 'leaky-relu':
        activation_fn = tf.nn.leaky_relu
    elif args.activation_fn == 'tanh':
        activation_fn = tf.nn.tanh
    elif args.activation_fn == 'sigmoid':
        activation_fn = tf.nn.sigmoid
    else:
        raise NotImplementedError()

    layer = x
    layers = []
    assert len(weights) == len(biases)
    n_layer = len(weights)
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.matmul(layer, weight)
        layer = tf.add(layer, bias)

        if i < n_layer - 1:
            layer = activation_fn(layer)
            layers.append(layer)
        else:
            layers.append(layer)
    return layers


def _init_weights(args, n_in, n_out):
    n_units = [n_in] + [args.n_hidden_unit] * args.n_hidden_layer + [n_out]

    weights, biases = [], []
    for i, (n_in, n_out) in enumerate(zip(n_units[:-1], n_units[1:])):
        if (i == 0 and args.sparsity_normalization):
            limit = math.sqrt(2 / args.K_prime)
        else:
            limit = math.sqrt(2 / n_in)

        weight = tf.Variable(tf.truncated_normal([n_in, n_out], mean=0, stddev=limit))
        bias = tf.Variable(tf.zeros([n_out]))
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def init_models(args, n_in, n_out):
    x = tf.placeholder(tf.float32, [None, n_in], name="x")
    y = tf.placeholder(tf.float32, [None], name="y")
    training = tf.placeholder(tf.bool, name="training")

    weights, biases = _init_weights(args, n_in, n_out)

    layers = _forward(x, weights, biases, training, args)
    y_hat = tf.reshape(layers[-1], [-1])

    costs = tf.square(y - y_hat)
    loss = tf.reduce_sum(costs)
    accuracy = -tf.reduce_mean(costs)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss)

    return {
        'x': x,
        'y': y,
        'training': training,
        'loss': loss,
        'accuracy': accuracy,
        'layers': layers,
        'train_op': train_op,
    }
