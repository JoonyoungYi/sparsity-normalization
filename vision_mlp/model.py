import math

import tensorflow as tf


def _forward(x, weights, biases, training, args):
    if args.activation_fn == 'relu':
        activation_fn = tf.nn.relu
    elif args.activation_fn == 'leaky_relu':
        activation_fn = tf.nn.leaky_relu
    elif args.activation_fn == 'tanh':
        activation_fn = tf.nn.tanh
    elif args.activation_fn == 'sigmoid':
        activation_fn = tf.nn.sigmoid
    elif args.activation_fn == 'softplus':
        activation_fn = tf.nn.softplus
    elif args.activation_fn == 'elu':
        activation_fn = tf.nn.elu
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

    if args.K_prime_opt == 'n0': 
        K_prime = n_in
    elif args.K_prime_opt == 'n1':
        raise NotImplementedError()
    elif args.K_prime_opt == 'custom':
        K_prime = args.K_prime

    weights, biases = [], []
    for i, (n_in, n_out) in enumerate(zip(n_units[:-1], n_units[1:])):
        # if (i == 0 and args.sparsity_normalization):
        #     limit = math.sqrt(6 / args.K_prime)
        # else:
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


def init_models(args, n_in, n_out):
    x = tf.placeholder(tf.float32, [None, n_in], name="x")
    y = tf.placeholder(tf.int32, [None], name="y")
    training = tf.placeholder(tf.bool, name="training")

    y_hot = tf.one_hot(y, n_out)
    weights, biases = _init_weights(args, n_in, n_out)

    layers = _forward(x, weights, biases, training, args)
    y_hat = layers[-1]

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
        'layers': layers,
        'train_op': train_op,
    }
