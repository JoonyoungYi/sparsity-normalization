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
        layer = tf.matmul(layer, weight)
        
        if (i + 1) not in args.ignore_bias_layer_indexes:
            layer += bias

        if i < len(biases) - 1:

            if (i + 1) in args.normalize_layer_indexes:
                if args.L0_option == 'sigmoid':
                    sigmoid_L0 = tf.sigmoid(layer)
                    divider = tf.keras.backend.sum(sigmoid_L0, axis=1)

            if (i + 1) not in args.ignore_activation_layer_indexes:
                layer = activation_fn(layer)

            layers.append(layer)

            layer = tf.layers.dropout(
                layer, rate=args.drop_rate, training=training)

            if (i + 1) in args.normalize_layer_indexes:
                if args.L0_option == 'vanilla':
                    divider = tf.maximum(
                        tf.cast(tf.math.count_nonzero(layer, axis=1), tf.float32),
                        1)
                    divider = tf.stop_gradient(divider)  # make this as constant.

                divider = tf.matmul(
                    tf.expand_dims(divider, axis=1),
                    tf.ones([1, layer.shape[1]]))

                if args.drop_rate and args.drop_rate > 1e-8:
                    divider /= tf.where(training, args.drop_rate, 1.0)
                layer *= args.K_prime
                layer /= divider
        else:
            layers.append(layer)
    return layers


def _init_weights(args, n_in, n_out):
    n_units = [n_in] + [args.n_hidden_unit] * args.n_hidden_layer + [n_out]

    weights, biases = [], []
    for i, (n_in, n_out) in enumerate(zip(n_units[:-1], n_units[1:])):
        if i == 0 and args.sparsity_normalization or i in args.normalize_layer_indexes:
            limit = math.sqrt(6 / 1)
        else:
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
    activations = [tf.reduce_mean(layer) for layer in layers]
    variances = [tf.nn.moments(layer, [0, 1])[1] for layer in layers]

    raw_activations = layers

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
        'activations': activations,
        'variances': variances,
        'raw_activations': raw_activations,
        'train_op': train_op,
    }
