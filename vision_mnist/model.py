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
    assert len(weights) == len(biases)
    n_layer = len(weights)
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.matmul(layer, weight)
        layer = tf.add(layer, bias)

        if i < n_layer - 1:
            layer = activation_fn(layer)
            layers.append(layer)
            # layer = tf.layers.dropout(layer, 0.2, training=training)
        else:
            layers.append(layer)
    return layers


def _init_weights(args, n_in, n_out):
    n_units = [n_in] + [args.n_hidden_unit] * args.n_hidden_layer + [n_out]

    weights, biases = [], []
    for i, (n_in, n_out) in enumerate(zip(n_units[:-1], n_units[1:])):
        # if (i == 0 and args.sparsity_normalization):
        #     limit = math.sqrt(6 / args.K_prime)
        # else:
        # stddev =

        weight = tf.Variable(
            tf.truncated_normal(
                [n_in, n_out], mean=0, stddev=math.sqrt(2 / n_in)))
        bias = tf.Variable(tf.zeros([n_out]))
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def _get_accuracy(y_hot, y_hat):
    return tf.count_nonzero(
        tf.cast(tf.equal(tf.argmax(y_hot, 1), tf.argmax(y_hat, 1)),
                tf.int64)) / tf.cast(tf.shape(y_hot)[0], tf.int64)


# def _get_conv2d_with_sn()


def init_models(args, n_in, n_out):
    x = tf.placeholder(tf.float32, [None, n_in], name="x")
    y = tf.placeholder(tf.int32, [None], name="y")
    training = tf.placeholder(tf.bool, name="training")

    # # Convolutional Layer #1
    # conv1 = tf.layers.conv2d(
    #     inputs=x,
    #     filters=32,
    #     kernel_size=[5, 5],
    #     padding="same",
    #     use_bias=False,
    #     activation=None)
    #
    # divider = tf.maximum(
    #     tf.layers.conv2d(
    #         inputs=tf.where(x > 1e-8, tf.ones_like(x), tf.zeros_like(x)),
    #         filters=1,
    #         kernel_size=[5, 5],
    #         padding="same",
    #         trainable=False,
    #         use_bias=False,
    #         kernel_initializer=tf.initializers.ones,
    #         activation=None), 1.0)
    # conv1 = conv1 * 25 / divider
    # conv1 = tf.nn.relu(conv1)
    # print(conv1)
    # conv1 += tf.Variable(tf.zeros([1, ]))
    # print(conv1)
    # input('')

    # 0.9690

    # # Pooling Layer #1
    # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    #
    # # Convolutional Layer #2 and Pooling Layer #2
    # conv2 = tf.layers.conv2d(
    #     inputs=pool1,
    #     filters=64,
    #     kernel_size=[5, 5],
    #     padding="same",
    #     activation=tf.nn.relu)
    # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    #
    # # Dense Layer
    # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # dense = tf.layers.dense(
    #     inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=training)
    #
    # # Logits Layer
    # logits = tf.layers.dense(inputs=dropout, units=10)

    y_hot = tf.one_hot(y, n_out)
    weights, biases = _init_weights(args, n_in, n_out)

    layers = _forward(x, weights, biases, training, args)
    y_hat = layers[-1]
    # y_hat = logits
    y_hat_hot = tf.argmax(y_hat, 1)

    costs = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_hot, logits=y_hat)
    loss = tf.reduce_sum(costs)
    accuracy = _get_accuracy(y_hot, y_hat)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss)

    return {
        'x': x,
        'y': y,
        'y_hat_hot': y_hat_hot,
        'training': training,
        'loss': loss,
        'accuracy': accuracy,
        'layers': layers,
        'train_op': train_op,
    }
