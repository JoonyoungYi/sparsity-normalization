import math

import tensorflow as tf


def __dropout(layer, training, args):
    """
        tf.layers.dropout(layer, dropout_rate, training=training)
        위 함수와 동일한 역할을 하는 함수 구현.
        단, args.dropout_with_sn이 거짓일 때에만 동일함.
        args.dropout_with_sn이 참일 경우에는,
        실제 drop된 unit 갯수를 고려해 activation 값을 뻥튀기한다.
    """
    if args.dropout_rate > 1e-8:
        r = tf.random.uniform(tf.shape(layer), minval=0., maxval=1.)
        m = tf.where(r < args.dropout_rate,
                     tf.zeros_like(layer), tf.ones_like(layer))
        train_layer = tf.multiply(layer, m)

        if args.dropout_inference == 'sn':
            m_number = tf.reduce_sum(m, axis=1, keepdims=True)
            m_number = tf.maximum(m_number, tf.ones_like(m_number))
            train_layer = train_layer * (
                tf.cast(tf.shape(layer)[1], tf.float32) / m_number)
        elif args.dropout_inference == 'vanilla':
            train_layer /= (1. - args.dropout_rate)
        else:
            raise NotImplementedError()

        return tf.cond(training, lambda: train_layer, lambda: layer)
    else:
        return layer


def __dropconnect(layer, weight, training, args):
    if args.dropconnect_rate > 1e-8:
        test_layer = tf.matmul(layer, weight)

        r = tf.random.uniform(tf.shape(weight), minval=0., maxval=1.)
        m = tf.where(r < args.dropconnect_rate,
                     tf.zeros_like(weight), tf.ones_like(weight))
        train_weight = tf.multiply(weight, m)
        train_layer = tf.matmul(layer, train_weight)

        if args.dropconnect_inference == 'simple':
            train_layer = train_layer / (1. - args.dropconnect_rate)
        elif args.dropconnect_inference == 'sn':
            m_number = tf.reduce_sum(m, axis=0, keepdims=True)
            m_number = tf.maximum(m_number, tf.ones_like(m_number))
            train_layer = train_layer * (
                tf.cast(tf.shape(m)[0], tf.float32) / m_number)
        else:
            raise NotImplementedError()

        return tf.cond(training, lambda: train_layer, lambda: test_layer)
    else:
        return tf.matmul(layer, weight)


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
        layer = __dropconnect(layer, weight, training=training, args=args)
        layer = tf.add(layer, bias)

        if i < n_layer - 1:
            layer = activation_fn(layer)
            layers.append(layer)
            layer = __dropout(layer, training=training, args=args)
        else:
            layers.append(layer)
    return layers


def _init_weights(args, n_in, n_out):
    n_units = [n_in] + [args.n_hidden_unit] * args.n_hidden_layer + [n_out]

    weights, biases = [], []
    for i, (n_in, n_out) in enumerate(zip(n_units[:-1], n_units[1:])):
        if (i == 0 and args.sparsity_normalization):
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
