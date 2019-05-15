import math

import tensorflow as tf


def _forward(x, weights, biases):
    # if args.activation_fn == 'relu':
    #     activation_fn = tf.nn.relu
    # elif args.activation_fn == 'tanh':
    #     activation_fn = tf.nn.tanh
    # elif args.activation_fn == 'sigmoid':
    #     activation_fn = tf.nn.sigmoid
    # else:
    #     raise NotImplementedError()

    activation_fn = tf.nn.relu

    layer = x
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.matmul(layer, weight) + bias

        if i < len(biases) - 1:
            layer = activation_fn(layer)
        #     layer = tf.layers.dropout(
        #         layer, rate=args.drop_rate, training=training)
        #
        #     if (i + 1) in args.normalize_layer_indexes:
        #         divider = tf.maximum(
        #             tf.cast(tf.math.count_nonzero(layer, axis=1), tf.float32),
        #             1)
        #         divider = tf.stop_gradient(divider)  # make this as constant.
        #         divider = tf.matmul(
        #             tf.expand_dims(divider, axis=1),
        #             tf.ones([1, layer.shape[1]]))
        #         if args.drop_rate and args.drop_rate > 1e-8:
        #             divider /= tf.where(training, args.drop_rate, 1.0)
        #         layer /= divider
        # else:
        #     layers.append(layer)
    return layer


def _init_weights(args):
    n_units = [
        args.embedding_size * args.num_steps
    ] + [args.n_hidden_unit] * args.n_hidden_layer + [args.vocabulary_size]

    weights, biases = [], []
    for i, (n_in, n_out) in enumerate(zip(n_units[:-1], n_units[1:])):
        limit = math.sqrt(6 / n_in)

        weight = tf.Variable(tf.random_uniform([n_in, n_out], -limit, limit))
        bias = tf.Variable(tf.zeros([n_out]))
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def init_models(args):
    train_inputs = tf.placeholder(tf.int32, shape=[None, args.num_steps])
    train_labels = tf.placeholder(tf.int32, shape=[None])
    train_labels_hot = tf.one_hot(train_labels, depth=args.vocabulary_size)

    weights, biases = _init_weights(args)

    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([args.vocabulary_size, args.embedding_size], -1.0,
                          1.0))  # first hidden layer의 weight
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    embed = tf.where(train_inputs == -1, tf.zeros_like(embed), embed)

    input_layer = tf.reshape(embed, [-1, args.embedding_size * args.num_steps])
    m = tf.where(train_inputs == -1,
                 tf.zeros_like(train_inputs), tf.ones_like(train_inputs))
    divider = tf.cast(tf.reduce_sum(m, axis=1, keepdims=True), tf.float32)
    input_layer /= divider
    output_layer = _forward(input_layer, weights, biases)

    costs = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=train_labels_hot, logits=output_layer)
    loss = tf.reduce_sum(costs)

    # Construct the SGD optimizer using a learning rate of 1.0.
    if args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
    else:
        raise NotImplementedError()

    train_op = optimizer.minimize(loss)

    return {
        'train_op': train_op,
        'loss': loss,
        'train_inputs': train_inputs,
        'train_labels': train_labels,
    }
