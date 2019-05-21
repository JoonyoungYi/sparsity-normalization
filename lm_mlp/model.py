import math

import tensorflow as tf


def _forward(x, weights, biases, args, training):
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
            if args.drop_rate > 1e-5:
                layer = tf.layers.dropout(
                    layer, rate=args.drop_rate, training=training)
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
    _assert_ops = []

    train_inputs = tf.placeholder(tf.int32, shape=[None, args.num_steps])
    train_labels = tf.placeholder(tf.int32, shape=[None])
    learning_rate = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)
    train_labels_hot = tf.one_hot(train_labels, depth=args.vocabulary_size)

    weights, biases = _init_weights(args)
    for w_or_b in weights + biases:
        _assert_ops.append([
            tf.debugging.assert_all_finite(w_or_b,
                                           'weight or bias is not finite!')
        ])

    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([args.vocabulary_size, args.embedding_size], -0.1,
                          0.1))  # first hidden layer의 weight
    _assert_ops.append(
        tf.debugging.assert_all_finite(embeddings,
                                       'embeddings is not finite!'))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    input_layer = tf.reshape(embed, [-1, args.embedding_size * args.num_steps])
    m = tf.where(
        tf.greater(train_inputs, -1),
        tf.ones_like(train_inputs), tf.zeros_like(train_inputs))
    divider = tf.maximum(
        tf.cast(tf.reduce_sum(m, axis=1, keepdims=True), tf.float32), 1.0)
    input_layer /= divider
    _assert_ops.append(
        tf.debugging.assert_all_finite(input_layer,
                                       'input_layer is not finite!'))
    output_layer = _forward(input_layer, weights, biases, args, training)

    costs = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=train_labels_hot, logits=output_layer)
    loss = tf.reduce_sum(costs)

    # Construct the SGD optimizer using a learning rate of 1.0.
    if args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars), args.max_grad_norm)
        train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())
    elif args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
    else:
        raise NotImplementedError()

    return {
        '_assert_ops': _assert_ops,
        '_embed': embed,
        '_embeddings': embeddings,
        'train_op': train_op,
        'loss': loss,
        'training': training,
        'train_inputs': train_inputs,
        'train_labels': train_labels,
        'learning_rate': learning_rate,
    }
