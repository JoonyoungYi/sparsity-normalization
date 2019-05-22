# References:
# - https://github.com/golbin/TensorFlow-Tutorials/blob/master/10%20-%20RNN/01%20-%20MNIST.py
import math

import tensorflow as tf


def _forward_rnn(x, weights_dict, training, args):
    activation_fn = tf.nn.relu

    cell = tf.nn.rnn_cell.BasicRNNCell(
        args.n_hidden_unit, activation=args.activation_fn)

    # defining initial state
    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    # 결과를 Y의 다음 형식과 바꿔야 하기 때문에
    # Y : [batch_size, n_class]
    # outputs 의 형태를 이에 맞춰 변경해야합니다.
    # outputs : [batch_size, n_step, n_hidden]
    #        -> [n_step, batch_size, n_hidden]
    #        -> [batch_size, n_hidden]
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = outputs[-1]

    if args.use_normalization_1:
        outputs /= list(range(1, args.num_steps + 1))

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs, weights_dict['W']) + weights_dict['b']


def _init_weights_dict(args):
    weights_dict = {}

    # Wxh = tf.get_variable("Wxh", [vocab_size, hidden_size], initializer=initializer)
    # Whh = tf.get_variable("Whh", [hidden_size, hidden_size], initializer=initializer)
    # Why = tf.get_variable("Why", [hidden_size, vocab_size], initializer=initializer)
    # bh  = tf.get_variable("bh", [hidden_size], initializer=initializer)
    # by  = tf.get_variable("by", [vocab_size], initializer=initializer)
    # std = 1.0
    # weights_dict[0] = {
    #     'W_xh':
    #     tf.Variable(
    #         tf.random.truncated_normal(
    #             [args.vocabulary_size, args.n_hidden_unit], 0, std)),
    #     'W_hh':
    #     tf.Variable(
    #         tf.random.truncated_normal(
    #             [args.n_hidden_unit, args.n_hidden_unit], 0, std)),
    #     'W_hy':
    #     tf.Variable(
    #         tf.random.truncated_normal(
    #             [args.n_hidden_unit, args.vocabulary_size], 0, std)),
    # }
    weights_dict['W'] = tf.Variable(
        tf.random_normal([args.n_hidden_unit, args.vocabulary_size]))
    weights_dict['b'] = tf.Variable(tf.random_normal([args.vocabulary_size]))

    # n_in, n_out = args.n_hidden_unit, args.vocabulary_size
    # limit = math.sqrt(6 / n_in)
    # weights_dict[1] = {
    #     'W': tf.Variable(tf.random_uniform([n_in, n_out], -limit, limit)),
    #     'b': tf.Variable(tf.zeros([n_out]))
    # }
    return weights_dict


def init_models(args):
    _assert_ops = []

    train_inputs = tf.placeholder(tf.int32, shape=[None, args.num_steps])
    train_labels = tf.placeholder(tf.int32, shape=[None])
    learning_rate = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)

    train_inputs_hot = tf.one_hot(train_inputs, depth=args.vocabulary_size)
    train_labels_hot = tf.one_hot(train_labels, depth=args.vocabulary_size)

    weights_dict = _init_weights_dict(args)
    y_hat = _forward_rnn(train_inputs_hot, weights_dict, training, args)

    # for w_or_b in [weight, bias]:
    #     _assert_ops.append([
    #         tf.debugging.assert_all_finite(w_or_b,
    #                                        'weight or bias is not finite!')
    #     ])

    # # Look up embeddings for inputs.
    # embeddings = tf.Variable(
    #     tf.random_uniform([args.vocabulary_size, args.embedding_size], -0.1,
    #                       0.1))  # first hidden layer의 weight
    # _assert_ops.append(
    #     tf.debugging.assert_all_finite(embeddings,
    #                                    'embeddings is not finite!'))
    # embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # input_layer = tf.reshape(embed, [-1, args.embedding_size * args.num_steps])
    # m = tf.where(
    #     tf.greater(train_inputs, -1),
    #     tf.ones_like(train_inputs), tf.zeros_like(train_inputs))
    # divider = tf.maximum(
    #     tf.cast(tf.reduce_sum(m, axis=1, keepdims=True), tf.float32), 1.0)
    # input_layer /= divider
    # _assert_ops.append(
    #     tf.debugging.assert_all_finite(input_layer,
    #                                    'input_layer is not finite!'))
    # output_layer = _forward(input_layer, weights, biases, args, training)

    costs = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=train_labels_hot, logits=y_hat)
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
        # '_embed': embed,
        # '_embeddings': embeddings,
        'train_op': train_op,
        'loss': loss,
        'training': training,
        'train_inputs': train_inputs,
        'train_labels': train_labels,
        'learning_rate': learning_rate,
    }
