import math

import tensorflow as tf


def _get_cosine_similarity(a, b):
    normalized_a = tf.nn.l2_normalize(a, axis=1)
    normalized_b = tf.nn.l2_normalize(b, axis=1)
    return tf.matmul(normalized_a, normalized_b, transpose_b=True)


def _get_pred_scores(a, b):
    normalized_a = tf.nn.l2_normalize(a, axis=1)
    normalized_b = tf.nn.l2_normalize(b, axis=1)
    return tf.reduce_sum(normalized_a * normalized_b, axis=1)


def init_models(args):
    train_inputs = tf.placeholder(tf.int32, shape=[None, args.context_size])
    train_labels = tf.placeholder(tf.int32, shape=[None])
    train_labels_hot = tf.one_hot(train_labels, depth=args.vocabulary_size)

    test_labels_a = tf.placeholder(tf.int32, shape=[None])
    test_labels_b = tf.placeholder(tf.int32, shape=[None])

    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([args.vocabulary_size, args.embedding_size], -1.0,
                          1.0))  # first hidden layer의 weight
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    embed = tf.where(embed > -0.1, embed, tf.zeros_like(embed))

    embed_context = tf.reduce_sum(embed, 1)
    if args.normalize_input:
        m = tf.reduce_sum(
            tf.cast(tf.greater(train_inputs, -1), tf.float32),
            axis=1,
            keepdims=True)
        embed_context /= m

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal(
            [args.vocabulary_size, args.embedding_size],
            stddev=1.0 /
            math.sqrt(args.embedding_size)))  # output layer의 weight
    nce_biases = tf.Variable(tf.zeros(
        [args.vocabulary_size]))  # output layer의 bias.

    # Construct the SGD optimizer using a learning rate of 1.0.
    if args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
    else:
        raise NotImplementedError()

    # Compute perplexity
    logits = tf.linalg.matmul(
        embed_context, nce_weights, transpose_b=True) + nce_biases
    costs = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=train_labels_hot, logits=logits)
    # costs = tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=train_labels_hot, logits=logits)
    loss = tf.reduce_sum(costs)
    train_op = optimizer.minimize(loss)

    embed_as = tf.nn.embedding_lookup(embeddings, test_labels_a)
    embed_bs = tf.nn.embedding_lookup(embeddings, test_labels_b)

    pred_scores = _get_pred_scores(embed_as, embed_bs)

    return {
        'train_op': train_op,
        'loss': loss,
        'train_inputs': train_inputs,
        'train_labels': train_labels,
        'test_labels_a': test_labels_a,
        'test_labels_b': test_labels_b,
        'pred_scores': pred_scores,
    }
