import math

import tensorflow as tf


def init_models(args):
    train_inputs = tf.placeholder(tf.int32, shape=[None, args.context_size])
    train_labels = tf.placeholder(tf.int32, shape=[None, 1])

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

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            inputs=embed_context,
            labels=train_labels,
            num_sampled=args.num_sampled,
            num_classes=args.vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    if args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
    else:
        raise NotImplementedError()

    train_op = optimizer.minimize(loss)

    # Compute perplexity
    logits = tf.linalg.matmul(
        embed_context, nce_weights,
        transpose_b=True) + tf.transpose(nce_biases)
    costs = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.squeeze(train_labels), logits=logits)
    cross_entropy_loss = tf.reduce_sum(costs)

    return {
        'train_op': train_op,
        'loss': loss,
        'train_inputs': train_inputs,
        'train_labels': train_labels,
        'cross_entropy_loss': cross_entropy_loss,
    }
