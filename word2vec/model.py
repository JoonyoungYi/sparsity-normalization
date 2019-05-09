import tensorflow as tf


def init_models(args):
    train_inputs = tf.placeholder(
        tf.int32, shape=[args.batch_size, args.context_size])
    train_labels = tf.placeholder(tf.int32, shape=[args.batch_size, 1])

    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([args.vocabulary_size, args.embedding_size], -1.0,
                          1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    # take mean of embeddings of context words for context embedding
    embed_context = tf.reduce_mean(embed, 1)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal(
            [args.vocabulary_size, args.embedding_size],
            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

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
    optimizer = tf.train.GradientDescentOptimizer(1.0)
    train_op = optimizer.minimize(loss)

    return {
        'train_op': train_op,
        'loss': loss,
        'train_inputs': train_inputs,
        'train_labels': train_labels
    }
