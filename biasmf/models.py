import tensorflow as tf
import numpy as np


def init_models(K, lambda_value, batch_manager):
    N, M = batch_manager.n_user, batch_manager.n_item
    mu = batch_manager.mu

    u = tf.placeholder(tf.int32, [None], name='u')
    i = tf.placeholder(tf.int32, [None], name='i')
    r = tf.placeholder(tf.float32, [None], name='r')

    p = tf.Variable(tf.random_normal([N, K]) / np.sqrt(N))  # p latent matrix
    q = tf.Variable(tf.random_normal([M, K]) / np.sqrt(M))  # q latent matrix
    p_lookup = tf.nn.embedding_lookup(p, u)
    q_lookup = tf.nn.embedding_lookup(q, i)

    b_u = tf.Variable(tf.zeros([N]))
    b_i = tf.Variable(tf.zeros([M]))
    b_u_lookup = tf.nn.embedding_lookup(b_u, u)
    b_i_lookup = tf.nn.embedding_lookup(b_i, i)

    b_ui = mu + tf.add(b_u_lookup, b_i_lookup)
    r_ui_hat = tf.add(b_ui, tf.reduce_sum(tf.multiply(p_lookup, q_lookup), 1))

    reconstruction_loss = tf.reduce_sum(
        tf.square(tf.subtract(r, r_ui_hat)), reduction_indices=[0])
    regularizer_loss = tf.add_n([
        tf.reduce_sum(tf.square(p)),
        tf.reduce_sum(tf.square(q)),
        tf.reduce_sum(tf.square(b_u)),
        tf.reduce_sum(tf.square(b_i)),
    ])
    loss = tf.add(reconstruction_loss, lambda_value * regularizer_loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(r, r_ui_hat))))

    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(loss, var_list=[b_u, b_i, p, q])
    return {
        'u': u,
        'i': i,
        'r': r,
        'p': p,
        'q': q,
        'b_u': b_u,
        'b_i': b_i,
        'train_op': train_op,
        'r_ui_hat': r_ui_hat,
        'rmse': rmse,
        'loss': loss,
    }
