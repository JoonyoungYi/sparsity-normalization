import os

import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from common.utils import set_seed


def nr(mu, sigma):
    non_zero = tf.not_equal(sigma, 0.)
    new_sigma = tf.where(non_zero, sigma, tf.fill(tf.shape(sigma), 1e-20))
    sqrt_sigma = tf.sqrt(new_sigma)

    w = tf.div(mu, sqrt_sigma)
    nr_values = sqrt_sigma * (
        tf.div(tf.exp(tf.div(-tf.square(w), 2.)), np.sqrt(2 * np.pi)) +
        tf.multiply(tf.div(w, 2.), 1 + tf.erf(tf.div(w, np.sqrt(2)))))

    nr_values = tf.where(non_zero, nr_values, (mu + tf.abs(mu)) / 2.)
    return nr_values


def conv_first(x, means, covs, p, gamma, weights, biases, args):
    gamma_ = tf.abs(gamma)
    # gamma_ = tf.cond(tf.less(gamma_[0], 1.), lambda: gamma_, lambda: tf.square(gamma_))
    covs_ = tf.abs(covs)
    p_ = tf.nn.softmax(p, axis=0)

    check_isnan = tf.is_nan(x)
    check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

    x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0),
                                     [-1]))  # data_rbfn with missing values
    x = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)),
                                [-1]))  # data_rbfn without missing values

    # data_rbfn without missing
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['0']), biases['0']))

    # data_rbfn with missing
    where_isnan = tf.is_nan(x_miss)
    where_isfinite = tf.is_finite(x_miss)
    size = tf.shape(x_miss)

    weights2 = tf.square(weights['0'])

    # Collect distributions
    distributions = tf.TensorArray(dtype=x.dtype, size=args.n_distribution)
    q_collector = tf.TensorArray(dtype=x.dtype, size=args.n_distribution)

    # Each loop iteration calculates all per component
    def calculate_component(i, collect1, collect2):
        data_miss = tf.where(where_isnan,
                             tf.reshape(
                                 tf.tile(means[i, :], [size[0]]),
                                 [-1, size[1]]), x_miss)
        miss_cov = tf.where(where_isnan,
                            tf.reshape(
                                tf.tile(covs_[i, :], [size[0]]), [-1,
                                                                  size[1]]),
                            tf.zeros([size[0], size[1]]))

        layer_1_m = tf.add(tf.matmul(data_miss, weights['0']), biases['0'])
        layer_1_m = nr(layer_1_m, tf.matmul(miss_cov, weights2))

        norm = tf.subtract(data_miss, means[i, :])
        norm = tf.square(norm)
        q = tf.where(where_isfinite,
                     tf.reshape(
                         tf.tile(tf.add(gamma_, covs_[i, :]), [size[0]]),
                         [-1, size[1]]), tf.ones_like(x_miss))
        norm = tf.div(norm, q)
        norm = tf.reduce_sum(norm, axis=1)

        q = tf.log(q)
        q = tf.reduce_sum(q, axis=1)

        q = tf.add(q, norm)

        norm = tf.cast(
            tf.reduce_sum(tf.cast(where_isfinite, tf.int32), axis=1),
            tf.float32)
        norm = tf.multiply(norm, tf.log(2 * np.pi))

        q = tf.add(q, norm)
        q = -0.5 * q

        return i + 1, collect1.write(i, layer_1_m), collect2.write(i, q)

    i = tf.constant(0)
    _, final_distributions, final_q = tf.while_loop(
        lambda i, c1, c2: i < args.n_distribution,
        calculate_component,
        loop_vars=(i, distributions, q_collector),
        swap_memory=True,
        parallel_iterations=1)

    distrib = final_distributions.stack()
    log_q = final_q.stack()

    log_q = tf.add(log_q, tf.log(p_))
    r = tf.nn.softmax(log_q, axis=0)

    layer_1_miss = tf.multiply(distrib, r[:, :, tf.newaxis])
    layer_1_miss = tf.reduce_sum(layer_1_miss, axis=0)

    # join layer for data_rbfn with missing values with layer for data_rbfn without missing values
    layer_1 = tf.concat((layer_1, layer_1_miss), axis=0)
    return layer_1


# Building the encoder
def encoder(x, means, covs, p, gamma, weights, biases, args):
    return conv_first(x, means, covs, p, gamma, weights, biases, args)


# Building the decoder
def decoder(x, weights, biases):
    return tf.add(tf.matmul(x, weights['1']), biases['1'])


def prep_x(x, y):
    check_isnan = tf.is_nan(x)
    check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

    y_miss = tf.gather(y, tf.reshape(tf.where(check_isnan > 0), [-1]))
    y_valid = tf.gather(y, tf.reshape(
        tf.where(tf.equal(check_isnan, 0)), [-1]))
    return tf.concat((y_valid, y_miss), axis=0)


def _load_data(args):
    mnist = tf.keras.datasets.mnist
    (train_x, _), (_, _) = mnist.load_data()
    train_x = 1 - train_x / 255.0
    train_x = train_x[:1000]
    del mnist
    train_x = train_x.reshape([-1, 784])

    train_x[np.random.random(train_x.shape) < 0.9] = np.nan
    return train_x


def _init_gmm(data_train, args):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imp.fit_transform(data_train)

    gmm = GaussianMixture(
        n_components=args.n_distribution, covariance_type='diag').fit(data)

    p = tf.Variable(
        initial_value=np.log(gmm.weights_.reshape((-1, 1))), dtype=tf.float32)
    means = tf.Variable(initial_value=gmm.means_, dtype=tf.float32)
    covs = tf.Variable(initial_value=gmm.covariances_, dtype=tf.float32)
    gamma = tf.Variable(
        initial_value=tf.random_normal(shape=(1, ), mean=1., stddev=1.),
        dtype=tf.float32)
    del data, gmm
    return {
        'means': means,
        'covs': covs,
        'p': p,
        'gamma': gamma,
    }


def _init_models(gmm, args):
    # init tensor
    x = tf.placeholder("float", [None, args.n_input_unit])
    y = tf.placeholder("float", [None, args.n_input_unit])
    initializer = tf.contrib.layers.variance_scaling_initializer()
    weights = {
        '0': tf.Variable(initializer([args.n_input_unit, args.n_hidden_unit])),
        '1': tf.Variable(initializer([args.n_hidden_unit, args.n_input_unit])),
    }
    biases = {
        '0': tf.Variable(tf.random_normal([args.n_hidden_unit])),
        '1': tf.Variable(tf.random_normal([args.n_input_unit])),
    }

    # Construct model
    h = encoder(
        x,
        gmm['means'],
        gmm['covs'],
        gmm['p'],
        gmm['gamma'],
        weights,
        biases,
        args=args)
    y_pred = decoder(h, weights, biases)
    y_true = prep_x(x, y)  # Targets (Labels) are the input data_rbfn.

    where_isnan = tf.is_nan(y_true)
    y_pred = tf.where(where_isnan, tf.zeros_like(y_pred), y_pred)
    y_true = tf.where(where_isnan, tf.zeros_like(y_true), y_true)

    # Define loss and optimizer, minimize the squared error
    loss = tf.reduce_sum(tf.pow(y_true - y_pred, 2))
    rmse = tf.sqrt(loss / tf.reduce_sum(
        tf.where(where_isnan, tf.zeros_like(y_pred), tf.ones_like(y_pred))))
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss)
    return {
        'x': x,
        'y': y,
        'loss': loss,
        'rmse': rmse,
        'train_op': train_op,
    }


def _train(args):
    set_seed(args.seed)

    train_x = _load_data(args)
    gmm = _init_gmm(train_x, args)

    models = _init_models(gmm, args)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_idx in range(1, args.max_epoch + 1):
            print('>> EPOCH:', epoch_idx)

            _, train_rmse = sess.run(
                (models['train_op'], models['rmse']),
                feed_dict={
                    models['x']: train_x,
                    models['y']: train_x,
                })
            print('%.6f' % train_rmse)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='AutoRecGMMC')
    parser.add_argument('--n_distribution', type=int, default=5, help='')
    parser.add_argument('--n_batch', type=int, default=2000, help='')
    parser.add_argument('--max_epoch', type=int, default=100000, help='')
    parser.add_argument('--n_hidden_unit', type=int, default=500, help='')
    parser.add_argument('--n_input_unit', type=int, default=784, help='')
    parser.add_argument('--seed', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='')
    args = parser.parse_args()
    _train(args)


if __name__ == '__main__':
    main()
