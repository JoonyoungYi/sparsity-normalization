import math
import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from common.utils import integer_list
from common.utils import set_seed


# read data potentially with missing values
def read_data(path, sep=',', val_type='f8'):
    return np.genfromtxt(path, dtype=val_type, delimiter=sep)


def _min_max_normalization(X, feature_range=(-1, 1), min_x=None, max_x=None):
    if min_x is None:
        min_x = np.nanmin(X, axis=0)
        max_x = np.nanmax(X, axis=0)

    X_std = (X - min_x) / (max_x - min_x)
    X_scaled = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    return X_scaled, min_x, max_x


def _max_normalization(train_x, test_x):
    m = np.nanmax(train_x, axis=0, keepdims=True)
    train_x = train_x / m
    test_x = test_x / m
    return train_x, test_x


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


def _mlp_gmmc(x, means, covs, p, gamma, n_distribution, weights, biases):

    gamma_ = tf.abs(gamma)
    # gamma_ = tf.cond(tf.less(gamma_[0], 1.), lambda: gamma_, lambda: tf.square(gamma_))
    covs_ = tf.abs(covs)
    p_ = tf.nn.softmax(p, axis=0)

    check_isnan = tf.is_nan(x)
    check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

    x_miss = tf.gather(x, tf.reshape(tf.where(check_isnan > 0),
                                     [-1]))  # data with missing values
    x = tf.gather(x, tf.reshape(tf.where(tf.equal(check_isnan, 0)),
                                [-1]))  # data without missing values

    # data without missing
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights[0]), biases[0]))

    # data with missing
    where_isnan = tf.is_nan(x_miss)
    where_isfinite = tf.is_finite(x_miss)
    size = tf.shape(x_miss)

    weights2 = tf.square(weights[0])

    # Collect distributions
    distributions = tf.TensorArray(dtype=x.dtype, size=n_distribution)
    q_collector = tf.TensorArray(dtype=x.dtype, size=n_distribution)

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

        layer_1_m = tf.add(tf.matmul(data_miss, weights[0]), biases[0])
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
        lambda i, c1, c2: i < n_distribution,
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

    layer = layer_1
    for weight, bias in zip(weights[1:-1], biases[1:-1]):
        layer = tf.matmul(layer, weight) + bias
        layer = tf.nn.relu(layer)
    return tf.matmul(layer, weights[-1]) + biases[-1]


def _mlp(x, weights, biases):
    layer = x
    for weight, bias in zip(weights[:-1], biases[:-1]):
        layer = tf.matmul(layer, weight) + bias
        layer = tf.nn.relu(layer)
    return tf.matmul(layer, weights[-1]) + biases[-1]


def prep_labels(x, y):
    check_isnan = tf.is_nan(x)
    check_isnan = tf.reduce_sum(tf.cast(check_isnan, tf.int32), 1)

    y_miss = tf.gather(y, tf.reshape(tf.where(check_isnan > 0), [-1]))
    y = tf.gather(y, tf.reshape(tf.where(tf.equal(check_isnan, 0)), [-1]))
    return tf.concat((y, y_miss), axis=0)


def _get_auroc(sess, x, y, n_batch, models):
    y_trues = []
    y_preds = []
    costs = []
    for start_idx in range(0, y.shape[0], n_batch):
        _x = x[start_idx:start_idx + n_batch, :]
        _y = y[start_idx:start_idx + n_batch, :]

        y_true, y_pred, cost = sess.run(
            (models['y_true'], models['y_pred'], models['cost']),
            feed_dict={
                models['x']: _x,
                models['y']: _y,
            }, )
        # print(cost)
        # input('!')
        y_trues.append(y_true)
        y_preds.append(y_pred)
        costs.append(cost)

    y_true = np.concatenate(y_trues, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)

    auroc = _auroc(y_pred.reshape([-1]), y_true.reshape([-1]))
    # input('!')
    # print(auroc)
    return auroc, sum(costs) / y.shape[0]


def _init_gmm(data, args):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    complate_data = imp.fit_transform(data)
    gmm = GaussianMixture(
        n_components=args.n_distribution,
        covariance_type='diag').fit(complate_data)
    del complate_data, imp

    gmm_weights = np.log(gmm.weights_.reshape((-1, 1)))
    gmm_means = gmm.means_
    gmm_covariances = gmm.covariances_
    del gmm
    return {
        'weights': gmm_weights,
        'means': gmm_means,
        'covariances': gmm_covariances,
    }


def _auroc(p, y):
    fpr, tpr, th = roc_curve(y, p)
    _auc = auc(fpr, tpr)
    # _roc = (fpr, tpr)
    return _auc


def _init_models(gmm, data, args):
    n_feature = args.n_feature
    if args.n_hidden_units is None:
        args.n_hidden_units = [n_feature // 2] * 3
    units = [n_feature] + args.n_hidden_units + [1]

    weights = []
    biases = []
    for n_in_unit, n_out_unit in zip(units[:-1], units[1:]):
        limit = math.sqrt(6 / n_in_unit)
        weight = tf.Variable(
            tf.random_uniform([n_in_unit, n_out_unit], -limit, limit))
        # weight = tf.Variable(
        #     tf.random.truncated_normal(
        #         [n_in_unit, n_out_unit], mean=0, stddev=limit))
        bias = tf.Variable(tf.zeros([n_out_unit]))

        weights.append(weight)
        biases.append(bias)

    # Symbols
    x = tf.placeholder(shape=[None, n_feature], dtype=tf.float32)
    y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    if args.imputation == 'gmmc':
        p = tf.Variable(initial_value=gmm['weights'], dtype=tf.float32)
        means = tf.Variable(initial_value=gmm['means'], dtype=tf.float32)
        covs = tf.Variable(initial_value=gmm['covariances'], dtype=tf.float32)

        gamma = tf.Variable(
            initial_value=tf.random_normal(shape=(1, ), mean=2, stddev=1.),
            dtype=tf.float32)

        # Construct model
        y_pred = _mlp_gmmc(x, means, covs, p, gamma, args.n_distribution,
                           weights, biases)
        y_true = prep_labels(x, y)
    else:
        y_pred = _mlp(x, weights, biases)
        y_true = y

    # Mean squared error
    costs = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y_pred, labels=y_true)
    costs = (y_true * data['class_weight'] + (1 - y_true)) * costs
    cost = tf.reduce_mean(costs)
    # print(cost)
    # input('!')

    # Gradient descent
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(cost)

    return {
        'x': x,
        'y': y,
        'cost': tf.reduce_sum(costs),
        'y_true': y_true,
        'y_pred': tf.sigmoid(y_pred),
        'train_op': train_op,
    }


def _load_data(args):
    if args.dataset in [
            'bands', 'kidney_disease', 'hepatitis', 'horse', 'mammographics',
            'pima', 'winconsin'
    ]:
        data_dir = '.datasets/gmmc/' + args.dataset
        data = read_data(os.path.join(data_dir, '_data.txt'))
        data, minx, maxx = _min_max_normalization(data, feature_range=(-1, 1))

        labels = read_data(os.path.join(data_dir, '_labels.txt'))
        assert args.dataset in ['bands']

        assert len(labels.shape) == 1
        args.n_feature = data.shape[1]
        assert labels.shape[
            0] == labels[labels == -1].shape[0] + labels[labels == 1].shape[0]
        labels = np.maximum(labels, 0)

        perm = np.random.permutation(data.shape[0])
        data = data[perm]
        labels = labels[perm]

        s = data.shape[0] // 10
        train_x = data[s:]
        train_y = labels[s:].reshape([-1, 1])
        test_x = data[:s]
        test_y = labels[:s].reshape([-1, 1])

    elif args.dataset in [
            'cardiovascular', 'fattyliver', 'hypertension', 'heartfailure',
            'diabetes'
    ]:
        base_path = '.datasets/nhis/{dataset}/'.format(dataset=args.dataset)
        train_x = np.load(base_path + 'train_x.npy')
        train_y = np.load(base_path + 'train_y.npy').reshape([-1, 1])
        test_x = np.load(base_path + 'test_x.npy')
        test_y = np.load(base_path + 'test_y.npy').reshape([-1, 1])
        args.n_feature = train_x.shape[1]

        train_x, test_x = _max_normalization(train_x, test_x)
    else:
        raise NotImplementedError()

    if args.imputation == 'gmmc':
        pass

    elif args.imputation == 'sn':
        train_m = np.sum(~np.isnan(train_x), axis=1, keepdims=True)
        k = np.mean(train_m)
        train_x = train_x * k / train_m

        test_m = np.sum(~np.isnan(test_x), axis=1, keepdims=True)
        test_x = test_x * k / test_m

        train_x[np.isnan(train_x)] = 0
        test_x[np.isnan(test_x)] = 0

    elif args.imputation == 'zero':
        train_x[np.isnan(train_x)] = 0
        test_x[np.isnan(test_x)] = 0

    else:
        raise NotImplementedError()

    class_weight = (train_y.shape[0] - np.sum(train_y)) / np.sum(train_y)

    return {
        'train_x': train_x,
        'train_y': train_y,
        'test_x': test_x,
        'test_y': test_y,
        'class_weight': class_weight,
    }


def _train(args):
    set_seed(args.dataset_seed)
    data = _load_data(args)
    # print(data['train_x'][0])
    # input('!')

    set_seed(args.seed)

    if args.imputation == 'gmmc':
        gmm = _init_gmm(data['train_x'], args)
    else:
        gmm = None
    models = _init_models(gmm, data, args)

    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    max_test_auroc = float('-inf')
    max_test_epoch_idx = -1

    for epoch_idx in range(args.max_epoch):
        for start_idx in range(0, data['train_y'].shape[0], args.n_batch):
            x_batch = data['train_x'][start_idx:start_idx + args.n_batch, :]
            y_batch = data['train_y'][start_idx:start_idx + args.n_batch, :]
            sess.run(
                models['train_op'],
                feed_dict={
                    models['x']: x_batch,
                    models['y']: y_batch,
                }, )

        train_auroc, train_cost = _get_auroc(
            sess,
            data['train_x'],
            data['train_y'],
            args.n_batch,
            models, )
        test_auroc, test_cost = _get_auroc(
            sess,
            data['test_x'],
            data['test_y'],
            args.n_batch,
            models, )

        if max_test_auroc < test_auroc:
            max_test_auroc = test_auroc
            max_test_epoch_idx = epoch_idx

        print('\n>> EPOCH:', epoch_idx)
        print('%9.6f' % train_auroc, '%9.6f' % test_auroc,
              '%9.6f' % train_cost, '%9.6f' % test_cost)
        print('%9d' % max_test_epoch_idx, '%9.6f' % max_test_auroc)

        if max_test_epoch_idx + args.early_stop_epoch < epoch_idx:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        # default='bands',
        # default='horse',
        default='winconsin',
        choices=[
            'bands',
            'kidney_disease',
            'hepatitis',
            'horse',
            'mammographics',
            'pima',
            'winconsin',
            #
            'cardiovascular',
            'fattyliver',
            'hypertension',
            'heartfailure',
            'diabetes',
        ],
        help='', )
    parser.add_argument('--seed', type=int, default=1, help='')
    parser.add_argument(
        '--imputation',
        type=str,
        # default='zero',
        default='gmmc',
        # default='sn',
        choices=['zero', 'gmmc', 'sn'],
        help='')
    parser.add_argument('--dataset_seed', type=int, default=1, help='')
    parser.add_argument(
        '--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument(
        '--n_batch',
        type=int,
        default=500000,  # 50000이면 gmmc rbf 데이터셋, nhis은 모두 full batch.
        help='Batch size. None이면 full batch.')
    parser.add_argument(
        '--n_hidden_units',
        type=integer_list,
        # default=None,  # GMMC-RBF
        # default=[4096] * 3,
        default=[50, 30],  # NHIS
        help='', )
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=10000000,
        help='Number of epochs', )
    parser.add_argument(
        '--early_stop_epoch',
        type=int,
        default=500,
        help='', )
    parser.add_argument(
        '--n_distribution',
        type=int,
        default=3,
        help='Number of distributions')
    args = parser.parse_args()

    _train(args)


if __name__ == '__main__':
    main()
    # python -m gmmc.trainer --dataset=bands
