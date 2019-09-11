from __future__ import division, print_function, absolute_import
import re
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from common.utils import str2bool
from common.utils import set_seed
from common.utils import init_file
from common.utils import append_text_results

CODE_NAME = 'auto_impute'


def get_name(sparsity_normalization, masking_percentage, seed):
    return 'drop{masking_percentage}-seed{seed}-sn{sparsity_normalization}'.format(
        sparsity_normalization='T' if sparsity_normalization else 'F',
        masking_percentage=masking_percentage,
        seed=seed)


def _evaluate(sess, X, processed_count_matrix, matrix_mask, y_pred,
              store_for_future, indices, mask):
    imputed_count_matrix = sess.run(
        [y_pred], feed_dict={
            X: processed_count_matrix,
            mask: matrix_mask,
        })

    predictions = []
    for idx, value in enumerate(store_for_future):
        prediction = imputed_count_matrix[0][indices[0][idx], indices[1][idx]]
        predictions.append(prediction)

    store_for_future = np.array(store_for_future)
    predictions = np.array(predictions)
    last_mae = mean_absolute_error(store_for_future, predictions)
    last_rmse = mean_squared_error(store_for_future, predictions)**0.5
    last_nmse = (np.linalg.norm(store_for_future - predictions) /
                 np.linalg.norm(store_for_future))

    return last_mae, last_rmse, last_nmse, imputed_count_matrix


def _train(args):
    set_seed(args.dataset_seed)
    matplotlib.use('Agg')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # started clock
    start_time = datetime.now()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # reading dataset
    try:
        extn = args.data.split('.')[1]
        if (extn == 'mat'):
            print("[!data read] Reading from data/" + args.data)
            processed_count_matrix = scipy.io.mmread("auto_impute/data/" +
                                                     args.data)
            processed_count_matrix = processed_count_matrix.toarray()
            processed_count_matrix = np.array(processed_count_matrix)
        else:
            print("[!data read] Reading from data/" + args.data)
            with open("auto_impute/data/" + args.data) as f:
                ncols = len(f.readline().split(','))
            processed_count_matrix = np.loadtxt(
                open("auto_impute/data/" + args.data, "rb"),
                delimiter=",",
                skiprows=1,
                usecols=range(1, ncols + 1))
    except:
        from traceback import format_exc
        print(format_exc())
        print(
            "[!data read] Please make sure that your processed dataset is in data/ in .mat or .csv format and you have entered the filename as data parameter. e.g. blakeley.csv"
        )
        exit()

    masking_percentage = args.masking_percentage
    masking_percentage = masking_percentage / 100.0

    name = get_name(
        sparsity_normalization=args.sparsity_normalization,
        masking_percentage=int(masking_percentage * 100),
        seed=args.seed)
    np.save(
        'auto_impute/original_matrices/{name}.npy'.format(name=name),
        processed_count_matrix)

    dataset = args.dataset
    idxi, idxj = np.nonzero(processed_count_matrix)
    ix = np.random.choice(
        len(idxi),
        int(np.floor(masking_percentage * len(idxi))),
        replace=False)
    store_for_future = processed_count_matrix[idxi[ix], idxj[ix]]
    indices = idxi[ix], idxj[ix]

    processed_count_matrix[idxi[ix], idxj[ix]] = 0  # making masks 0
    matrix_mask = processed_count_matrix.copy()
    matrix_mask[matrix_mask.nonzero()] = 1

    np.save(
        'auto_impute/masked_matrices/{name}.npy'.format(name=name),
        processed_count_matrix)

    init_file(CODE_NAME, args.dataset, 'epochs', name)

    set_seed(args.seed)
    # finding number of genes and cells.
    genes = processed_count_matrix.shape[1]
    cells = processed_count_matrix.shape[0]
    print("[info] Genes : {0}, Cells : {1}".format(genes, cells))

    K_prime = np.sum(matrix_mask) / cells

    # placeholder definitions
    X = tf.placeholder("float32", [None, genes], name="X")
    mask = tf.placeholder("float32", [None, genes], name="mask")

    matrix_mask = processed_count_matrix.copy()
    matrix_mask[matrix_mask.nonzero()] = 1

    print("[info] Hyper-parameters")
    print("\t Hidden Units : " + str(args.hidden_units))
    print("\t Lambda : {0}".format(args.lambda_val))
    print("\t Threshold : " + str(args.threshold))
    print("\t Iterations : " + str(args.max_epoch))
    print("\t Initial learning rate : " + str(args.learning_rate))

    # model definition
    weights = {
        'encoder_h': tf.Variable(tf.random_normal([genes, args.hidden_units])),
        'decoder_h': tf.Variable(tf.random_normal([args.hidden_units, genes])),
    }
    biases = {
        'encoder_b': tf.Variable(tf.random_normal([args.hidden_units])),
        'decoder_b': tf.Variable(tf.random_normal([genes])),
    }

    def encoder(x):
        layer_1 = tf.nn.sigmoid(
            tf.add(tf.matmul(x, weights['encoder_h']), biases['encoder_b']))
        return layer_1

    def decoder(x):
        layer_1 = tf.add(
            tf.matmul(x, weights['decoder_h']), biases['decoder_b'])
        return layer_1

    if args.sparsity_normalization:
        encoder_op = encoder(
            X * K_prime / tf.maximum(tf.reduce_sum(mask, keepdims=True), 1.0))
    else:
        encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    # loss definition
    y_pred = decoder_op
    y_true = X
    rmse_loss = tf.pow(tf.norm(y_true - y_pred * mask), 2)
    regularization = tf.multiply(
        tf.constant(args.lambda_val / 2.0, dtype="float32"),
        tf.add(
            tf.pow(tf.norm(weights['decoder_h']), 2),
            tf.pow(tf.norm(weights['encoder_h']), 2)))
    loss = tf.add(tf.reduce_mean(rmse_loss), regularization)
    optimizer = tf.train.RMSPropOptimizer(args.learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    sess.run(init)

    prev_loss = 0
    min_test_rmse = float('inf')
    min_test_epoch_idx = -1
    for epoch_idx in range(1, args.max_epoch + 1):
        _, loss = sess.run(
            [optimizer, rmse_loss],
            feed_dict={
                X: processed_count_matrix,
                mask: matrix_mask,
            })
        lpentry = loss / cells
        change = abs(prev_loss - lpentry)
        if (change <= args.threshold):
            print("Reached the threshold value.")
            break

        test_mae, test_rmse, test_nmse, _ = _evaluate(
            sess, X, processed_count_matrix, matrix_mask, y_pred,
            store_for_future, indices, mask)

        if test_rmse < min_test_rmse:
            min_test_rmse = test_rmse
            min_test_epoch_idx = epoch_idx

        append_text_results(CODE_NAME, args.dataset, 'epochs', name, '\t'.join([
            str(r)
            for r in
            [epoch_idx, loss, lpentry, change, test_mae, test_rmse, test_nmse]
        ]))

        prev_loss = lpentry
        if (args.debug):
            print('[Epoch %i] MeanL: %.4f, Test RMSE: %.4f(min: %.4f, %4d)' %
                  (epoch_idx, lpentry, test_rmse, min_test_rmse,
                   min_test_epoch_idx))

    last_mae, last_rmse, last_nmse, imputed_count_matrix = _evaluate(
        sess, X, processed_count_matrix, matrix_mask, y_pred, store_for_future,
        indices, mask)
    np.save(
        'auto_impute/imputed_matrices/{name}.npy'.format(name=name),
        imputed_count_matrix)
    w1, w2, b1, b2 = sess.run([
        weights['encoder_h'], weights['decoder_h'], biases['encoder_b'],
        biases['decoder_b']
    ])
    np.save('auto_impute/weights/w1/{name}.npy'.format(name=name), w1)
    np.save('auto_impute/weights/w2/{name}.npy'.format(name=name), w2)
    np.save('auto_impute/weights/b1/{name}.npy'.format(name=name), b1)
    np.save('auto_impute/weights/b2/{name}.npy'.format(name=name), b2)
    sess.close()

    print("<------------------------------------Statistics for " + args.data.
          split('.')[0] + " dataset------------------------------------>")
    print("MAE = {0}".format(last_mae))
    print("RMSE = {0}".format(last_rmse))
    print("NMSE = {0}".format(last_nmse))

    finish_time = datetime.now()
    print("[info] Total time taken = {0}".format(finish_time - start_time))

    results = []
    results.append(datetime.now())
    results.append(args.sparsity_normalization)
    results.append(args.threshold)
    results.append(args.max_epoch)
    results.append(args.learning_rate)
    results.append(args.seed)
    results.append(last_mae)
    results.append(last_rmse)
    results.append(last_nmse)
    file_name = re.sub('-seed\d+', '', name)
    file_name = re.sub('-sn[TF]$', '', file_name)
    append_text_results(CODE_NAME, args.dataset, 'results', file_name,
                        '\t'.join([str(r) for r in results]))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='auto_impute')

    # Print debug statements
    parser.add_argument(
        '--debug',
        type=bool,
        default=True,
        nargs='+',
        help="Want debug statements.")

    # Hyper-parameters
    parser.add_argument(
        '--hidden_units',
        type=int,
        default=2000,
        help="Size of hidden layer or latent space dimensions.")
    parser.add_argument(
        '--lambda_val',
        type=int,
        default=1,
        help=
        "Regularization coefficient, to control the contribution of regularization term in the cost function."
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help="Initial value of learning rate.")
    parser.add_argument(
        '--max_epoch',
        type=int,
        # default=0,
        default=7000,
        # default=100000,
        help="Number of max_epoch to train the model for.")
    parser.add_argument(
        '--threshold',
        type=int,
        # default=0.0001, # original
        # 바꾼 이유 원래 threshold는 underfit 되는 것 같아 늘렸더니 저자보다 좋은 성능!
        # default=0.00001,
        default=-1,
        help=
        "To stop gradient descent after the change in loss function value in consecutive max_epoch is less than the threshold, implying convergence."
    )

    # Data
    parser.add_argument(
        '--dataset',
        type=str,
        default='blakeley',
        help=
        "Dataset to run the script on. In the paper we choose from : ['blakeley.csv', 'jurkat-293T.mat', 'kolodziejczyk.csv', 'PBMC.csv', 'preimplantation.mat', 'quake.csv', 'usoskin.csv', 'zeisel.csv']"
    )

    # Run the masked matrix recovery test
    parser.add_argument(
        '--sparsity_normalization',
        type=str2bool,
        # default=True,
        default=False,
        help="")
    parser.add_argument(
        '--masking_percentage',
        type=float,
        default=10,
        # nargs='+',
        help="Percentage of masking required. Like 10, 20, 12.5 etc")

    parser.add_argument('--seed', type=int, default=1, help="")
    parser.add_argument(
        '--dataset_seed', type=int, default=1, help="seed for test set split")

    args = parser.parse_args()
    args.data = args.dataset + '.csv'
    _train(args)


if __name__ == '__main__':
    main()
