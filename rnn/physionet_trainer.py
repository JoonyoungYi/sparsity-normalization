""" Recurrent Neural Network.

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
"""
""" Recurrent Neural Network.

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.metrics import roc_auc_score
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

from common.utils import set_seed
from common.utils import str2bool


def RNN(x, weights, biases, timesteps, num_hidden, args):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    if args.model == 'rnn':
        rnn_cell = rnn.BasicRNNCell(num_hidden)
    elif args.model == 'lstm':
        rnn_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    else:
        raise NotImplmentedError()

    # Get lstm cell output
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def _apply_sn(batch_x, args):
    if args.sparsity_normalization:
        # print(batch_x.shape)
        l0 = np.linalg.norm(batch_x, ord=0, axis=2, keepdims=True)
        # print(l0.shape)
        l0 = np.maximum(l0, 1.0)
        # print(l0.shape)
        batch_x = batch_x * args.K_prime / l0
        # print(batch_x.shape)
        # input('')
    return batch_x


def _validate(test_x, test_y, timesteps, num_input, X, Y, auc, sess):
    # Calculate accuracy for 128 mnist test images
    test_acc = sess.run(auc, feed_dict={X: test_x, Y: test_y})
    print("Testing Accuracy:", test_acc)
    return test_acc


def _get_auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def _train(args):
    set_seed(args.seed)
    # mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    '''
    To classify images using a recurrent neural network, we consider every image
    row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
    handle 28 sequences of 28 steps for every sample.
    '''

    train_x = np.load('rnn/physionet/physionet_dataset/1_train_x.npy')
    train_y = np.load('rnn/physionet/physionet_dataset/1_train_y.npy').astype(
        np.int64).reshape([-1])

    test_x = np.load('rnn/physionet/physionet_dataset/1_val_x.npy')
    test_y = np.load('rnn/physionet/physionet_dataset/1_val_y.npy').astype(
        np.int64).reshape([-1])

    # print(train_x.shape)
    # print(test_y.shape)

    # print(train_y[train_y == 0].shape)
    # print(train_y[train_y == 1].shape)
    # input('')

    # Training Parameters
    learning_rate = args.learning_rate
    training_steps = 10000000000
    # batch_size = 5000

    # Network Parameters
    num_input = 35  # MNIST data input (img shape: 28*28)
    timesteps = 155  # timesteps
    num_hidden = args.n_hidden_unit  # hidden layer num of features
    num_classes = 2  # MNIST total classes (0-9 digits)

    train_y = np.eye(num_classes)[train_y]
    test_y = np.eye(num_classes)[test_y]

    # print(train_x[0][0])
    train_x = _apply_sn(train_x, args)
    test_x = _apply_sn(test_x, args)

    # print(train_x[0][0])
    # input('')

    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Define weights
    weights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

    logits = RNN(X, weights, biases, timesteps, num_hidden, args)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y)
    weights = tf.reduce_sum(Y * np.array([[1, 6.22]]), axis=1)
    # print(losses, weights)
    loss_op = tf.reduce_mean(weights * losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    auc = _get_auc(Y, prediction)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # Run the initializer
        sess.run(init)

        max_test_acc, max_test_iter = 0.0, 0.0
        for step in range(1, training_steps + 1):

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: train_x, Y: train_y})
            if step % args.epoch_validation == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run(
                    [loss_op, auc], feed_dict={X: train_x,
                                               Y: train_y})
                print("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc))

                test_acc = _validate(test_x, test_y, timesteps, num_input, X,
                                     Y, auc, sess)

                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    max_test_iter = step
                print(max_test_acc, max_test_iter)

                if max_test_iter + args.epoch_early_stop < step:
                    break

        print("Optimization Finished!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='RNN')
    parser.add_argument(
        '--model', type=str, default='rnn', choices=['rnn', 'lstm'], help='')
    parser.add_argument(
        '--epoch_validation',
        type=int,
        default=1,
        help='epoch unit for validation')
    # parser.add_argument(
    #     '--max_epoch', type=int, default=100000, help='Max epoch')
    parser.add_argument(
        '--n_hidden_unit',
        type=int,
        default=128,
        help='The number of hidden unit')
    parser.add_argument(
        '--K_prime',
        type=int,
        default=2.7838,
        # default=35,
        help='The number of hidden unit')
    # parser.add_argument(
    #     '--n_hidden_layer',
    #     type=int,
    #     default=1,
    #     help='The number of hidden unit')
    parser.add_argument(
        '--sparsity_normalization',
        type=str2bool,
        default=True,
        # default=False,
        help='Whether sparsity normalize input.')
    # parser.add_argument(
    #     '--use_user_vector',
    #     type=str2bool,
    #     # default=True,
    #     default=False,
    #     help='Whether use user vector.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5,
        help='learning rate for optimizer.')
    # parser.add_argument(
    #     '--dropout_rate',
    #     type=float,
    #     default=0.0,
    #     help='drop rate for dropout')
    parser.add_argument('--seed', type=int, default=1, help='the seed')
    parser.add_argument(
        '--epoch_early_stop',
        type=int,
        default=1000,
        help='the number of epochation for early stop.')

    args = parser.parse_args()
    _train(args)


if __name__ == '__main__':
    main()
