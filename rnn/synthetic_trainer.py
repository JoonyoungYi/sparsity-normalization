import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


def RNN(x, weights, biases, timesteps, num_hidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def _apply_sn(batch_x):
    sparsity_normalization = True
    if sparsity_normalization:
        # print(batch_x.shape)
        l0 = np.linalg.norm(batch_x, ord=0, axis=2, keepdims=True)
        # print(l0.shape)
        # print(np.mean(l0))
        # input('')
        l0 = np.maximum(l0, 1.0)
        # print(l0.shape)
        batch_x = batch_x * 16.0 / l0
        # print(batch_x.shape)
        # input('')
    return batch_x


# def _validate(mnist, timesteps, num_input, X, Y, accuracy, sess):
#     # Calculate accuracy for 128 mnist test images
#     test_len = 5000
#     test_data = mnist.test.images[:test_len].reshape((-1, timesteps,
#                                                       num_input))
#     test_data = _apply_sn(test_data)
#     test_label = mnist.test.labels[:test_len]
#     test_acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
#     print("Testing Accuracy:", test_acc)
#     return test_acc


def _get_batch(batch_size, timesteps, num_input, num_classes):
    batch_x = np.random.uniform(
        low=-.9, high=1.0, size=[batch_size, timesteps, num_input])

    batch_y = np.random.randint(2, size=(batch_size))
    batch_x[batch_y == 0] *= -1

    batch_x[np.random.rand(*batch_x.shape) < 0.5] = 0

    return batch_x, np.eye(num_classes)[batch_y]


def _train():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Training Parameters
    learning_rate = 0.01
    training_steps = 10000000000
    batch_size = 5000
    display_step = 100

    # Network Parameters
    num_input = 32  # MNIST data input (img shape: 28*28)
    timesteps = 4  # timesteps
    num_hidden = 8  # hidden layer num of features
    num_classes = 2  # MNIST total classes (0-9 digits)

    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Define weights
    weights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

    logits = RNN(X, weights, biases, timesteps, num_hidden)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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

            batch_x, batch_y = _get_batch(batch_size, timesteps, num_input,
                                          num_classes)
            batch_x = _apply_sn(batch_x)

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run(
                    [loss_op, accuracy], feed_dict={X: batch_x,
                                                    Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc))

                # test_acc = _validate(mnist, timesteps, num_input, X, Y,
                #                      accuracy, sess)
                #
                # if test_acc > max_test_acc:
                #     max_test_acc = test_acc
                #     max_test_iter = step
                # print(max_test_acc, max_test_iter)

        print("Optimization Finished!")


def main():
    _train()


if __name__ == '__main__':
    main()
