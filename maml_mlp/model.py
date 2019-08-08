import math

import tensorflow as tf

from .configs import *


def _init_mlp_weights():
    weights, biases = [], []

    # input -> 1st hidden layer
    limit = math.sqrt(
        3 / 1)  # kaiming init 이다. (first layer는 xavier init과 동일.)
    weight = tf.Variable(tf.random_uniform([1, 40], -limit, limit))
    bias = tf.Variable(tf.zeros([40]))
    weights.append(weight)
    biases.append(bias)

    # 1st hidden layer -> 2nd hidden layer
    limit = math.sqrt(6 / 40)  # kaiming init 이다.
    weight = tf.Variable(tf.random_uniform([40, 40], -limit, limit))
    bias = tf.Variable(tf.zeros([40]))
    weights.append(weight)
    biases.append(bias)

    # 2nd hidden layer -> output layer
    limit = math.sqrt(6 / 40)  # kaiming init 이다.
    weight = tf.Variable(tf.random_uniform([40, 1], -limit, limit))
    bias = tf.Variable(tf.zeros([1]))
    weights.append(weight)
    biases.append(bias)

    return weights, biases


def _forward(x, weights, biases):
    assert len(weights) == len(biases)

    layer = x
    for layer_idx, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.matmul(layer, weight) + bias
        if layer_idx < len(weights) - 1:
            layer = tf.nn.relu(layer)
    return layer


def _get_weight_bias_gradients(alpha_loss, weights, biases):
    gradients = tf.gradients(alpha_loss, weights + biases)
    gradients = [tf.stop_gradient(gradient) for gradient in gradients]
    weight_gradients = gradients[:len(gradients) // 2]
    bias_gradients = gradients[len(gradients) // 2:]
    return weight_gradients, bias_gradients


def init_models():
    # MLP의 weights, biases 정의
    weights, biases = _init_mlp_weights()

    # Feeding 해줄 placeholder를 정의.
    alpha_xs, alpha_ys, beta_xs, beta_ys = [], [], [], []
    for sample_batch_index in range(SAMPLE_BATCH_NUMBER):
        # k가 미정이라고 했으므로, alpha_x의 사이즈는 None이 되는 것이 맞다. 다른 것들도 마찬가지!
        alpha_x = tf.placeholder(
            tf.float32, [None, 1],
            name="alpha_x_{}".format(sample_batch_index))
        alpha_xs.append(alpha_x)

        alpha_y = tf.placeholder(
            tf.float32, [None, 1],
            name="alpha_y_{}".format(sample_batch_index))
        alpha_ys.append(alpha_y)

        beta_x = tf.placeholder(
            tf.float32, [None, 1], name="beta_x_{}".format(sample_batch_index))
        beta_xs.append(beta_x)

        beta_y = tf.placeholder(
            tf.float32, [None, 1], name="beta_y_{}".format(sample_batch_index))
        beta_ys.append(beta_y)

    # y_hat을 정의함.
    alpha_y_hats = [_forward(alpha_x, weights, biases) for alpha_x in alpha_xs]

    # pre step의 l2-loss를 정의함.
    alpha_losses = [
        tf.reduce_mean(tf.square(alpha_y_hat - alpha_y))
        for alpha_y, alpha_y_hat in zip(alpha_ys, alpha_y_hats)
    ]
    alpha_loss = tf.reduce_mean(alpha_losses)

    # loss를 기반으로 optimizer를 정의합니다.
    gradients_list = [
        _get_weight_bias_gradients(alpha_loss, weights, biases)
        for alpha_loss in alpha_losses
    ]
    beta_losses = []
    for (weight_gradients, bias_gradients), beta_x, beta_y in zip(
            gradients_list, beta_xs, beta_ys):
        beta_weights = [
            weight - LEARNING_RATE_ALPHA * weight_gradient
            for weight, weight_gradient in zip(weights, weight_gradients)
        ]
        beta_biases = [
            bias - LEARNING_RATE_ALPHA * bias_gradient
            for bias, bias_gradient in zip(biases, bias_gradients)
        ]
        beta_y_hat = _forward(beta_x, beta_weights, beta_biases)

        beta_loss = tf.reduce_mean(tf.square(beta_y_hat - beta_y))
        beta_losses.append(beta_loss)

    # beta step의 l2-loss를 정의함.
    beta_loss = tf.reduce_mean(beta_losses)
    train_op = tf.train.AdamOptimizer(LEARNING_RATE_BETA).minimize(beta_loss)

    # test할 때 retraining 하는 용도.
    test_alpha_x = tf.placeholder(tf.float32, [None, 1], name="test_alpha_x")
    test_alpha_y = tf.placeholder(tf.float32, [None, 1], name="test_alpha_y")
    test_alpha_y_hat = _forward(test_alpha_x, weights, biases)
    test_alpha_loss = tf.reduce_mean(
        tf.square(test_alpha_y_hat - test_alpha_y))
    test_op = tf.train.GradientDescentOptimizer(LEARNING_RATE_ALPHA).minimize(
        test_alpha_loss)

    test_beta_x = tf.placeholder(tf.float32, [None, 1], name="test_beta_x")
    test_beta_y = tf.placeholder(tf.float32, [None, 1], name="test_beta_y")
    test_beta_y_hat = _forward(test_beta_x, weights, biases)
    test_beta_loss = tf.reduce_mean(
        tf.abs((test_beta_y_hat - test_beta_y) / test_beta_y))
    test_beta_percent_loss = tf.reduce_mean(
        tf.abs((test_beta_y_hat - test_beta_y) / test_beta_y))

    return {
        'alpha_xs': alpha_xs,
        'alpha_ys': alpha_ys,
        'beta_xs': beta_xs,
        'beta_ys': beta_ys,
        'alpha_loss': alpha_loss,
        'beta_loss': beta_loss,
        'train_op': train_op,
        'test_alpha_x': test_alpha_x,
        'test_alpha_y': test_alpha_y,
        'test_alpha_loss': test_alpha_loss,
        'test_op': test_op,
        'test_beta_x': test_beta_x,
        'test_beta_y': test_beta_y,
        'test_beta_loss': test_beta_loss,
        'test_beta_percent_loss': test_beta_percent_loss,
    }
