import os

import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from tensorflow.keras.datasets import mnist

from .model import init_models

# import keras.backend as K
# from keras.layers import Input, Dense, LSTM, GRU, Embedding, Activation, Lambda
# from keras.models import Model, load_model
# from keras.optimizers import Adam
# from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
# from keras.utils.vis_utils import model_to_dot


def _load_mnist(pack=0):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train[pack * 10000:(pack + 1) * 10000], y_train[pack * 10000:(
        pack + 1) * 10000]


def _generate_test_data(num_examples, length):
    img, label = _load_mnist(np.random.randint(1, 6))
    Y = np.zeros((num_examples, length))
    M = np.zeros((num_examples, length), dtype=np.int64)
    sum_Y = np.zeros((num_examples))
    m = 0
    for i in range(num_examples):
        for j in range(1, length + 1):
            while label[m] == 0.:
                m = (m + 1) % 10000
            Y[i, -j] = m
            M[i, -j] = 1
            sum_Y[i] += label[m]
            m = (m + 1) % 10000
    return img.reshape([-1, 28 * 28]).astype(np.float32) / 255, Y, M.reshape(
        [-1, length, 1]), sum_Y


def _generate_train_data(num_train_examples, max_train_length):
    # generate train data
    img, label = _load_mnist(0)

    # Random shuffle of data
    rng_state = np.random.get_state()
    np.random.shuffle(img)
    np.random.set_state(rng_state)
    np.random.shuffle(label)

    X = -np.ones((num_train_examples, max_train_length), dtype=np.int64)
    M = np.zeros((num_train_examples, max_train_length), dtype=np.int64)
    sum_X = np.zeros((num_train_examples))
    m = 0
    for i in tqdm(
            range(num_train_examples), desc='Generating train examples: '):
        n = np.random.randint(1, max_train_length)
        for j in range(1, n + 1):
            while label[m] == 0.:
                m = (m + 1) % 10000
            X[i, -j] = m
            M[i, -j] = 1
            sum_X[i] += label[m]
            m = (m + 1) % 10000
    return img.reshape([-1, 28 * 28]).astype(np.float32) / 255, X, M.reshape(
        [-1, max_train_length, 1]), sum_X


def _sparsity_normalization(x, K_prime):
    for i in range(x.shape[0]):
        row = x[i]
        count_nonzero = max(np.where(row > 1e-8)[0].shape[0], 1)
        row *= K_prime
        row /= count_nonzero
    return x


def main():
    # set parameters
    num_train_examples = 10000
    max_train_length = 10

    # num_test_examples = 10000
    # min_test_length = 5
    # max_test_length = 55
    # step_test_length = 5

    # input('')
    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    models = init_models()
    session.run(tf.global_variables_initializer())

    test_dict = {}
    for length in range(5, 20):
        image, test_x, test_m, test_y = _generate_test_data(1000, length)
        # image = _sparsity_normalization(image, 784)
        test_dict[length] = {
            models['x_index']: test_x,
            models['image']: image,
            models['m']: test_m,
            models['y']: test_y,
        }

    image, X, M, sum_X = _generate_train_data(num_train_examples,
                                              max_train_length)
    # image = _sparsity_normalization(image, 784)
    for epoch_idx in range(1, 10000 + 1):
        _, mae = session.run(
            [
                models['train_op'],
                models['mae'],
            ],
            feed_dict={
                models['x_index']: X,
                models['m']: M,
                models['image']: image,
                models['y']: sum_X,
            })

        if epoch_idx % 100 == 0:
            print(epoch_idx, mae)
            for length in range(5, 20):
                mae = session.run(models['mae'], feed_dict=test_dict[length])
                print('>> length:', length, mae)

    # assert False
    #
    # # deep set models
    # def get_deepset_model(images, max_length):
    #     input_img = Input(shape=(max_length, ))
    #     x = Embedding(
    #         images.shape[0], images.shape[1], mask_zero=True,
    #         trainable=False)(input_img)
    #     x = Dense(300, activation='tanh')(x)
    #     x = Dense(100, activation='tanh')(x)
    #     x = Dense(30, activation='tanh')(x)
    #     Adder = Lambda(
    #         lambda x: K.sum(x, axis=1),
    #         output_shape=(lambda shape: (shape[0], shape[2])))
    #     x = Adder(x)
    #     encoded = Dense(1)(x)
    #     summer = Model(input_img, encoded)
    #     adam = Adam(lr=1e-3, epsilon=1e-3)
    #     summer.compile(optimizer=adam, loss='mae')
    #     summer.get_layer(index=1).set_weights([images])
    #     return summer
    #
    # # model
    # K.clear_session()
    # model = get_deepset_model(img, max_train_length)
    #
    # # train
    # reduce_lr = ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=0.5,
    #     verbose=1,
    #     patience=20,
    #     min_lr=0.000001)
    #
    # model.fit(
    #     X,
    #     sum_X,
    #     epochs=500,
    #     batch_size=128,
    #     shuffle=True,
    #     validation_split=0.0123456789,
    #     callbacks=[reduce_lr])
    #
    # # save weights
    # deep_we = []
    # for i in [2, 3, 4, 6]:
    #     w = model.get_layer(index=i).get_weights()
    #     deep_we.append(w)
    #
    # # Evaluating models
    # metrics = {'deepsets': {'acc': [], 'mae': [], 'mse': []}}
    #
    # lengths = range(min_test_length, max_test_length, step_test_length)
    # for l in tqdm(lengths):
    #     print('Evaluating at length: ', l)
    #     # generate test data
    #     img, Y, sum_Y = gen_test_data(num_test_examples, l)
    #
    #     # model
    #     K.clear_session()
    #     model = get_deepset_model(img, l)
    #
    #     # load weights
    #     for i, idx in enumerate([2, 3, 4, 6]):
    #         model.get_layer(index=idx).set_weights(deep_we[i])
    #
    #     # prediction
    #     preds = model.predict(Y, batch_size=128, verbose=1)
    #     metrics['deepsets']['acc'].append(
    #         1.0 * np.sum(np.squeeze(np.round(preds)) == sum_Y) / len(sum_Y))
    #     metrics['deepsets']['mae'].append(
    #         1.0 * np.sum(np.abs(np.squeeze(preds) - sum_Y)) / len(sum_Y))
    #     metrics['deepsets']['mse'].append(
    #         np.dot(np.squeeze(preds) - sum_Y, np.squeeze(preds) - sum_Y) /
    #         len(sum_Y))
    #
    #     # model
    #     K.clear_session()
    #
    # font = {'size': 14}
    # matplotlib.rc('font', **font)
    #
    # scale = 0.5
    # plt.figure(figsize=(10 * scale, 8 * scale))
    #
    # plt.plot(lengths, metrics['deepsets']['acc'], 'o-')
    # # plt.plot(lengths, metrics['lstm']['acc'], 'o-')
    # # plt.plot(lengths, metrics['gru']['acc'], 'o-')
    # plt.xlabel('Number of digits to add')
    # plt.ylabel('Accuracy')
    # plt.ylim(0, 1.1)
    # plt.xlim(5, 50)
    # plt.title('Accuracy')
    # plt.legend(['Deepsets'], fontsize=14)
    # plt.tight_layout()
    # plt.savefig('deepsets/accuracy.png', format='png')
    #
    # font = {'size': 14}
    # matplotlib.rc('font', **font)
    #
    # scale = 0.5
    # plt.figure(figsize=(10 * scale, 8 * scale))
    #
    # plt.plot(lengths, np.array(metrics['deepsets']['mae']) / 1e2, 'x-')
    # plt.plot(lengths, np.array(metrics['lstm']['mae']) / 1e2, 'x-')
    # plt.plot(lengths, np.array(metrics['gru']['mae']) / 1e2, 'x-')
    # plt.xlabel('Number of digits to add')
    # plt.ylabel('Mean absolute error/1e4')
    # plt.title('MAE')
    # plt.legend(['Deepsets'], fontsize=14)
    # plt.tight_layout()
    # plt.savefig('deepsets/mae.png', format='png')
    #
    # font = {'size': 14}
    # matplotlib.rc('font', **font)
    #
    # scale = 0.5
    # plt.figure(figsize=(10 * scale, 8 * scale))
    #
    # plt.plot(lengths, np.array(metrics['deepsets']['mse']) / 1e4, 'x-')
    # plt.plot(lengths, np.array(metrics['lstm']['mse']) / 1e4, 'x-')
    # plt.plot(lengths, np.array(metrics['gru']['mse']) / 1e4, 'x-')
    # plt.xlabel('Number of digits to add')
    # plt.ylabel('Mean square error/1e4')
    # plt.title('MSE')
    # plt.legend(['Deepsets'], fontsize=14)
    # plt.tight_layout()
    # plt.savefig('deepsets/mse.png', format='png')


if __name__ == '__main__':
    main()
