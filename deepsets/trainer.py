import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import SVG
import keras.backend as K
from keras.layers import Input, Dense, LSTM, GRU, Embedding, Activation, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import model_to_dot
from tqdm import tqdm, trange

# set parameters
num_train_examples = 150000
max_train_length = 10

num_test_examples = 10000
min_test_length = 5
max_test_length = 55
step_test_length = 5


# load mnist
def load_mnist(pack=0):
    with open(
            '/home/skottur/florainritter/code/data/mnist/mnist8m_{0}_features.bin'.
            format(pack), 'r') as f:
        v = np.fromfile(f, dtype='int32', count=1)
        D = np.asscalar(np.fromfile(f, dtype='int32', count=1))
        N = np.asscalar(np.fromfile(f, dtype='int32', count=1))
        arr = np.fromfile(f).astype('float32')
    img = np.reshape(arr, (N, D))

    with open(
            '/home/skottur/florainritter/code/data/mnist/mnist8m_{0}_labels.bin'.
            format(pack), 'r') as f:
        v = np.fromfile(f, dtype='int32', count=1)
        D = np.asscalar(np.fromfile(f, dtype='int32', count=1))
        N = np.asscalar(np.fromfile(f, dtype='int32', count=1))
        arr = np.fromfile(f).astype('float32')
    label = np.reshape(arr, (N, D))

    return img, label


# generate train data
img, label = load_mnist(0)

# Random shuffle of data
rng_state = np.random.get_state()
np.random.shuffle(img)
np.random.set_state(rng_state)
np.random.shuffle(label)

X = np.zeros((num_train_examples, max_train_length))
sum_X = np.zeros((num_train_examples))
m = 0
for i in tqdm(range(num_train_examples), desc='Generating train examples: '):
    n = np.random.randint(1, max_train_length)
    for j in range(1, n + 1):
        while label[m] == 0.:
            m += 1
        X[i, -j] = m
        sum_X[i] += label[m]
        m += 1
img = img[:m]


# generate test data
def gen_test_data(num_examples, length):
    img, label = load_mnist(np.random.randint(1, 8))
    Y = np.zeros((num_examples, length))
    sum_Y = np.zeros((num_examples))
    m = 0
    for i in range(num_examples):
        for j in range(1, length + 1):
            while label[m] == 0.:
                m += 1
            Y[i, -j] = m
            sum_Y[i] += label[m]
            m += 1
    return img[:m], Y, sum_Y


# deep set models
def get_deepset_model(images, max_length):
    input_img = Input(shape=(max_length, ))
    x = Embedding(
        images.shape[0], images.shape[1], mask_zero=True,
        trainable=False)(input_img)
    x = Dense(300, activation='tanh')(x)
    x = Dense(100, activation='tanh')(x)
    x = Dense(30, activation='tanh')(x)
    Adder = Lambda(
        lambda x: K.sum(x, axis=1),
        output_shape=(lambda shape: (shape[0], shape[2])))
    x = Adder(x)
    encoded = Dense(1)(x)
    summer = Model(input_img, encoded)
    adam = Adam(lr=1e-3, epsilon=1e-3)
    summer.compile(optimizer=adam, loss='mae')
    summer.get_layer(index=1).set_weights([images])
    return summer


# model
K.clear_session()
model = get_deepset_model(img, max_train_length)

# train
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, verbose=1, patience=20, min_lr=0.000001)

model.fit(
    X,
    sum_X,
    epochs=500,
    batch_size=128,
    shuffle=True,
    validation_split=0.0123456789,
    callbacks=[reduce_lr])

# save weights
deep_we = []
for i in [2, 3, 4, 6]:
    w = model.get_layer(index=i).get_weights()
    deep_we.append(w)

# Evaluating models
metrics = {'deepsets': {'acc': [], 'mae': [], 'mse': []}}

lengths = range(min_test_length, max_test_length, step_test_length)
for l in tqdm(lengths):
    print('Evaluating at length: ', l)
    # generate test data
    img, Y, sum_Y = gen_test_data(num_test_examples, l)

    # model
    K.clear_session()
    model = get_deepset_model(img, l)

    # load weights
    for i, idx in enumerate([2, 3, 4, 6]):
        model.get_layer(index=idx).set_weights(deep_we[i])

    # prediction
    preds = model.predict(Y, batch_size=128, verbose=1)
    metrics['deepsets']['acc'].append(
        1.0 * np.sum(np.squeeze(np.round(preds)) == sum_Y) / len(sum_Y))
    metrics['deepsets']['mae'].append(
        1.0 * np.sum(np.abs(np.squeeze(preds) - sum_Y)) / len(sum_Y))
    metrics['deepsets']['mse'].append(
        np.dot(np.squeeze(preds) - sum_Y, np.squeeze(preds) - sum_Y) /
        len(sum_Y))

    # model
    K.clear_session()

font = {'size': 14}
matplotlib.rc('font', **font)

scale = 0.5
plt.figure(figsize=(10 * scale, 8 * scale))

plt.plot(lengths, metrics['deepsets']['acc'], 'o-')
# plt.plot(lengths, metrics['lstm']['acc'], 'o-')
# plt.plot(lengths, metrics['gru']['acc'], 'o-')
plt.xlabel('Number of digits to add')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
plt.xlim(5, 50)
plt.title('Accuracy')
plt.legend(['Deepsets'], fontsize=14)
plt.tight_layout()
plt.savefig('deepsets/accuracy.png', format='png')

font = {'size': 14}
matplotlib.rc('font', **font)

scale = 0.5
plt.figure(figsize=(10 * scale, 8 * scale))

plt.plot(lengths, np.array(metrics['deepsets']['mae']) / 1e2, 'x-')
plt.plot(lengths, np.array(metrics['lstm']['mae']) / 1e2, 'x-')
plt.plot(lengths, np.array(metrics['gru']['mae']) / 1e2, 'x-')
plt.xlabel('Number of digits to add')
plt.ylabel('Mean absolute error/1e4')
plt.title('MAE')
plt.legend(['Deepsets'], fontsize=14)
plt.tight_layout()
plt.savefig('deepsets/mae.png', format='png')


font = {'size': 14}
matplotlib.rc('font', **font)

scale = 0.5
plt.figure(figsize=(10*scale, 8*scale))

plt.plot(lengths, np.array(metrics['deepsets']['mse'])/1e4, 'x-')
plt.plot(lengths, np.array(metrics['lstm']['mse'])/1e4, 'x-')
plt.plot(lengths, np.array(metrics['gru']['mse'])/1e4, 'x-')
plt.xlabel('Number of digits to add')
plt.ylabel('Mean square error/1e4')
plt.title('MSE')
plt.legend(['Deepsets'], fontsize=14)
plt.tight_layout()
plt.savefig('deepsets/mse.png', format='png')
