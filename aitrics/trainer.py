import argparse
import random
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import keras
from keras.optimizers import Adam
from common.utils import str2bool
from common.utils import save_results_to_file
from .model import init_model
from .dataset import init_data

startTime = time.time()

parser = argparse.ArgumentParser(description='AITRICS')

parser.add_argument(
    '--data_path',
    type=str,
    default='aitrics/data',
    help='e.g. [data_path]/cardiovascular1')
parser.add_argument(
    '--choose_dataset',
    type=int,
    default=1)
parser.add_argument(
    '--screen_print',
    type=str2bool,
    default=False)
parser.add_argument(
    '--file_print',
    type=str2bool,
    default=True)
parser.add_argument(
    '--seed',
    type=int,
    default=1)

parser.add_argument(
    '--model',
    type=str,
    default='mlp',
    choices=['mlp', 'lstm', '1dcnn'])
parser.add_argument(
    '--dropout',
    type=str2bool,
    default=False,
    help='Whether to add dropout')
parser.add_argument(
    '--batch_norm_mlp',
    type=str2bool,
    default=False,
    help='Whether to add batch_norm in mlp model')
parser.add_argument(
    '--layer_norm_mlp',
    type=str2bool,
    default=False,
    help='Whether to add layer_norm in mlp model')
parser.add_argument(
    '--layer_norm_lstm',
    type=str2bool,
    default=False,
    help='Whether to add layer_norm in lstm model')

parser.add_argument(
    '--conv_kernel_size',
    type=int,
    default=2,
    choices=[2, 3])
parser.add_argument(
    '--conv_layer_num',
    type=int,
    default=1,
    choices=[1, 2])
parser.add_argument(
    '--conv_padding',
    type=str,
    default='valid',
    choices=['valid', 'causal', 'same'])
parser.add_argument(
    '--conv_act',
    type=str,
    default='relu',
    choices=['relu', 'tanh'])
parser.add_argument(
    '--conv_filters',
    type=int,
    default=17)

parser.add_argument(
    '--sparsity_normalization',
    type=str2bool,
    default=False,
    help='Whether to apply sparsity normalization')
parser.add_argument(
    '--K_prime',
    type=float,
    default=1.0,
    help='K_prime in Algorithm 1 of the paper')

parser.add_argument(
    '--epochs',
    type=int,
    default=1000000)
parser.add_argument(
    '--epochs_early_stop',
    type=int,
    default=500)
parser.add_argument(
    '--batch_size',
    type=int,
    default=500000)
parser.add_argument(
    '--learning_rate',
    type=float,
    default=1e-2)

parser.add_argument(
    '--verbose',
    type=int,
    default=1,
    choices=[0, 1],
    help='0: silent, 1: progress bar')
parser.add_argument(
    '--gpu_memory_fraction',
    type=float,
    default=0.3)

args = parser.parse_args()

# set gpu memory fraction
if args.gpu_memory_fraction < 0.9:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    keras.backend.tensorflow_backend.set_session(sess)

# manual_seed setting for reproduction
random.seed(args.seed)
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

# get data
X_train, y_train, X_test, y_test, class_weights, disease = init_data(args)

# construct model
model, early_stopping_callback = init_model(args)

# model training
model.compile(loss='binary_crossentropy', optimizer=Adam(args.learning_rate), metrics=['accuracy', auc])
hist = model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          verbose=args.verbose,
          class_weight=class_weights,
          epochs=args.epochs,
          batch_size=args.batch_size,
          callbacks=[early_stopping_callback]) # early stopping callback

# model evaluation
scores = model.evaluate(X_test, y_test, verbose=0, batch_size=args.batch_size)

if args.screen_print:
    print("\nResult of {}".format(disease))
    print("Best AUC: {}\n".format(scores[2]))

elapsedTime = time.time() - startTime
elapsedTime = int(elapsedTime/60)

if args.file_print:
    # save results to file
    results = []
    results.append(disease)
    results.append(args.seed)
    results.append(args.learning_rate)

    results.append(args.model)
    results.append(args.dropout)

    results.append(args.conv_kernel_size)
    results.append(args.conv_layer_num)
    results.append(args.conv_padding)
    results.append(args.conv_act)
    results.append(args.conv_filters)

    results.append(args.epochs_early_stop)
    results.append(hist.epoch[-1])

    results.append(scores[2])
    results.append(elapsedTime)
    save_results_to_file('aitrics', args.model, results)
