import argparse
import random
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import keras
from keras.optimizers import Adam
from common.utils import str2bool
from common.utils import save_results_to_file
from .model import init_model
from .dataset import init_data
import time

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
    '--use_modified',
    type=str2bool,
    default=False)

parser.add_argument(
    '--seed', type=int, default=1)
parser.add_argument(
    '--model',
    type=str,
    default='mlp',
    choices=['mlp', 'lstm'])
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
    '--layer_norm_lstm',
    type=str2bool,
    default=False,
    help='Whether to add layer_norm in lstm model')
parser.add_argument(
    '--normalize_input',
    type=str2bool,
    default=False,
    help='Whether normalize input layer.')

parser.add_argument(
    '--epochs',
    type=int,
    default=1000000)
parser.add_argument(
    '--epochs_early_stop',
    type=int,
    default=1000)
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
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          verbose=args.verbose,
          class_weight=class_weights,
          epochs=args.epochs,
          batch_size=args.batch_size,
          callbacks=[early_stopping_callback]) # early stopping callback

# model evaluation
model.compile(loss='binary_crossentropy', optimizer=Adam(args.learning_rate), metrics=['accuracy', auc])
scores = model.evaluate(X_test, y_test, verbose=0, batch_size=args.batch_size)

#print("\nResult of {}".format(disease))
#print("Best AUC: {}\n".format(scores[2]))

endTime = time.time() - startTime
endTime = int(endTime/60)

# save results to file
results = []
results.append(args.dropout)
results.append(args.batch_norm_mlp)
results.append(args.layer_norm_lstm)
results.append(args.normalize_input)
results.append(disease)
results.append(args.seed)
results.append(scores[2])
results.append(args.epochs_early_stop)
results.append(endTime)
save_results_to_file('aitrics', args.model, results)
