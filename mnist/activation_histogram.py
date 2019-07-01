import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='HIST')

parser.add_argument(
    '--dataset',
    type=str,
    default='mnist',
    choices=['mnist', 'fashion', 'cifar10', 'cifar100'],
    help='MNIST or Fashion MNIST or CIFAR-10 or CIFAR-100')

parser.add_argument(
    '--choose_activation',
    type=str,
    default='before',
    choices=['before', 'after'],
    help='whether plot before training activation or after training activation')

args = parser.parse_args()

file_name = "{}_raw_activations_{}_training".format(args.dataset, args.choose_activation)

raw_activations = np.load('results/mnist/{}.npy'.format(file_name))

count_nonzero = np.count_nonzero(raw_activations > 1e-15, 2)

fig, axs = plt.subplots(1, 2, tight_layout=True)

axs[0].set_title("First Hidden Layer")
axs[0].hist(count_nonzero[0], bins=len(np.unique(count_nonzero[0])))
axs[1].set_title("Second Hidden Layer")
axs[1].hist(count_nonzero[1], bins=len(np.unique(count_nonzero[1])))

fig.savefig('results/mnist/{}_hist.png'.format(file_name), format='png')
