import numpy as np
import matplotlib.pyplot as plt

file_name = "first_raw_activations"

raw_activations = np.load('results/mnist/{}.npy'.format(file_name))

count_nonzero = np.count_nonzero(raw_activations > 1e-15, 2)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, tight_layout=True)

axs[0].set_title("First Layer")
axs[0].hist(count_nonzero[0], bins=len(np.unique(count_nonzero[0])))
axs[1].set_title("Second Layer")
axs[1].hist(count_nonzero[1], bins=len(np.unique(count_nonzero[1])))

fig.savefig('results/mnist/{}_hist.png'.format(file_name), format='png')
