import matplotlib.pyplot as plt
import numpy as np


def _load_array(dataset, sparsity_normalization):
    with open('vision_mnist/result-epochs-{}-{}.txt'.format(
            sparsity_normalization, dataset).lower(), 'r') as f:
        rows = []
        for line in f:
            cols = [float(t) for t in line.strip().split('\t')]
            if cols[0] < 0.5:
                rows = []
            rows.append(cols)

        return np.array(rows)


dataset = 'cifar100'
# dataset = 'mnist'
# dataset = 'fashion'
result_with_sn = _load_array(dataset, True)
result_without_sn = _load_array(dataset, False)

plt.rcParams['lines.linewidth'] = 2

plt.plot(result_with_sn[:, 0], result_with_sn[:, 2])
plt.plot(result_without_sn[:, 0], result_without_sn[:, 2])
plt.plot(result_with_sn[:, 0],
         0.5 * (result_with_sn[:, 4] + result_with_sn[:, 3]))
plt.plot(result_without_sn[:, 0],
         0.5 * (result_without_sn[:, 4] + result_without_sn[:, 3]))
plt.savefig(
    'vision_mnist/result-epoch-{}.png'.format(dataset).lower(), format='png')
plt.close()
