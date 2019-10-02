import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .trainer import get_name


def _forward(x, w1, w2, b1, b2):
    x = np.matmul(x, w1) + b1
    # print(x)
    activation_f = lambda x: 1 / (1 + np.exp(-x))
    x = activation_f(x)
    x = np.matmul(x, w2) + b2
    return x


def _plot(ax, sparsity_normalization=False, masking_percentage=10):
    name = get_name(
        sparsity_normalization=sparsity_normalization,
        masking_percentage=masking_percentage,
        seed=1,
        dataset_seed=1)
    name = 'drop10-seed1-dseed1-snF'
    original_matrix = np.load(
        'auto_impute/original_matrices/blakeley/{}.npy'.format(name))
    processed_count_matrix = np.load(
        'auto_impute/masked_matrices/blakeley/{}.npy'.format(name))

    w1 = np.load('auto_impute/weights/w1/blakeley/{}.npy'.format(name))
    w2 = np.load('auto_impute/weights/w2/blakeley/{}.npy'.format(name))
    b1 = np.load('auto_impute/weights/b1/blakeley/{}.npy'.format(name))
    b2 = np.load('auto_impute/weights/b2/blakeley/{}.npy'.format(name))

    matrix_mask = processed_count_matrix.copy()
    matrix_mask[matrix_mask.nonzero()] = 1

    target_cell_idx = 19  # 13
    assert 0 <= target_cell_idx < 30

    x = processed_count_matrix[target_cell_idx]
    y = original_matrix[target_cell_idx]
    m = matrix_mask[target_cell_idx]
    known_entries = m.nonzero()[0]

    for gene_idx in range(x.shape[0]):
        if m[gene_idx] > 1e-20:
            continue
        if y[gene_idx] > 1e-20:
            break
    target_gene_idx = gene_idx
    target_activation_value = y[target_gene_idx]

    T = 100
    xs = []
    ys = []
    d = {}
    max_known_entry = int(np.sum(m))
    # max_known_entry = 100
    for n_known_entry in tqdm(range(1, max_known_entry + 1)):
        for _ in range(T):
            _known_entries = np.random.choice(known_entries, n_known_entry)

            _x = np.zeros(x.shape[0])
            for j in range(n_known_entry):
                _x[_known_entries[j]] = x[_known_entries[j]]

            if sparsity_normalization:
                _x = _x * (_x.shape[0] *
                           (1 - masking_percentage / 100)) / n_known_entry

            try:
                y_hat = _forward(_x, w1, w2, b1, b2)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                continue

            # print(_, np.sum(_x))
            # print(_, y_hat)
            # input('!')
            target = y_hat[target_gene_idx]

            xs.append(n_known_entry)
            ys.append(target)

            _hats = d.get(n_known_entry, [])
            _hats.append(target)
            d[n_known_entry] = _hats

    if sparsity_normalization:
        color = "#008EFD"
    else:
        color = "#FF343A"

    samples = np.array([item for key, item in sorted(d.items())])
    # print(samples)
    # print(samples.shape)
    # input('!')

    ax.scatter(xs, ys, s=0.5, alpha=0.05, color=color)
    ax.set_title("w/ SN" if sparsity_normalization else "w/o SN")
    results = np.array([[key, sum(item) / len(item)]
                        for key, item in sorted(d.items())])
    ax.plot(results[:, 0], results[:, 1], color=color)
    # ax.axhline(y=target_activation_value, color='#FFA336', linestyle='-')
    # ax.set_ylim([-10, 10])
    ax.set_xlim([0, 600])

    np.save('auto_impute/figures/vsp/{}.npy'.format(
        'w_sn' if sparsity_normalization else 'wo_sn'), samples)


def main():
    # percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    masking_percentage = 10
    fig, axs = plt.subplots(1, 2, figsize=(5, 2), sharey=True)
    plt.subplots_adjust(wspace=0.1)

    for j, sparsity_normalization in enumerate([False, True]):
        _plot(
            axs[j],
            sparsity_normalization=sparsity_normalization,
            masking_percentage=masking_percentage)

    axs[0].set_ylabel("prediction")

    name = "autoimpute-vsp"
    plt.savefig(
        'auto_impute/figures/vsp/{name}.png'.format(name=name),
        format='png',
        bbox_inches='tight',
        pad_inches=0)
    plt.savefig(
        'auto_impute/figures/vsp/{name}.pdf'.format(name=name),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0)
    plt.close()


if __name__ == '__main__':
    main()
