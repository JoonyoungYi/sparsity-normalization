import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def _plot_mean_and_confidence_interval(ax, x, mean, ci, color=None,
                                       label=None):
    (_, caps, _) = ax.errorbar(
        x,
        mean,
        yerr=ci,
        color=color,
        barsabove=True,
        fmt='o',
        # markersize=8,
        capsize=4)
    # for cap in caps:
    #     cap.set_markeredgewidth(1)

    # plot the mean on top
    ax.plot(x, mean, label=label, color=color)


def _plot(ax, dataset):
    f = open('dropout/results/{}.tsv'.format(dataset), 'r')
    results = []
    for line in f:
        line = line.strip()
        cols = line.split("\t")
        results.append([float(c) for c in cols])
    results = np.array(results)
    assert results.shape == (8, 9)

    drop_rates = np.array([i * 0.1 for i in range(1, 10)])

    _plot_mean_and_confidence_interval(
        ax,
        drop_rates,
        results[0, :],
        results[1, :],
        label="w/o SN",
        color="#FF343A")
    _plot_mean_and_confidence_interval(
        ax,
        drop_rates,
        results[4, :],
        results[5, :],
        label="w/ SN",
        color="#008EFD")
    ax.set_title(dataset)
    # ax.xlabel('# train / # total')
    # plt.ylabel('RMSE')
    ax.set_xlim([0.05, .95])


def main():
    fig, axs = plt.subplots(1, 3, figsize=(8.1875, 2))
    plt.subplots_adjust(wspace=0.25)

    for j, dataset in enumerate(['boston', 'diabetes', 'california']):
        _plot(axs[j], dataset)
    # _plot(dataset='blakeley')
    # _plot(dataset='usoskin')
    # _plot(dataset='zeisel')
    # _plot(dataset='quake')
    # _plot(dataset='kolodziejczyk')
    # _plot(dataset='preimplantation')
    # _plot(dataset='jurkat')

    # axs[1].legend(loc='upper center', bbox_to_anchor=(-0.2, -0.3), ncol=2, edgecolor='none')
    axs[0].set_ylabel('RMSE')
    axs[1].legend(
        loc='upper center',
        bbox_to_anchor=(+0.5, -0.2),
        ncol=2,
        edgecolor='none')
    # axs[1].set_xlabel('# train / # total')

    plt.savefig(
        'dropout/results/summary.png',
        format='png',
        bbox_inches='tight',
        pad_inches=0)
    plt.savefig(
        'dropout/results/summary.pdf',
        format='pdf',
        bbox_inches='tight',
        pad_inches=0)
    plt.close()


# def sub():
#     fig, axs = plt.subplots(1, 3, figsize=(11, 2.7))
#     plt.subplots_adjust(wspace=0.25)
#
#     for j, dataset in enumerate(['quake', 'preimplantation', 'jurkat']):
#         _plot(axs[j], dataset)
#     # _plot(dataset='blakeley')
#     # _plot(dataset='usoskin')
#     # _plot(dataset='zeisel')
#     # _plot(dataset='quake')
#     # _plot(dataset='kolodziejczyk')
#     # _plot(dataset='preimplantation')
#     # _plot(dataset='jurkat')
#
#     # axs[1].legend(loc='upper center', bbox_to_anchor=(-0.2, -0.3), ncol=2, edgecolor='none')
#     axs[0].set_ylabel('RMSE')
#     axs[1].legend(
#         loc='upper center',
#         bbox_to_anchor=(+0.5, -0.2),
#         ncol=2,
#         edgecolor='none')
#     # axs[1].set_xlabel('# train / # total')
#
#     plt.savefig(
#         'auto_impute/imputation_performances/summary-sub.png',
#         format='png',
#         bbox_inches='tight',
#         pad_inches=0)
#     plt.savefig(
#         'auto_impute/imputation_performances/summary-sub.pdf',
#         format='pdf',
#         bbox_inches='tight',
#         pad_inches=0)
#     plt.close()

if __name__ == '__main__':
    main()
    # sub()
