import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def _plot_mean_and_confidence_interval(ax,
                                       x,
                                       mean,
                                       err,
                                       color=None,
                                       label=None):
    (_, caps, _) = ax.errorbar(
        x,
        mean,
        yerr=err,
        color=color,
        barsabove=True,
        fmt='o',
        # markersize=8,
        capsize=4)
    # for cap in caps:
    #     cap.set_markeredgewidth(1)

    # plot the mean on top
    ax.plot(x, mean, label=label, color=color)


def main():
    f = open('made/results/lr5e-3.tsv', 'r')
    results = []
    for line in f:
        line = line.strip()
        cols = line.split("\t")
        results.append([float(c) for c in cols])
    results = np.array(results)
    assert results.shape == (6, 8)

    fig, axs = plt.subplots(1, 1, figsize=(3, 2.5))
    plt.subplots_adjust(wspace=0.25)

    ax = axs

    masks = np.array([i for i in range(1, 9)])
    _plot_mean_and_confidence_interval(
        ax,
        masks,
        results[0, :],
        results[1, :],
        label="w/o SN",
        color="#FF343A")
    _plot_mean_and_confidence_interval(
        ax,
        masks,
        results[4, :],
        results[5, :],
        label="w/ SN",
        color="#008EFD")
    _plot_mean_and_confidence_interval(
        ax,
        masks,
        results[2, :],
        results[3, :],
        label="w/ SN (all)",
        color="#0D0EFD")

    # ax.set_title(dataset)
    ax.set_ylim([82, 90])

    ax.set_ylabel('NLL')
    ax.set_xlabel('# mask')
    ax.legend(
        # loc='upper center',
        # bbox_to_anchor=(+0.0, -0.0),
        ncol=1)

    plt.savefig(
        'made/performances/summary.png',
        format='png',
        bbox_inches='tight',
        pad_inches=0)
    plt.savefig(
        'made/performances/summary.pdf',
        format='pdf',
        bbox_inches='tight',
        pad_inches=0)
    plt.close()


def appendix():
    f = open('made/results/lr1e-3.tsv', 'r')
    results = []
    for line in f:
        line = line.strip()
        cols = line.split("\t")
        results.append([float(c) for c in cols])
    results = np.array(results)
    assert results.shape == (4, 8)

    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    plt.subplots_adjust(wspace=0.25)

    ax = axs

    masks = np.array([i for i in range(1, 9)])
    _plot_mean_and_confidence_interval(
        ax,
        masks,
        results[0, :],
        results[1, :],
        label="w/o SN",
        color="#FF343A")
    _plot_mean_and_confidence_interval(
        ax,
        masks,
        results[2, :],
        results[3, :],
        label="w/ SN",
        color="#008EFD")

    # ax.set_title(dataset)
    ax.set_ylim([78, 90])

    ax.set_ylabel('NLL')
    ax.set_xlabel('# mask')
    ax.legend(
        # loc='upper center',
        # bbox_to_anchor=(+0.0, -0.0),
        ncol=1)

    plt.savefig(
        'made/performances/summary-appendix.png',
        format='png',
        bbox_inches='tight',
        pad_inches=0)
    plt.savefig(
        'made/performances/summary-appendix.pdf',
        format='pdf',
        bbox_inches='tight',
        pad_inches=0)
    plt.close()


if __name__ == '__main__':
    # main()
    appendix()
