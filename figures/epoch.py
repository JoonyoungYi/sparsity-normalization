import matplotlib.pyplot as plt
import numpy as np


def _load_array(dataset, sparsity_normalization, num=12):
    with open('vision_mnist/result-epochs-{}-{}.txt.{}'.format(
            sparsity_normalization, dataset, num).lower(), 'r') as f:
        rows = []
        for line in f:
            cols = [float(t) for t in line.strip().split('\t')]
            if cols[0] < 0.5:
                rows = []
            rows.append(cols)

        return np.array(rows)


def _plot_1():
    # dataset = 'cifar100'
    # dataset = 'mnist'
    dataset = 'fashion'
    result_with_sn = _load_array(dataset, True, num=12)
    result_without_sn = _load_array(dataset, False, num=12)

    plt.rcParams['lines.linewidth'] = 1

    fig, axs = plt.subplots(1, 1, sharey=False)
    fig.set_figwidth(1.8)
    fig.set_figheight(1.2)

    red = "#FF4841"
    red_blur = '#BF3530'
    blue = "#0099FE"
    blue_blur = '#0073BF'

    axs = [axs]

    axs[0].plot(
        result_without_sn[:, 0],
        result_without_sn[:, 2],
        color=red_blur,
        label='train acc. w/o SN')
    axs[0].plot(
        result_without_sn[:, 0],
        0.5 * (result_without_sn[:, 4] + result_without_sn[:, 3]),
        color=red,
        label='test acc. w/o SN')
    axs[0].plot(
        result_with_sn[:, 0],
        result_with_sn[:, 2],
        color=blue_blur,
        label='train acc. w/ SN')
    axs[0].plot(
        result_with_sn[:, 0],
        0.5 * (result_with_sn[:, 4] + result_with_sn[:, 3]),
        color=blue,
        label='test acc. w/ SN')
    # axs[0].legend(loc='upper left')

    axs[0].set_ylabel('Test Acc.')
    axs[0].set_xlabel('# of Iteration')
    axs[0].set_xlim([0, 3000])
    axs[0].set_ylim([0.7, 0.85])

    axs[0].legend(
        loc='upper center',
        bbox_to_anchor=(0.4, -0.4),
        ncol=1,
        edgecolor='none')

    plt.savefig(
        'vision_mnist/result-epoch-acc-{}.png'.format(dataset).lower(),
        format='png',
        bbox_inches='tight',
        pad_inches=0)
    plt.savefig(
        'vision_mnist/result-epoch-acc-{}.pdf'.format(dataset).lower(),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0)
    plt.close()


def _plot_2():
    # dataset = 'cifar100'
    # dataset = 'mnist'
    dataset = 'fashion'
    result_with_sn = _load_array(dataset, True, 11)
    result_without_sn = _load_array(dataset, False, 11)

    fig, axs = plt.subplots(
        1,
        2,
        sharey=False, )
    plt.subplots_adjust(wspace=0.6)
    fig.set_figwidth(4)
    fig.set_figheight(.7)

    red = "#FF4841"
    red_blur = '#BF3530'
    blue = "#0099FE"
    blue_blur = '#0073BF'

    # axs = [axs]
    ax = axs[0]
    ax.plot(
        result_without_sn[:, 0],
        result_without_sn[:, 9],
        color=red_blur,
        label='sparse group w/o SN')
    ax.plot(
        result_without_sn[:, 0],
        result_without_sn[:, 10],
        color=red,
        label='dense group w/o SN')
    ax.plot(
        result_with_sn[:, 0],
        result_with_sn[:, 9],
        color=blue_blur,
        label='sparse group w/ SN')
    ax.plot(
        result_with_sn[:, 0],
        result_with_sn[:, 10],
        color=blue,
        label='dense group w/ SN')
    # ax.legend(loc='lower right')

    ax.set_ylabel('Average of\n output')
    ax.set_xlabel('# of Iteration')
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 20])

    ax = axs[1]
    l1 = ax.plot(
        result_without_sn[:, 0],
        result_without_sn[:, 5],
        color=red_blur,
        label='sparse group w/o SN')
    l2 = ax.plot(
        result_without_sn[:, 0],
        result_without_sn[:, 6],
        color=red,
        label='dense group w/o SN')
    l3 = ax.plot(
        result_with_sn[:, 0],
        result_with_sn[:, 5],
        color=blue_blur,
        label='sparse group w/ SN')
    l4 = ax.plot(
        result_with_sn[:, 0],
        result_with_sn[:, 6],
        color=blue,
        label='dense group w/ SN')
    # ax,
    # lines = [l1, l2, l3, l4]
    # labels = [
    #     , 'dense group w/o SN', 'sparse group w/ SN',
    #     'dense group w/ SN'
    # ]

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(-0.4, -0.7),
        ncol=2,
        edgecolor='none')

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(
    #     loc='upper center',
    #     ncol=2,
    #     bbox_to_anchor=(-0.2, -0.3),
    #     fontsize='medium',
    #     edgecolor='none')

    # axs[idx].legend(loc='upper center', , ncol=2, edgecolor='none')

    ax.set_ylabel('Test Acc.')
    ax.set_xlabel('# of Iteration')
    ax.set_xlim([0, 1000])
    ax.set_ylim([0.45, 0.85])

    # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #        ncol=1, mode="expand", borderaxespad=0.)

    # fig.subplots_adjust(wspace=0)  # or whatever
    # fig.subplots_adjust(bottom=0.4)

    # fig.text(0.5, 0.04, '# of Iterations', ha='center')
    # # ax.tight_layout()
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(
        'vision_mnist/result-epoch-acc-act-{}.png'.format(dataset).lower(),
        format='png',
        bbox_inches='tight',
        pad_inches=0)
    plt.savefig(
        'vision_mnist/result-epoch-acc-act-{}.pdf'.format(dataset).lower(),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0)
    plt.close()


if __name__ == '__main__':
    _plot_1()
    # _plot_2()
