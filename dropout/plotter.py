import matplotlib.pyplot as plt
import numpy as np


def _load_data(mode):
    f = open('.results/dropout/mnist/epochs/{}.txt'.format(mode), 'r')
    rows = []
    for line in f:
        cols = [float(v) for v in line.strip().split('\t')]
        rows.append(cols)
    return np.array(rows)


def main():
    default_result = _load_data('default')
    # none_result = _load_data('none')
    default_with_sn_result = _load_data('default-with-sn')
    expectation_result = _load_data('expectation')
    exact_result = _load_data('exact')

    fig, axs = plt.subplots(1, 1, sharey=False)

    red = "#FF4841"
    blue = "#0099FE"
    green = '#44FF44'
    orange = '#FFAA00'

    target_idx = 7
    ax = axs
    ax.plot(
        default_with_sn_result[:, 0],
        default_with_sn_result[:, target_idx],
        color=green,
        label='default-with-sn')

    ax.plot(
        default_result[:, 0],
        default_result[:, target_idx],
        color=orange,
        label='default')

    # ax.plot(
    #     expectation_result[:, 0],
    #     expectation_result[:, target_idx],
    #     color=red,
    #     label='expectation')
    #
    # ax.plot(
    #     exact_result[:, 0],
    #     exact_result[:, target_idx],
    #     color=blue,
    #     label='exact')
    # ax.set_ylim([0.88, 0.91])
    # ax.set_ylim([0.8, 0.91])

    plt.savefig(
        'dropout/test.png', format='png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()
