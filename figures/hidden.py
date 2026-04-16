import matplotlib.pyplot as plt
import numpy as np


def _load_csv():
    f = open('figures/hiddens/cifar-10.csv', 'r')
    lines = []
    for line in f:
        cols = line.strip().split(',')
        cols = [float(c) if c else None for c in cols]
        lines.append(cols)
    return lines


def _get_x_and_y(data, col_idx):
    x, y = [], []
    for i, row in enumerate(data):
        if row[col_idx] is not None:
            x.append(row[0])
            y.append(row[col_idx])
    return x, y


def main_a():
    data = _load_csv()
    x_1, y_1 = _get_x_and_y(data, 1)
    x_2, y_2 = _get_x_and_y(data, 2)
    x_5, y_5 = _get_x_and_y(data, 5)
    x_6, y_6 = _get_x_and_y(data, 6)

    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    fig.set_figwidth(12)
    fig.set_figheight(4)

    red = "#FF4841"
    blue = "#0099FE"

    axs[0].plot(x_1, y_1, color=red, label='w/o SN')
    axs[0].plot(x_2, y_2, color=blue, label='w/ SN to input layer')
    axs[0].legend(fontsize='medium', loc='lower right')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('# of hidden units')

    axs[1].plot(x_1, y_1, color=red, label='w/o SN')
    axs[1].plot(x_5, y_5, color=blue, label='w/ SN to 1st hidden layer')
    axs[1].legend(fontsize='medium', loc='lower right')
    axs[1].set_xlabel('# of hidden units')

    axs[2].plot(x_1, y_1, color=red, label='w/o SN')
    axs[2].plot(x_6, y_6, color=blue, label='w/ SN to 2nd hidden layer')
    axs[2].legend(fontsize='medium', loc='lower right')
    axs[2].set_xlabel('# of hidden units')

    fig.savefig('figures/hiddens/a.png', format='png')
    fig.savefig('figures/hiddens/a.pdf', format='pdf')


def main_b():
    data = _load_csv()
    x_1, y_1 = _get_x_and_y(data, 1)
    x_2, y_2 = _get_x_and_y(data, 2)
    x_3, y_3 = _get_x_and_y(data, 3)

    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    fig.set_figwidth(6)
    fig.set_figheight(4)

    red = "#FF4841"
    blue = "#0099FE"
    yellow = '#FFAE3B'

    axs.plot(x_1, y_1, color=red, label='w/o SN')
    axs.plot(x_2, y_2, color=blue, label='w/ SN to input layer')
    axs.plot(
        x_3,
        y_3,
        color=yellow,
        # linestyle='dashed',
        label='w/ SN to input and 1st hidden layer')
    axs.legend(fontsize='medium', loc='lower right')
    axs.set_ylabel('Accuracy')
    axs.set_xlabel('# of hidden units')

    fig.savefig('figures/hiddens/b.png', format='png')
    fig.savefig('figures/hiddens/b.pdf', format='pdf')


if __name__ == '__main__':
    # main_a()
    main_b()
