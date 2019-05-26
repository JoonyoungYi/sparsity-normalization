import matplotlib.pyplot as plt
import numpy as np


def main():

    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)
    fig.set_figwidth(12)
    fig.set_figheight(4)

    # red = "#FF4841"
    blue = "#0099FE"
    # yellow = '#FFAE3B'

    axs[0].plot(
        [400, 500, 600, 700, 800, 900, 1000],
        [0.8822, 0.8804, 0.8799, 0.8800, 0.8801, 0.880, 0.8802],
        color=blue)
    axs[0].set_ylabel('Test RMSE')
    axs[0].set_xlabel('# of hidden units')

    axs[1].plot(
        [500, 600, 700, 800, 900, 1000],
        [0.8292, 0.8289, 0.8288, 0.8285, 0.8275, 0.8277],
        color=blue)
    axs[1].set_xlabel('# of hidden units')

    fig.savefig('figures/capacities/autorec-hidden.png', format='png')
    fig.savefig('figures/capacities/autorec-hidden.pdf', format='pdf')


if __name__ == '__main__':
    main()
