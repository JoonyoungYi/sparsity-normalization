import matplotlib.pyplot as plt
import numpy as np


def _get_movielens_data():
    from movielens.dataset import Dataset
    dataset = Dataset("movielens-1m", Dataset.SPLIT_DEFAULT_10_PERCENT, seed=0)
    dataset.init_full_matrices()
    matrix = dataset.train_matrix + dataset.valid_matrix + dataset.test_matrix
    data = matrix.T
    c = np.sum(data > 1e-8, axis=1)
    return c


def _get_mnist_data():
    from mnist.datasets.mnist import load_data
    (x_train, y_train), (x_test, y_test) = load_data()
    data = np.concatenate([x_train, x_test])
    data = data.reshape([70000, 28 * 28])
    c = np.sum(data > 1e-8, axis=1)
    return c


def _get_fashion_data():
    from mnist.datasets.fashion import load_data
    (x_train, y_train), (x_test, y_test) = load_data()
    data = np.concatenate([x_train, x_test])
    data = data.reshape([70000, 28 * 28])
    c = np.sum(data > 1e-8, axis=1)
    print(np.max(c), np.min(c))
    return c


def _get_cifar_data():
    from mnist.datasets.cifar10 import load_data
    (x_train, y_train), (x_test, y_test) = load_data()
    data = np.concatenate([x_train, x_test])
    data = data.reshape([60000, 32 * 32 * 3])
    c = np.sum(data > 1e-8, axis=1)
    # print(np.max(c), np.min(c))
    return c


def main():
    try:
        movielens_data = np.load('.datasets/movielens-1m/count.npy')
    except:
        movielens_data = _get_movielens_data()
        np.save('.datasets/movielens-1m/count.npy', movielens_data)

    try:
        mnist_data = np.load('.datasets/mnist/count.npy')
    except:
        mnist_data = _get_mnist_data()
        np.save('.datasets/mnist/count.npy', mnist_data)

    try:
        fashion_data = np.load('.datasets/fashion-mnist/count.npy')
    except:
        fashion_data = _get_fashion_data()
        np.save('.datasets/fashion-mnist/count.npy', fashion_data)

    try:
        cifar_data = np.load('.datasets/cifar-10/count.npy')
    except:
        cifar_data = _get_cifar_data()
        np.save('.datasets/cifar-10/count.npy', cifar_data)

    # import matplotlib.pyplot as plt
    # plt.rcParams["font.family"] = "Times"

    fig, axs = plt.subplots(1, 4, sharey=False, tight_layout=True)
    fig.set_figwidth(12)
    fig.set_figheight(2)

    color = "#757575"
    # color = "#000000"

    axs[0].set_title("Movielens 1M")
    axs[0].hist(
        movielens_data,
        bins=50,
        range=(1, 3000),
        log=True,
        color=color,
        bottom=1,
        histtype='stepfilled')
    axs[0].set_ylabel('# of instances')
    axs[0].set_xlabel('# of nonzero units')


    axs[1].set_title("MNIST")
    axs[1].hist(
        mnist_data,
        bins=50,
        range=(1, 350),
        log=True,
        color=color,
        bottom=1,
        histtype='stepfilled')
    axs[1].set_xlabel('# of nonzero units')

    axs[2].set_title("Fashion MNIST")
    axs[2].hist(
        fashion_data,
        bins=50,
        range=(51, 750),
        log=True,
        color=color,
        bottom=1,
        histtype='stepfilled')
    axs[2].set_xlabel('# of nonzero units')

    axs[3].set_title("CIFAR")
    axs[3].hist(
        cifar_data,
        bins=50,
        range=(2723, 3072),
        log=True,
        color=color,
        bottom=1,
        histtype='stepfilled')
    axs[3].set_xlabel('# of nonzero units')

    fig.savefig('figures/sparsities/histogram.png', format='png')
    fig.savefig('figures/sparsities/histogram.pdf', format='pdf')


if __name__ == '__main__':
    main()
