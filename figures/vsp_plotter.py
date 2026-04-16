import numpy as np
from matplotlib import pyplot as plt

from common.datasets.movielens import Dataset


def _convert_lstm(x, T=50):
    x = x[:, 1].reshape([-1, 50])
    return x


def _plot(ax, ys, label, color):
    ax.plot(
        list(range(ys.shape[0])),
        np.mean(ys, axis=1),
        color=color,
        label=label)

    x = np.matmul(
        np.array(list(range(1, ys.shape[0] + 1))).reshape([-1, 1]),
        np.ones([1, ys.shape[1]])).reshape([-1])
    # print(x)
    ax.scatter(x, ys.reshape([-1, 1]), s=0.5, alpha=0.05, color=color)


def main():
    auto_impute_w_sn = np.load('auto_impute/figures/vsp/w_sn.npy')[:50]
    auto_impute_wo_sn = np.load('auto_impute/figures/vsp/wo_sn.npy')[:50]
    print(auto_impute_w_sn.shape)
    print(auto_impute_wo_sn.shape)

    lstm_w_sn = np.load('figures/vsps/samples/rnn_2_SN_default_1_test_1_2.npy')
    lstm_wo_sn = np.load(
        'figures/vsps/samples/rnn_2_woSN_default_1_test_1_2.npy')
    lstm_w_sn = _convert_lstm(lstm_w_sn)
    lstm_wo_sn = _convert_lstm(lstm_wo_sn)
    print(lstm_w_sn.shape)
    print(lstm_wo_sn.shape)

    autorec_w_sn = np.load(
        'autorec/vsp/samples-t100_usevectrue-target0-actsigmoid-cliptrue-sntrue-kprime52.44.npy'
    )[:50]
    autorec_wo_sn = np.load(
        'autorec/vsp/samples-t100_usevectrue-target0-actsigmoid-cliptrue-snfalse-kprime500.0.npy'
    )[:50]
    print(autorec_w_sn.shape)
    print(autorec_wo_sn.shape)

    fig, axs = plt.subplots(2, 3, figsize=(10, 5), sharey=False)
    plt.subplots_adjust(wspace=0.3)

    for i, sparsity_normalization in enumerate([False, True]):
        for j, (title, w_sn, wo_sn) in enumerate([
            ("Collaborative Filtering", autorec_w_sn, autorec_wo_sn),
            ("Electronic Medical Records", lstm_w_sn, lstm_wo_sn),
            ("Single Cell RNA", auto_impute_w_sn, auto_impute_wo_sn),
        ]):
            ax = axs[i][j]
            # if i == 0:
            #     ax.set_title(title)
            ax.set_xlim([1, wo_sn.shape[0] - 1])

            if sparsity_normalization:
                _plot(ax, w_sn, color="#008EFD", label="Zero Imputation w/ SN")
            else:
                _plot(
                    ax, wo_sn, color="#FF343A", label="Zero Imputation w/o SN")

    axs[1][1].set_xlabel("# known entries")
    axs[0][0].set_ylabel("model prediction w/o SN")
    axs[1][0].set_ylabel("model prediction w/ SN")
    # axs[1].legend(
    #     loc='upper center',
    #     bbox_to_anchor=(+0.5, -0.2),
    #     ncol=2,
    #     edgecolor='none')

    filename = 'figures/vsps/vsp-exp-wandwosn'
    plt.savefig(
        filename + '.png', format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(
        filename + '.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


def _get_count_dict(count_dict, train_data, use_user_vector=False):
    for i in range(train_data.shape[0]):
        _data = train_data[i]
        user_id = int(_data[0])
        item_id = int(_data[1])

        if use_user_vector:
            key = user_id
        else:
            key = item_id

        c = count_dict.get(key, 0)
        count_dict[key] = c + 1
    return count_dict


def _get_rating_dict(count_dict, train_data, use_user_vector=False):
    for i in range(train_data.shape[0]):
        _data = train_data[i]
        user_id = int(_data[0])
        item_id = int(_data[1])

        if use_user_vector:
            key = user_id
        else:
            key = item_id

        c = count_dict.get(key, 0)
        count_dict[key] = c + float(_data[2])
    return count_dict


def stat():
    fig, axs = plt.subplots(1, 3, figsize=(10, 1.6), sharey=False)
    plt.subplots_adjust(wspace=0.3)

    # Movielens 100K
    for j, title in enumerate([
            "Collaborative Filtering",
            "Electronic Medical Records",
            "Single Cell RNA",
    ]):
        ax = axs[j]
        ax.set_title(title)

    dataset = Dataset(
        'movielens-100k', Dataset.SPLIT_DEFAULT_10_PERCENT, seed=1)
    train_data = dataset.train_data
    valid_data = dataset.valid_data
    test_data = dataset.test_data

    # l0 와 mean activation의 관계 scatter
    count_dict = _get_count_dict({}, train_data, use_user_vector=True)
    count_dict = _get_count_dict(count_dict, valid_data, use_user_vector=True)

    rating_dict = _get_rating_dict({}, train_data, use_user_vector=True)
    rating_dict = _get_rating_dict(
        rating_dict, valid_data, use_user_vector=True)

    scatters = []
    for user_id, rating in rating_dict.items():
        c = max(count_dict.get(user_id, 1), 1)
        scatters.append([c, rating / c])
    movielens_points = np.array(scatters)

    axs[0].set_xlim([0, max(c for k, c in count_dict.items())])
    axs[0].scatter(
        movielens_points[:, 0],
        movielens_points[:, 1],
        s=1,
        alpha=1.0,
        color="#6A6A6A")
    axs[0].set_ylim([1.0, 5.0])

    # EMR
    data = np.load('figures/vsps/samples/data_sparsity.npz')
    units = data['units']
    labels = data['labels'].astype('int64')

    max_units = np.max(units)

    units_0 = units[np.where(labels == 0)[0]]
    units_1 = units[np.where(labels == 1)[0]]

    bins = 30

    data_per_bar = max_units // bins
    if max_units % bins != 0:
        data_per_bar += 1

    units_0_list = []
    units_1_list = []

    num_data_0 = 0
    num_data_1 = 0
    for i in range(data_per_bar, max_units + data_per_bar, data_per_bar):
        num_data_0 = np.where(units_0 <= i)[0].shape[0] - num_data_0
        num_data_1 = np.where(units_1 <= i)[0].shape[0] - num_data_1

        units_0_list.append(num_data_0)
        units_1_list.append(num_data_1)

    ind = np.arange(bins)

    units_0_array = np.array(units_0_list)
    units_1_array = np.array(units_1_list)
    units_total_array = units_0_array + units_1_array

    units_0_ratio = units_0_array / units_total_array
    units_1_ratio = units_1_array / units_total_array

    T = ind * data_per_bar
    power = 1 - units_0_ratio
    from scipy.interpolate import spline
    xnew = np.linspace(
        T.min(), T.max(),
        300)  #300 represents number of points to make between T.min and T.max
    power_smooth = spline(T, power, xnew)

    axs[1].plot(xnew, power_smooth, color="#6A6A6A")
    # print(units_0_ratio)
    axs[1].set_ylim([-0.1, 0.4])
    axs[1].set_xlim([0, 600])

    # rna
    name = 'drop10-seed1-dseed1-snF'
    original_matrix = np.load(
        'auto_impute/original_matrices/blakeley/{}.npy'.format(name))
    print(original_matrix)
    c = np.sum(original_matrix > 1e-20, axis=1)
    print(c.shape)
    y = np.sum(original_matrix, axis=1) / c

    axs[2].scatter(c, y, s=4, alpha=1.0, color="#6A6A6A")
    axs[2].set_ylim([0, 16])
    # axs[2].set_ylim([6.5, 9.5])

    # p1 = axs[1].bar(ind, units_0_ratio)
    # p2 = axs[1].bar(ind, units_1_ratio, bottom=units_0_ratio, label='In hospital death')

    #     ax.set_xlim([1, wo_sn.shape[0] - 1])
    #     _plot(ax, w_sn, color="#008EFD", label="Zero Imputation w/ SN")

    axs[0].set_ylabel("mean of target values")

    filename = 'figures/vsps/vsp-exp-stat'
    plt.savefig(
        filename + '.png', format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(
        filename + '.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


def _plot_vsp(fig, axs):
    auto_impute_w_sn = np.load('autoimpute/figures/vsp/w_sn.npy')[:50]
    auto_impute_wo_sn = np.load('autoimpute/figures/vsp/wo_sn.npy')[:50]
    print(auto_impute_w_sn.shape)
    print(auto_impute_wo_sn.shape)

    lstm_w_sn = np.load('figures/vsps/samples/rnn_2_SN_default_1_test_1_2.npy')
    lstm_wo_sn = np.load(
        'figures/vsps/samples/rnn_2_woSN_default_1_test_1_2.npy')
    lstm_w_sn = _convert_lstm(lstm_w_sn)
    lstm_wo_sn = _convert_lstm(lstm_wo_sn)
    print(lstm_w_sn.shape)
    print(lstm_wo_sn.shape)

    autorec_w_sn = np.load(
        'cfautorec/vsp/samples-t100_usevectrue-target0-actsigmoid-cliptrue-sntrue-kprime52.44.npy'
    )[:50]
    autorec_wo_sn = np.load(
        'cfautorec/vsp/samples-t100_usevectrue-target0-actsigmoid-cliptrue-snfalse-kprime500.0.npy'
    )[:50]
    print(autorec_w_sn.shape)
    print(autorec_wo_sn.shape)

    for i, sparsity_normalization in enumerate([False, True]):
        for j, (title, w_sn, wo_sn) in enumerate([
            ("Collaborative Filtering", autorec_w_sn, autorec_wo_sn),
            ("Electronic Medical Records", lstm_w_sn, lstm_wo_sn),
            ("Single-cell RNA Sequence", auto_impute_w_sn, auto_impute_wo_sn),
        ]):
            ax = axs[i + 1][j]
            # if i == 0:
            #     ax.set_title(title)
            ax.set_xlim([1, wo_sn.shape[0] - 1])

            if sparsity_normalization:
                _plot(ax, w_sn, color="#008EFD", label="Zero Imputation w/ SN")
            else:
                _plot(
                    ax, wo_sn, color="#FF343A", label="Zero Imputation w/o SN")

    axs[2][1].set_xlabel("# known entries")
    axs[1][0].set_ylabel("model prediction w/o SN")
    axs[2][0].set_ylabel("model prediction w/ SN")
    # axs[1].legend(
    #     loc='upper center',
    #     bbox_to_anchor=(+0.5, -0.2),
    #     ncol=2,
    #     edgecolor='none')


def _plot_stat(fig, axs):
    # Movielens 100K
    for j, title in enumerate([
            "Collaborative Filtering",
            "Electronic Medical Records",
            "Single-cell RNA Sequence",
    ]):
        ax = axs[j]
        ax.set_title(title)

    dataset = Dataset(
        'movielens-100k', Dataset.SPLIT_DEFAULT_10_PERCENT, seed=1)
    train_data = dataset.train_data
    valid_data = dataset.valid_data
    test_data = dataset.test_data

    # l0 와 mean activation의 관계 scatter
    count_dict = _get_count_dict({}, train_data, use_user_vector=True)
    count_dict = _get_count_dict(count_dict, valid_data, use_user_vector=True)

    rating_dict = _get_rating_dict({}, train_data, use_user_vector=True)
    rating_dict = _get_rating_dict(
        rating_dict, valid_data, use_user_vector=True)

    scatters = []
    for user_id, rating in rating_dict.items():
        c = max(count_dict.get(user_id, 1), 1)
        scatters.append([c, rating / c])
    movielens_points = np.array(scatters)

    axs[0].set_xlim([0, max(c for k, c in count_dict.items())])
    axs[0].scatter(
        movielens_points[:, 0],
        movielens_points[:, 1],
        s=1,
        alpha=1.0,
        color="#6A6A6A")
    axs[0].set_ylim([2.0, 5])

    # EMR
    data = np.load('figures/vsps/samples/data_sparsity.npz')
    units = data['units']
    labels = data['labels'].astype('int64')

    max_units = np.max(units)

    units_0 = units[np.where(labels == 0)[0]]
    units_1 = units[np.where(labels == 1)[0]]

    bins = 30

    data_per_bar = max_units // bins
    if max_units % bins != 0:
        data_per_bar += 1

    units_0_list = []
    units_1_list = []

    num_data_0 = 0
    num_data_1 = 0
    for i in range(data_per_bar, max_units + data_per_bar, data_per_bar):
        num_data_0 = np.where(units_0 <= i)[0].shape[0] - num_data_0
        num_data_1 = np.where(units_1 <= i)[0].shape[0] - num_data_1

        units_0_list.append(num_data_0)
        units_1_list.append(num_data_1)

    ind = np.arange(bins)

    units_0_array = np.array(units_0_list)
    units_1_array = np.array(units_1_list)
    units_total_array = units_0_array + units_1_array

    units_0_ratio = units_0_array / units_total_array
    units_1_ratio = units_1_array / units_total_array

    T = ind * data_per_bar
    power = 1 - units_0_ratio
    from scipy.interpolate import spline
    xnew = np.linspace(
        T.min(), T.max(),
        300)  #300 represents number of points to make between T.min and T.max
    power_smooth = spline(T, power, xnew)

    axs[1].plot(xnew, power_smooth, color="#6A6A6A")
    # print(units_0_ratio)
    axs[1].set_ylim([-0.0, 0.25])
    axs[1].set_xlim([0, 600])

    # rna
    name = 'drop10-seed1-dseed1-snF'
    original_matrix = np.load(
        'autoimpute/original_matrices/blakeley/{}.npy'.format(name))
    print(original_matrix)
    c = np.sum(original_matrix > 1e-20, axis=1)
    print(c.shape)
    y = np.sum(original_matrix, axis=1) / c

    axs[2].scatter(c, y, s=4, alpha=1.0, color="#6A6A6A")
    axs[2].set_ylim([0, 16])
    # axs[2].set_ylim([6.5, 9.5])

    # p1 = axs[1].bar(ind, units_0_ratio)
    # p2 = axs[1].bar(ind, units_1_ratio, bottom=units_0_ratio, label='In hospital death')

    #     ax.set_xlim([1, wo_sn.shape[0] - 1])
    #     _plot(ax, w_sn, color="#008EFD", label="Zero Imputation w/ SN")

    axs[0].set_ylabel("mean of target values")


def merge():
    fig, axs = plt.subplots(3, 3, figsize=(10, 8), sharey=False)
    plt.subplots_adjust(wspace=0.3)

    _plot_stat(fig, axs[0])
    _plot_vsp(fig, axs)

    filename = 'figures/vsps/vsp-exp-wandwosn'
    plt.savefig(
        filename + '.png', format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(
        filename + '.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    # stat()
    # main()
    merge()
