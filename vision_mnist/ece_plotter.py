import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def _softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    # y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def ___calibration_error(y, p_mean, num_bins=10):
    """Compute the calibration.
    References:
    https://github.com/google-research/google-research/blob/master/uncertainties/sources/postprocessing/metrics.py
    https://arxiv.org/abs/1706.04599
    https://arxiv.org/abs/1807.00263
    Args:
      y: one-hot encoding of the true classes, size (?, num_classes)
      p_mean: numpy array, size (?, num_classes)
             containing the mean output predicted probabilities
      num_bins: number of bins
    Returns:
      cal: a dictionary
        {reliability_diag: realibility diagram
         ece: Expected Calibration Error
         mce: Maximum Calibration Error
        }
    """
    # Compute for every test sample x, the predicted class.
    class_pred = np.argmax(p_mean, axis=1)
    # and the confidence (probability) associated with it.
    conf = np.max(p_mean, axis=1)
    # Convert y from one-hot encoding to the number of the class
    y = np.argmax(y, axis=1)
    # Storage
    acc_tab = np.zeros(num_bins)  # empirical (true) confidence
    mean_conf = np.zeros(num_bins)  # predicted confidence
    nb_items_bin = np.zeros(num_bins)  # number of items in the bins
    tau_tab = np.linspace(0, 1, num_bins + 1)  # confidence bins
    for i in np.arange(num_bins):  # iterate over the bins
        # select the items where the predicted max probability falls in the bin
        # [tau_tab[i], tau_tab[i + 1)]
        sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
        nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
        # select the predicted classes, and the true classes
        class_pred_sec, y_sec = class_pred[sec], y[sec]
        # average of the predicted max probabilities
        mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
        # compute the empirical confidence
        acc_tab[i] = np.mean(
            class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]

    # Reliability diagram
    reliability_diag = (mean_conf, acc_tab)
    # Expected Calibration Error
    ece = np.average(
        np.absolute(mean_conf - acc_tab),
        weights=nb_items_bin.astype(np.float) / np.sum(nb_items_bin))
    # Maximum Calibration Error
    mce = np.max(np.absolute(mean_conf - acc_tab))
    # Saving
    cal = {'reliability_diag': reliability_diag, 'ece': ece, 'mce': mce}
    return cal


def __get_calibration_error(outputs, ys, yhats, temparature=1.0, n_bin=10):
    p_mean = _softmax(outputs / temparature, axis=1)
    n_class = np.max(ys) + 1
    y = np.eye(n_class)[ys]

    # # print('>> temparature:', temparature)
    # outputs =
    #
    # confidence_dict = {}
    # accuracy_dict = {}
    # for i in range(outputs.shape[0]):
    #     output = outputs[i]
    #     y = ys[i]
    #     yhat = yhats[i]
    #
    #     for j in range(10):
    #         # j = y
    #         # j = yhat
    #         confidence = output[j]
    #         key = max(int(confidence * n_bin), 1)  # key for bin
    #
    #         confidences = confidence_dict.get(key, [])
    #         confidences.append(confidence)
    #         confidence_dict[key] = confidences
    #
    #         accuracies = accuracy_dict.get(key, [])
    #         accuracies.append(1 if y == j else 0)
    #         accuracy_dict[key] = accuracies
    #
    # calibration_errors = []  # CE for MCE, ECE
    # for key in range(1, n_bin + 1):
    #     confidences = confidence_dict.get(key, [])
    #     accuracies = accuracy_dict.get(key, [])
    #
    #     if len(confidences) == 0:
    #         continue
    #
    #     average_confidence = sum(confidences) / len(confidences)
    #     average_accuracy = sum(accuracies) / len(accuracies)
    #
    #     # print('>>', key, average_confidence, average_accuracy)
    #     calibration_errors.append(abs(average_confidence - average_accuracy))
    #
    # mce = max(calibration_errors)
    # ece = sum(calibration_errors) / len(calibration_errors)
    # return mce, ece
    result = ___calibration_error(y, p_mean, num_bins=n_class)
    return result.get('mce'), result.get('ece')


def _get_expected_calibration_errors(sparsity_normalization, dataset,
                                     temparatures):
    outputs = np.load('vision_mnist/result-outputs-{}-{}.npy'.format(
        sparsity_normalization, dataset).lower())
    ys = np.load('vision_mnist/result-test-ys-{}.npy'.format(dataset).lower())
    yhats = np.load('vision_mnist/result-yhats-{}-{}.npy'.format(
        sparsity_normalization, dataset).lower())

    eces = []
    for temparature in tqdm(temparatures):
        mce, ece = __get_calibration_error(
            outputs, ys, yhats, temparature=temparature)
        eces.append(ece)
    return eces


def main():
    # dataset = 'mnist'
    dataset = 'fashion'
    # dataset = 'cifar100'

    start_t = 0.9
    end_t = 2
    t_unit = 0.01

    t = start_t
    temparatures = []
    while t < end_t:
        temparatures.append(t)
        t += t_unit

    eces_with_sn = _get_expected_calibration_errors(True, dataset,
                                                    temparatures)
    eces_without_sn = _get_expected_calibration_errors(False, dataset,
                                                       temparatures)

    # print(eces_with_sn)
    # print(eces_without_sn)

    plt.plot(temparatures, eces_with_sn)
    plt.plot(temparatures, eces_without_sn)
    plt.savefig(
        'vision_mnist/result-calibration-errors-{}.png'.format(dataset)
        .lower(),
        format='png')
    plt.close()

    # x_class_a = outputs[:, 0][train_ys == 0]
    # y_class_a = train_outputs[:, 1][train_ys == 0]
    #
    # x_class_b = train_outputs[:, 0][train_ys == 1]
    # y_class_b = train_outputs[:, 1][train_ys == 1]
    #
    # plt.scatter(x_class_a, y_class_a, s=size)
    # plt.scatter(x_class_b, y_class_b, s=size)


if __name__ == '__main__':
    main()
