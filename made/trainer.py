import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from .models import MADE
from common.utils import str2bool
from common.utils import set_seed
from common.utils import append_text_results


def __evaluate(model, xte, args=None):
    torch.set_grad_enabled(False)
    model.eval()
    nsamples = args.samples
    x = xte
    N, D = x.size()
    B = 100  # batch size
    lossfs = []

    for start_data_index in range(0, N, B):
        # fetch the next batch of data
        xb = Variable(x[start_data_index:start_data_index + B])

        # get the logits, potentially run the same batch a number of times, resampling each time
        xbhat = torch.zeros_like(xb)
        for s in range(nsamples):
            model.update_masks()
            # forward the model
            xbhat += torch.sigmoid(model(xb))
        # input('')
        xbhat /= nsamples

        # evaluate the binary cross entropy loss
        loss = F.binary_cross_entropy(xbhat, xb, reduction='sum')
        lossf = loss.data.item()
        lossfs.append(lossf)

    nll = np.sum(lossfs) / N
    return nll


def __run_epoch(model, optimizer, xtr, args=None):
    torch.set_grad_enabled(True)
    model.train()
    x = xtr
    N, D = x.size()
    B = 100  # batch size
    lossfs = []
    nsamples = args.samples

    for start_data_index in tqdm(range(0, N, B)):

        # fetch the next batch of data
        xb = Variable(x[start_data_index:start_data_index + B])

        optimizer.zero_grad()
        for s in range(nsamples):
            # get the logits, potentially run the same batch a number of times, resampling each time
            model.update_masks()
            xbhat = model(xb)

            # evaluate the binary cross entropy loss
            _loss = F.binary_cross_entropy_with_logits(
                xbhat, xb, reduction='sum') / nsamples
            _lossf = _loss.data.item()
            lossfs.append(_lossf)
            _loss.backward()
        optimizer.step()

    nll = np.sum(lossfs) / N
    return nll


def _train(args):
    set_seed(args.seed, tensorflow=False, torch=True)

    # load the dataset
    print("loading binarized mnist from", args.data_path)
    mnist = np.load(args.data_path)
    train_data, valid_data, test_data = mnist['train_data'], mnist[
        'valid_data'], mnist['test_data']

    train_data = torch.from_numpy(train_data).cuda()
    valid_data = torch.from_numpy(valid_data).cuda()
    test_data = torch.from_numpy(test_data).cuda()

    # construct model and ship to GPU
    hidden_list = list(map(int, args.hiddens.split(',')))
    model = MADE(
        train_data.size(1),
        hidden_list,
        train_data.size(1),
        num_masks=args.num_masks,
        sparsity_normalization=args.sparsity_normalization)
    print("number of model parameters:",
          sum([np.prod(p.size()) for p in model.parameters()]))
    model.cuda()

    # set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # start the training
    min_test_nll = float('inf')
    min_test_epoch_idx = -1
    min_valid_nll = float('inf')
    min_valid_epoch_idx = -1
    final_test_nll = float('inf')
    train_nll = float('nan')
    for epoch_idx in range(args.max_epoch):
        # scheduler.step(epoch)
        valid_nll = __evaluate(model, valid_data, args=args)
        test_nll = __evaluate(model, test_data, args=args)
        if test_nll < min_test_nll:
            min_test_nll = test_nll
            min_test_epoch_idx = epoch_idx

        if valid_nll < min_valid_nll:
            min_valid_nll = valid_nll
            min_valid_epoch_idx = epoch_idx
            final_test_nll = test_nll

        print("%.4f" % train_nll, "%.4f" % valid_nll, "%.4f" % test_nll)
        print("%.4f" % min_valid_nll, "(%5d)" % min_valid_epoch_idx,
              "%.4f" % final_test_nll, "%.4f" % min_test_nll,
              "(%5d)" % min_test_epoch_idx)

        if min_valid_epoch_idx + args.early_stop_epoch < epoch_idx:
            break

        print("\n>> epoch %d" % (epoch_idx, ))
        train_nll = __run_epoch(model, optimizer, train_data, args=args)

    print("optimization done. full test set eval:")

    results = []
    results.append(args.seed)
    results.append(args.max_epoch)
    results.append(args.hiddens)
    results.append(args.num_masks)
    results.append(args.samples)
    results.append(args.learning_rate)
    results.append(args.early_stop_epoch)
    results.append(args.sparsity_normalization)
    results.append(train_nll)
    results.append(min_valid_nll)
    results.append(min_valid_epoch_idx)
    results.append(final_test_nll)
    results.append(min_test_nll)
    results.append(min_test_epoch_idx)
    append_text_results('made', 'binarized_mnist', 'results', 'default',
                        '\t'.join([str(r) for r in results]))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-q',
        '--hiddens',
        type=str,
        default='500',
        help="Comma separated sizes for hidden layers, e.g. 500, or 500,500")
    parser.add_argument(
        '-n',
        '--num_masks',
        type=int,
        default=1,
        help="Number of orderings for order/connection-agnostic training")
    parser.add_argument('--seed', type=int, default=1, help="")
    parser.add_argument(
        '-s',
        '--samples',
        type=int,
        default=1,
        help=
        "How many samples of connectivity/masks to average logits over during inference"
    )
    parser.add_argument(
        '--max_epoch',
        type=int,
        # default=1,
        default=1000000,
        help="")
    parser.add_argument('--early_stop_epoch', type=int, default=30, help="")
    parser.add_argument('--learning_rate', type=float, default=5e-3, help="")
    parser.add_argument(
        '--sparsity_normalization',
        type=str2bool,
        # default=True,
        default=False,
        help="")
    args = parser.parse_args()
    args.data_path = 'made/binarized_mnist.npz'
    _train(args)


if __name__ == '__main__':
    main()
