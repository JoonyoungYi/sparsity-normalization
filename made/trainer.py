"""
Trains MADE on Binarized MNIST, which can be downloaded here:
https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .models import MADE


# ------------------------------------------------------------------------------
def run_epoch(split, upto=None):
    torch.set_grad_enabled(
        split == 'train'
    )  # enable/disable grad for efficiency of forwarding test batches
    model.train() if split == 'train' else model.eval()
    nsamples = 1 if split == 'train' else args.samples
    x = xtr if split == 'train' else xte
    N, D = x.size()
    B = 100  # batch size
    nsteps = N // B if upto is None else min(N // B, upto)
    lossfs = []

    _total_batch = 0
    for step in range(nsteps):

        # fetch the next batch of data
        xb = Variable(x[step * B:step * B + B])
        _n_batch = xb.shape[0]

        # get the logits, potentially run the same batch a number of times, resampling each time
        xbhat = torch.zeros_like(xb)
        for s in range(nsamples):
            # perform order/connectivity-agnostic training by resampling the masks
            if step % args.resample_every == 0 or split == 'test':  # if in test, cycle masks every time
                model.update_masks()
            # forward the model
            xbhat += model(xb)
        # input('')
        xbhat /= nsamples

        # evaluate the binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(xbhat, xb, reduction='sum')
        lossf = loss.data.item()
        lossfs.append(lossf)
        _total_batch += _n_batch

        # backward/update
        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

    nll = np.sum(lossfs) / _total_batch
    print("%s epoch average loss: %f" % (split, nll))
    return nll


# ------------------------------------------------------------------------------

if __name__ == '__main__':
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
    parser.add_argument(
        '-r',
        '--resample_every',
        type=int,
        default=20,
        help=
        "For efficiency we can choose to resample orders/masks only once every this many steps"
    )
    parser.add_argument(
        '-s',
        '--samples',
        type=int,
        default=1,
        help=
        "How many samples of connectivity/masks to average logits over during inference"
    )
    parser.add_argument('--max_epoch', type=int, default=10000, help="")
    parser.add_argument('--learning_rate', type=float, default=1e-2, help="")
    parser.add_argument(
        '--sparsity_normalization',
        # default=True,
        default=False,
        help="")
    args = parser.parse_args()
    args.data_path = 'made/binarized_mnist.npz'

    # --------------------------------------------------------------------------

    # reproducibility is good
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # load the dataset
    print("loading binarized mnist from", args.data_path)
    mnist = np.load(args.data_path)
    xtr, xte = mnist['train_data'], mnist['valid_data']
    xtr = torch.from_numpy(xtr).cuda()
    xte = torch.from_numpy(xte).cuda()

    # construct model and ship to GPU
    hidden_list = list(map(int, args.hiddens.split(',')))
    model = MADE(
        xtr.size(1),
        hidden_list,
        xtr.size(1),
        num_masks=args.num_masks,
        sparsity_normalization=args.sparsity_normalization)
    print("number of model parameters:",
          sum([np.prod(p.size()) for p in model.parameters()]))
    model.cuda()

    # set up the optimizer
    opt = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
    # opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=45, gamma=0.1)

    # start the training
    min_test_nll = float('inf')
    min_test_epoch_idx = -1
    for epoch_idx in range(args.max_epoch):
        print("epoch %d" % (epoch_idx, ))
        # scheduler.step(epoch)
        test_nll = run_epoch(
            'test',
            upto=5)  # run only a few batches for approximate test accuracy
        if test_nll < min_test_nll:
            min_test_nll = test_nll
            min_test_epoch_idx = epoch_idx

        print("min test epoch average loss:", "%.4f" % min_test_nll,
              "(%d)" % min_test_epoch_idx)

        run_epoch('train')

    print("optimization done. full test set eval:")
    run_epoch('test')
