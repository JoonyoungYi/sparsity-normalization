import os
import random
import argparse
import collections

from tqdm import tqdm
import numpy as np
import torch

from .dataset import init_boston
from .model import MLP
from .utils import dict2tsv
from common.utils import set_torch_seed


def main(args):
    # Setting seed for reproducing
    set_torch_seed(args.seed)

    # get boston dataset
    train_x, train_y, test_x, test_y = init_boston(args.input_normalization,
                                                   args.split_ratio)

    # numpy to torch tensor
    train_x = torch.from_numpy(train_x).type(torch.FloatTensor).cuda()
    train_y = torch.from_numpy(train_y).type(
        torch.FloatTensor).unsqueeze(1).cuda()
    test_x = torch.from_numpy(test_x).type(torch.FloatTensor).cuda()
    test_y = torch.from_numpy(test_y).type(
        torch.FloatTensor).unsqueeze(1).cuda()

    # construct model
    model = MLP(
        n_input_unit=train_x.shape[1],
        n_hidden_unit=args.n_hidden_unit,
        n_hidden_layer=args.n_hidden_layer,
        dropout_rate=args.dropout_rate,
        dropout_mode=args.dropout_mode)
    model = model.cuda()

    # define criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.learning_rate)

    for epoch in tqdm(range(1, args.max_epoch + 1)):
        # training
        model.train()
        pred_y = model(train_x)
        train_loss = criterion(pred_y, train_y)

        # backprop
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # testing
        with torch.no_grad():
            model.eval()
            pred_y = model(test_x)
            test_loss = criterion(pred_y, test_y)

        if not args.testing:
            # save results
            res = collections.OrderedDict()
            res['epoch'] = epoch
            res['train_loss'] = np.sqrt(train_loss.item())
            res['test_loss'] = np.sqrt(test_loss.item())
            # dict2tsv(res, 'results/{}_{}_{}_{}_{}.txt'.format(
            #     args.hidden_unit, args.layer_num, args.dropout_rate, args.lr,
            #     args.dropout_option))

        # plot correlation of sparsity and expected activation
    with torch.no_grad():
        model.eval()

        expected_act = []
        for rate in [i * 0.1 for i in range(1, 10)]:
            pred_y = model(test_x, test_dropout_rate=rate)
            print(rate, pred_y.mean().item())
            expected_act.append(pred_y.mean().item())

        print('correlation of sparsity and expected activation:')
        print('\t'.join(str(f) for f in expected_act))


if __name__ == '__main__':
    # arguement passing
    parser = argparse.ArgumentParser(description='dropout')
    parser.add_argument(
        '--input_normalization',
        type=str,
        # default='min-max',
        default='standardization',
        choices=['none', 'min-max', 'standardization'])
    parser.add_argument('--n_hidden_unit', type=int, default=1024)
    parser.add_argument('--n_hidden_layer', type=int, default=4)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument(
        '--dropout_mode',
        type=str,
        default='default',
        # default='sparsity-normalization',
        choices=['default', 'sparsity-normalization'])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=1000)

    parser.add_argument('--split_ratio', type=float, default=0.8)

    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    main(args)
