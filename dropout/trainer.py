import os
import random
import argparse
import collections

from tqdm import tqdm
import numpy as np
import torch

from .dataset import load_data
from .model import MLP
from common.utils import set_torch_seed
from common.utils import append_text_results


def _save_results(args, epoch_idx, train_rmse, test_rmse):
    file_name = 'seed{seed}_in{input_normalization}_n-hunit{n_hidden_unit}_n-hlayer{n_hidden_layer}_dropout-rate{dropout_rate}_dropout-mode{dropout_mode}_lr{learning_rate}_max-epoch{max_epoch}'.format(
        seed=args.seed,
        input_normalization=args.input_normalization,
        n_hidden_unit=args.n_hidden_unit,
        n_hidden_layer=args.n_hidden_layer,
        dropout_rate=args.dropout_rate,
        dropout_mode=args.dropout_mode,
        learning_rate=args.learning_rate,
        max_epoch=args.max_epoch, )
    append_text_results('dropout', args.dataset, 'epoch', file_name,
                        "\t".join([epoch_idx, train_rmse, test_rmse]))


def main(args):
    # Setting seed for reproducing
    set_torch_seed(args.seed)

    # get boston dataset
    train_x, train_y, test_x, test_y = load_data(args.input_normalization,
                                                 args.split_ratio)

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

    for epoch_idx in tqdm(range(1, args.max_epoch + 1)):
        # training
        model.train()
        train_y_pred = model(train_x)
        train_loss = criterion(train_y_pred, train_y)
        train_rmse = np.sqrt(train_loss.item())

        # backprop
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # testing
        with torch.no_grad():
            model.eval()
            test_y_pred = model(test_x)
            test_loss = criterion(test_y_pred, test_y)
            test_rmse = np.sqrt(test_loss.item())

        # save results
        _save_results(args, epoch_idx, train_rmse, test_rmse)

    with torch.no_grad():
        # plot relation of sparsity and expected activation
        model.eval()

        for rate in [i * 0.1 for i in range(1, 10)]:
            test_y_pred = model(test_x, test_dropout_rate=rate)
            test_y_pred_mean = test_y_pred.mean().item()
            print(rate, test_y_pred_mean)


if __name__ == '__main__':
    # arguement passing
    parser = argparse.ArgumentParser(description='dropout')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='boston')
    parser.add_argument(
        '--input_normalization',
        type=str,
        # default='none',
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

    args = parser.parse_args()
    assert args.dataset == 'boston'

    main(args)
