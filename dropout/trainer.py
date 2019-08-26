import os
import random
import argparse
import collections

from tqdm import tqdm
import numpy as np
import torch

from .dataset import load_data
from .model import MLP
from .logger import Logger
from common.utils import set_torch_seed

def main(args):
    # get boston dataset
    train_x, train_y, test_x, test_y = load_data(args.input_normalization,
                                                 args.split_ratio)
    y_mean = train_y.mean().item()
    y_std = train_y.std().item()

    train_y = (train_y - y_mean) / y_std

    # Setting seed for reproducing
    set_torch_seed(args.seed)
    logger = Logger(args)

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
        params=model.parameters(), lr=args.learning_rate, eps=1e-7)

    min_test_rmse = float('inf')

    for epoch_idx in tqdm(range(1, args.max_epoch + 1)):
        # training
        model.train()
        train_mses = []
        for i in range(0, train_x.shape[0], args.n_batch):
            batch_x = train_x[i:i + args.n_batch]
            batch_y = train_y[i:i + args.n_batch]
            batch_y_pred = model(batch_x)
            batch_loss = criterion(batch_y_pred, batch_y)
            train_mses.append([batch_loss.item(), batch_x.shape[0]])

            # backprop
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        train_rmse = np.sqrt(sum([l * c for l, c in train_mses]) / sum(c for _, c in train_mses))

        # testing
        with torch.no_grad():
            model.eval()
            test_y_pred = model(test_x)
            test_y_pred = test_y_pred * y_std + y_mean

            test_loss = criterion(test_y_pred, test_y)
            test_rmse = np.sqrt(test_loss.item())

        if min_test_rmse > test_rmse:
            min_test_rmse = test_rmse

        # save results
        logger.log_epoch(epoch_idx, train_rmse, test_rmse, min_test_rmse)

    print('MIN Test RMSE:', min_test_rmse)

    with torch.no_grad():
        # plot relation of sparsity and expected activation
        model.eval()

        for i in range(1, 100):
            rate = i * 0.01
            test_y_pred = model(test_x, test_dropout_rate=rate)
            test_y_pred = test_y_pred * y_std + y_mean

            test_y_pred_mean = test_y_pred.mean().item()
            if i % 10 == 0:
                print(rate, test_y_pred_mean)
            logger.log_test_drop(rate, test_y_pred_mean)


if __name__ == '__main__':
    # arguement passing
    parser = argparse.ArgumentParser(description='dropout')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_batch', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='boston')
    parser.add_argument(
        '--input_normalization',
        type=str,
        # default='none',
        # default='min-max',
        default='standardization',
        choices=['none', 'min-max', 'standardization'])
    parser.add_argument('--n_hidden_unit', type=int, default=50)
    parser.add_argument('--n_hidden_layer', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.05)
    parser.add_argument(
        '--dropout_mode',
        type=str,
        default='default',
        # default='sparsity-normalization',
        choices=['default', 'sparsity-normalization'])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=2000)
    parser.add_argument('--split_ratio', type=float, default=0.9)

    args = parser.parse_args()
    assert args.dataset == 'boston'

    main(args)
