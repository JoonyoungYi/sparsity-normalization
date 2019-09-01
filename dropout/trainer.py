import os
import random
import argparse
import collections

from tqdm import tqdm
import numpy as np
from scipy.special import logsumexp
import torch

from .dataset import load_data
from .model import MLP
from .logger import Logger
from common.utils import set_torch_seed

def _get_variance(model, test_x, args):
    test_y_preds = [
        model(test_x, test_dropout_rate=args.dropout_rate)
        for _ in range(args.T)
    ]
    test_y_preds = torch.stack(test_y_preds, dim=0)
    variance = test_y_preds.std(dim=0)
    return variance


def main(args):
    # get boston dataset
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(
        args.input_normalization, dataset_seed=args.dataset_seed)

    # Setting seed for reproducing
    set_torch_seed(args.seed)
    logger = Logger(args)

    # construct model
    model = MLP(
        n_input_unit=train_x.shape[1],
        n_hidden_unit=args.n_hidden_unit,
        n_hidden_layer=args.n_hidden_layer,
        dropout_rate=args.dropout_rate,
        dropout_mode=args.dropout_mode,
        activation_fn=args.activation_fn)
    model = model.cuda()

    # define criterion and optimizer
    criterion = torch.nn.MSELoss()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=args.learning_rate, eps=1e-8)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params=model.parameters(), lr=args.learning_rate, momentum=0.0)
    elif args.optimizer == 'momentum':
        optimizer = torch.optim.SGD(
            params=model.parameters(), lr=args.learning_rate, momentum=0.9)
    else:
        raise NotImplementedError()

    if not args.n_batch or args.n_batch <= 0:
        n_batch = train_x.shape[0]
    else:
        n_batch = args.n_batch

    min_valid_rmse = float('inf')
    min_valid_epoch_idx = -1
    final_test_rmse = float('inf')
    for epoch_idx in tqdm(range(1, args.max_epoch + 1)):
        # training
        model.train()

        batch_mses = []
        for i in range(0, train_x.shape[0], n_batch):
            batch_x = train_x[i:i + n_batch]
            batch_y = train_y[i:i + n_batch]
            batch_y_pred = model(batch_x)
            batch_loss = criterion(batch_y_pred, batch_y)
            batch_mses.append([batch_loss.item(), batch_x.shape[0]])

            # backprop
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        batch_rmse = np.sqrt(
            sum([l * c for l, c in batch_mses]) / sum(c
                                                      for _, c in batch_mses))

        # testing
        with torch.no_grad():
            model.eval()
            train_y_pred = model(train_x)
            train_loss = criterion(train_y_pred, train_y)
            train_rmse = np.sqrt(train_loss.item())

            valid_y_pred = model(valid_x)
            valid_loss = criterion(valid_y_pred, valid_y)
            valid_rmse = np.sqrt(valid_loss.item())

            test_y_pred = model(test_x)
            test_y_pred = test_y_pred * std_y_train + mean_y_train

            test_loss = criterion(test_y_pred, test_y)
            test_rmse = np.sqrt(test_loss.item())

        if min_valid_rmse > valid_rmse:
            min_valid_rmse = valid_rmse
            min_valid_epoch_idx = epoch_idx
            final_test_rmse = test_rmse

        # print(epoch_idx, train_rmse, test_rmse, min_test_rmse)

        # save results
        logger.log_epoch(epoch_idx, batch_rmse, train_rmse, valid_rmse,
                         test_rmse)

    print('MIN Test RMSE:', final_test_rmse, min_valid_epoch_idx)

    with torch.no_grad():
        model.eval()

        std = _get_variance(model, test_x, args)
        print('Mean Standard Deviation:', std.mean())

    with torch.no_grad():
        # plot relation of sparsity and expected activation
        model.eval()

        for i in range(1, 100):
            rate = i * 0.01
            test_y_pred = model(test_x, test_dropout_rate=rate)
            test_y_pred = test_y_pred * std_y_train + mean_y_train

            test_y_pred_mean = test_y_pred.mean().item()
            if i % 10 == 0:
                print('{:.2f} {}'.format(rate, test_y_pred_mean))
            logger.log_test_drop(rate, test_y_pred_mean)


def parse_arguments(parser):
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'sgd', 'momentum'])
    parser.add_argument(
        '--activation_fn',
        type=str,
        default='relu',
        choices=['relu', 'leaky_relu', 'elu', 'tanh', 'sigmoid'])
    parser.add_argument('--n_hidden_layer', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument(
        '--n_batch',
        type=int,
        # default=None,
        default=128,
        help='None이면 full batch를 의미합니다.')
    parser.add_argument('--n_hidden_unit', type=int, default=50)


if __name__ == '__main__':
    # arguement passing
    parser = argparse.ArgumentParser(description='dropout')
    parse_arguments(parser)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset_seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='boston')
    parser.add_argument(
        '--input_normalization',
        type=str,
        # default='none',
        # default='min-max',
        default='standardization',
        choices=['none', 'min-max', 'standardization'])
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument(
        '--dropout_mode',
        type=str,
        # default='default',
        default='sparsity-normalization',
        choices=['default', 'sparsity-normalization'])
    parser.add_argument('--max_epoch', type=int, default=100000)
    parser.add_argument('--T', type=int, default=10000)

    args = parser.parse_args()
    assert args.dataset == 'boston'

    main(args)
