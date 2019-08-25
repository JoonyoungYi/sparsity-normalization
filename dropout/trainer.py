import os
import random
import argparse
import collections
from utils import dict2tsv
from tqdm import tqdm

import numpy as np
import torch
from dataset import init_boston
from model import MLP


def main():
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
    input_dim = train_x.shape[1]
    model = MLP(input_dim, args.hidden_unit, args.layer_num, args.dropout_rate,
                args.dropout_option)
    model = model.cuda()

    # define criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    for epoch in tqdm(range(1, args.epochs + 1)):
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
            dict2tsv(res, 'results/{}_{}_{}_{}_{}.txt'.format(
                args.hidden_unit, args.layer_num, args.dropout_rate, args.lr,
                args.dropout_option))

    # plot correlation of sparsity and expected activation
    with torch.no_grad():
        model.train()

        expected_act = []
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pred_y = model(test_x, dropout_new=p)
            expected_act.append(pred_y.mean().item())

        print('correlation of sparsity and expected activation:')
        print('\t'.join(str(f) for f in expected_act))


if __name__ == '__main__':
    # arguement passing
    parser = argparse.ArgumentParser(description='dropout')
    parser.add_argument(
        '--input_normalization',
        type=str,
        default='min-0-max-1',
        choices=['none', 'min-0-max-1', 'min--1-max-1', 'standardization'])
    #default='standardization')
    parser.add_argument('--split_ratio', type=float, default=0.8)

    parser.add_argument('--hidden_unit', type=int, default=1024)
    parser.add_argument('--layer_num', type=int, default=5)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument(
        '--dropout_option',
        type=str,
        choices=['sn', 'default_train', 'default_test', 'none'],
        default='default_train')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1000)

    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # Setting seed for reproducing
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()
