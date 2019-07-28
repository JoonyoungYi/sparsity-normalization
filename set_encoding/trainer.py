import torch
import torch.nn as nn

from common.utils import set_seed
from .dataset import Dataset


def _train(args):
    set_seed(args.seed)
    dataset = Dataset(args)

    if args.model == 'set-transformer':
        from .models import SetTransformer
        model = SetTransformer()
    elif args.model == 'deep-set':
        from .models import DeepSet
        model = DeepSet(args.deepset_pooling)
    else:
        raise NotImplementedError()

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.L1Loss().cuda()
    losses = []
    for iter_idx in range(500):
        x, y = dataset.get_train_batch(iter_idx)
        x, y = torch.from_numpy(x).float().cuda(), torch.from_numpy(
            y).float().cuda()
        y_hat = model(x)
        # print(y_hat)
        # input('')
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        losses.append(loss.item())
    return losses


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Set')
    parser.add_argument(
        '--model',
        type=str,
        default='deep-set',
        # default='set-transformer',
        choices=['deep-set', 'set-transformer'],
        help='')
    parser.add_argument(
        '--max_iter',
        type=int,
        default='1000',
        help='')
    parser.add_argument(
        '--deepset_pooling',
        type=str,
        default='max',
        # default='mean',
        # default='sum',
        # default='min',
        choices=['max', 'mean', 'sum', 'min'],
        help='')
    parser.add_argument(
        '--data_pooling',
        type=str,
        # default='max',
        # default='mean',
        default='sum',
        # default='min',
        choices=['max', 'mean', 'sum', 'min'],
        help='')
    parser.add_argument(
        '--seed', type=int, default=1, help='the seed for training')

    args = parser.parse_args()
    _train(args)


if __name__ == '__main__':
    main()
