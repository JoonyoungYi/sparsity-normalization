import argparse
import collections
from copy import deepcopy

import torch
from tqdm import tqdm
tqdm = lambda x: x

from .dataset import load_data
from .model import LeNet
from .logger import Logger
from common.utils import set_torch_seed
from common.utils import str2bool


def __get_loss_and_accuracy(loader, model, criterion):
    count, correct, loss = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader):
            x, y = x.cuda(), y.cuda()
            y_hat = model(x)
            loss += criterion(y_hat, y).sum().item()
            _, pred = y_hat.max(1)
            count += y.shape[0]
            correct += pred.eq(y).sum().item()
    loss, accuracy = loss / count, correct / count
    return loss, accuracy


def _get_loss_and_accuracy(model, criterion, train_loader, valid_loader,
                           test_loader):
    train_loss, train_acc = __get_loss_and_accuracy(train_loader, model,
                                                    criterion)
    valid_loss, valid_acc = __get_loss_and_accuracy(valid_loader, model,
                                                    criterion)
    test_loss, test_acc = __get_loss_and_accuracy(test_loader, model,
                                                  criterion)
    return (train_loss, train_acc), (valid_loss, valid_acc), (test_loss,
                                                              test_acc)


def main(args):
    train_loader, valid_loader, test_loader = load_data(
        batch_size=args.n_batch, dataset_seed=args.dataset_seed)

    # set seed for reproducing
    set_torch_seed(args.seed)

    # get model, optimizer and criterion
    model = LeNet(
        dropout_rate=args.dropout_rate, dropout_mode=args.dropout_mode).cuda()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    logger = Logger(args)
    for epoch_idx in range(1, args.max_epoch + 1):
        print('\n>> Epoch: {}'.format(epoch_idx))

        # training phase
        model.train()
        for x, y in tqdm(train_loader):
            x, y = x.cuda(), y.cuda()
            y_hat = model(x)
            loss = criterion(y_hat, y).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation phase
        model.eval()
        result = _get_loss_and_accuracy(model, criterion, train_loader,
                                        valid_loader, test_loader)
        logger.log_epoch(epoch_idx, result)

        # model.start_batch_norm_post_statistics(train_loader)
        # print(
        #     _get_loss_and_accuracy(model, criterion, train_loader,
        #                            valid_loader, test_loader))
        # model.finish_batch_norm_post_statistics()


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser(description='LeNet')
    parser.add_argument(
        '--use_batch_normalization', type=str2bool, default=False)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument(
        '--dropout_mode',
        type=str,
        default='default',
        choices=['default', 'sparsity-normalization'])
    parser.add_argument('--n_batch', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset_seed', type=int, default=2019)
    args = parser.parse_args()

    main(args)
