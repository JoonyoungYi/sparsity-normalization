import argparse
import collections
from copy import deepcopy

import torch
from tqdm import tqdm

from .dataset import load_data
from .model import LeNet, add_momentum, init_bn
from .utils import dict2tsv, write_result_to_file
from common.utils import set_torch_seed
from common.utils import str2bool


def _get_loss_and_accuracy(loader, model, criterion):
    count, correct, loss = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            y_hat = model(x)
            loss += criterion(y_hat, y).sum().item()
            _, pred = y_hat.max(1)
            count += y.shape[0]
            correct += pred.eq(y).sum().item()
    loss, accuracy = loss / count, correct / count
    return loss, accuracy


def main(args):
    train_loader, valid_loader, test_loader = load_data(
        batch_size=args.batch_size, dataset_seed=args.dataset_seed)

    # set seed for reproducing
    set_torch_seed(args.seed)

    # get model, optimizer and criterion
    model = LeNet(
        batch_norm=args.batch_norm,
        dropout=args.dropout,
        dropout_rate=args.dropout_rate,
        dropout_mode=args.dropout_mode).cuda()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(
        reduction='none')  # QUESTION: 왜 reduction이 none일까?

    # resume training
    if args.resume:
        checkpoint = torch.load(model._get_model_path())
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']

        best_moving_stats_valid_acc = checkpoint['best_moving_stats_valid_acc']
        best_moving_stats_epoch = checkpoint['best_moving_stats_epoch']
        model_moving_stats_test_acc = checkpoint['model_moving_stats_test_acc']

        best_mini_stats_valid_acc = checkpoint['best_mini_stats_valid_acc']
        best_mini_stats_epoch = checkpoint['best_mini_stats_epoch']
        model_mini_stats_test_acc = checkpoint['model_mini_stats_test_acc']

        best_valid_acc = checkpoint['best_valid_acc']
        best_epoch = checkpoint['best_epoch']
        model_test_acc = checkpoint['model_test_acc']
    else:
        last_epoch = 0

        best_moving_stats_valid_acc = 0
        best_moving_stats_epoch = 0
        model_moving_stats_test_acc = 0

        best_mini_stats_valid_acc = 0
        best_mini_stats_epoch = 0
        model_mini_stats_test_acc = 0

        best_valid_acc = 0
        best_epoch = 0
        model_test_acc = 0

        bn1_state_dict = None
        bn2_state_dict = None

    for epoch in range(last_epoch + 1, args.epochs + 1):
        print('Epoch: {}'.format(epoch))

        # restore bn state
        if bn1_state_dict is not None and bn2_state_dict is not None:
            model.bn1.load_state_dict(bn1_state_dict)
            model.bn2.load_state_dict(bn2_state_dict)

        # training phase
        model.train()
        model.apply(add_momentum)

        total = 0
        correct = 0
        for x, y in tqdm(train_loader):
            x = x.cuda()
            y = y.cuda()

            out = model(x)
            loss = criterion(out, y).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # to restore bn state
        bn1_state_dict = deepcopy(model.bn1.state_dict())
        bn2_state_dict = deepcopy(model.bn2.state_dict())

        # validation phase
        model.eval()
        # get validation accuracy using moving average batch statistics
        _, moving_stats_valid_acc = _get_loss_and_accuracy(
            valid_loader, model, criterion)

        # get validation accuracy using mini-batch statistics
        model.apply(init_bn)
        _, mini_stats_valid_acc = _get_loss_and_accuracy(
            valid_loader, model, criterion)

        # calibration
        model.apply(init_bn)
        with torch.no_grad():
            for x, y in train_loader:
                x = x.cuda()
                y = y.cuda()

                out = model(x)

        # get train, validation, test results after calibration
        model.eval()
        # get train result to see learning curve
        train_loss, train_acc = _get_loss_and_accuracy(train_loader, model,
                                                       criterion)
        valid_loss, valid_acc = _get_loss_and_accuracy(valid_loader, model,
                                                       criterion)
        test_loss, test_acc = _get_loss_and_accuracy(test_loader, model,
                                                     criterion)

        # save the result
        res = collections.OrderedDict()
        res['epoch'] = epoch
        res['train_loss'] = train_loss
        res['valid_loss'] = valid_loss
        res['test_loss'] = test_loss
        res['train_acc'] = train_acc
        res['moving_stats_valid_acc'] = moving_stats_valid_acc
        res['mini_stats_valid_acc'] = mini_stats_valid_acc
        res['valid_acc'] = valid_acc
        res['test_acc'] = test_acc
        dict2tsv(res, 'results/{}_{}_{:.1f}_{}_acc.txt'.format(
            args.batch_norm, args.dropout, args.dropout_rate,
            args.dropout_mode))

        # track best result
        if moving_stats_valid_acc > best_moving_stats_valid_acc:
            best_moving_stats_valid_acc = moving_stats_valid_acc
            best_moving_stats_epoch = epoch
            model_moving_stats_test_acc = test_acc

        if mini_stats_valid_acc > best_mini_stats_valid_acc:
            best_mini_stats_valid_acc = mini_stats_valid_acc
            best_mini_stats_epoch = epoch
            model_mini_stats_test_acc = test_acc

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            model_test_acc = test_acc

        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'last_epoch': epoch,
                'best_moving_stats_valid_acc': best_moving_stats_valid_acc,
                'best_moving_stats_epoch': best_moving_stats_epoch,
                'model_moving_stats_test_acc': model_moving_stats_test_acc,
                'best_mini_stats_valid_acc': best_mini_stats_valid_acc,
                'best_mini_stats_epoch': best_mini_stats_epoch,
                'model_mini_stats_test_acc': model_mini_stats_test_acc,
                'best_valid_acc': best_valid_acc,
                'best_epoch': best_epoch,
                'model_test_acc': model_test_acc
            },
            model._get_model_path(), )

    # save best result
    best_result = {
        'best_moving_stats_valid_acc': best_moving_stats_valid_acc,
        'model_moving_stats_test_acc': model_moving_stats_test_acc,
        'best_moving_stats_epoch': best_moving_stats_epoch,
        'best_mini_stats_valid_acc': best_mini_stats_valid_acc,
        'model_mini_stats_test_acc': model_mini_stats_test_acc,
        'best_mini_stats_epoch': best_mini_stats_epoch,
        'best_valid_acc': best_valid_acc,
        'model_test_acc': model_test_acc,
        'best_epoch': best_epoch
    }

    write_result_to_file(args, best_result, 'results/best_result.txt')


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser(description='LeNet')
    parser.add_argument('--batch_norm', type=str2bool, default=False)
    parser.add_argument('--dropout', type=str2bool, default=False)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument(
        '--dropout_mode',
        type=str,
        choices=['default', 'sparsity-normalization'],
        default='default')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset_seed', type=int, default=2019)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.dropout:
        assert args.dropout_rate > 1e-20
    else:
        assert args.dropout_rate < 1e-20

    main(args)
