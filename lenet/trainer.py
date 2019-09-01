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


def main(args):
    train_loader, val_loader, test_loader = load_data(
        batch_size=args.batch_size, dataset_seed=args.dataset_seed)

    # set seed for reproducing
    set_torch_seed(args.seed)

    # get model, optimizer and criterion
    model = LeNet(
        batch_norm=args.batch_norm,
        dropout=args.dropout,
        dropout_rate=args.dropout_rate,
        dropout_mode=args.dropout_mode).cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # resume training
    if args.resume:
        checkpoint = torch.load('results/{}_{}_{:.1f}_{}_latest.pt'.format(
            args.batch_norm, args.dropout, args.dropout_rate,
            args.dropout_mode))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']

        best_moving_stats_val_acc = checkpoint['best_moving_stats_val_acc']
        best_moving_stats_epoch = checkpoint['best_moving_stats_epoch']
        model_moving_stats_test_acc = checkpoint['model_moving_stats_test_acc']

        best_mini_stats_val_acc = checkpoint['best_mini_stats_val_acc']
        best_mini_stats_epoch = checkpoint['best_mini_stats_epoch']
        model_mini_stats_test_acc = checkpoint['model_mini_stats_test_acc']

        best_val_acc = checkpoint['best_val_acc']
        best_epoch = checkpoint['best_epoch']
        model_test_acc = checkpoint['model_test_acc']
    else:
        last_epoch = 0

        best_moving_stats_val_acc = 0
        best_moving_stats_epoch = 0
        model_moving_stats_test_acc = 0

        best_mini_stats_val_acc = 0
        best_mini_stats_epoch = 0
        model_mini_stats_test_acc = 0

        best_val_acc = 0
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
        for src, trg in train_loader:
            src = src.cuda()
            trg = trg.cuda()

            out = model(src)
            loss = criterion(out, trg).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # to restore bn state
        bn1_state_dict = deepcopy(model.bn1.state_dict())
        bn2_state_dict = deepcopy(model.bn2.state_dict())

        # validation phase
        model.eval()

        # get validation accuracy using moving average batch statistics
        with torch.no_grad():
            total = 0
            correct = 0
            for src, trg in val_loader:
                src = src.cuda()
                trg = trg.cuda()

                out = model(src)

                _, pred = out.max(1)
                total += trg.shape[0]
                correct += pred.eq(trg).sum().item()

            moving_stats_val_acc = correct / total

        # get validation accuracy using mini-batch statistics
        model.apply(init_bn)
        with torch.no_grad():
            total = 0
            correct = 0
            for src, trg in val_loader:
                src = src.cuda()
                trg = trg.cuda()

                out = model(src)

                _, pred = out.max(1)
                total += trg.shape[0]
                correct += pred.eq(trg).sum().item()

            mini_stats_val_acc = correct / total

        # calibration
        model.apply(init_bn)
        with torch.no_grad():
            for src, trg in train_loader:
                src = src.cuda()
                trg = trg.cuda()

                out = model(src)

        # get train, validation, test results after calibration
        model.eval()
        with torch.no_grad():
            # get train result to see learning curve
            total = 0
            correct = 0
            loss = 0
            for src, trg in train_loader:
                src = src.cuda()
                trg = trg.cuda()

                out = model(src)
                loss += criterion(out, trg).sum().item()

                _, pred = out.max(1)
                total += trg.shape[0]
                correct += pred.eq(trg).sum().item()

            train_loss = loss / total
            train_acc = correct / total

            total = 0
            correct = 0
            loss = 0
            for src, trg in val_loader:
                src = src.cuda()
                trg = trg.cuda()

                out = model(src)
                loss += criterion(out, trg).sum().item()

                _, pred = out.max(1)
                total += trg.shape[0]
                correct += pred.eq(trg).sum().item()

            val_loss = loss / total
            val_acc = correct / total

            # get test result for convenience
            total = 0
            correct = 0
            loss = 0
            for src, trg in test_loader:
                src = src.cuda()
                trg = trg.cuda()

                out = model(src)
                loss += criterion(out, trg).sum().item()

                _, pred = out.max(1)
                total += trg.shape[0]
                correct += pred.eq(trg).sum().item()

            test_loss = loss / total
            test_acc = correct / total

        # save the result
        res = collections.OrderedDict()
        res['epoch'] = epoch
        res['train_loss'] = train_loss
        res['val_loss'] = val_loss
        res['test_loss'] = test_loss
        res['train_acc'] = train_acc
        res['moving_stats_val_acc'] = moving_stats_val_acc
        res['mini_stats_val_acc'] = mini_stats_val_acc
        res['val_acc'] = val_acc
        res['test_acc'] = test_acc
        dict2tsv(res, 'results/{}_{}_{:.1f}_{}_acc.txt'.format(
            args.batch_norm, args.dropout, args.dropout_rate,
            args.dropout_mode))

        # track best result
        if moving_stats_val_acc > best_moving_stats_val_acc:
            best_moving_stats_val_acc = moving_stats_val_acc
            best_moving_stats_epoch = epoch
            model_moving_stats_test_acc = test_acc

        if mini_stats_val_acc > best_mini_stats_val_acc:
            best_mini_stats_val_acc = mini_stats_val_acc
            best_mini_stats_epoch = epoch
            model_mini_stats_test_acc = test_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model_test_acc = test_acc

        # save the latest model
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'last_epoch': epoch,
            'best_moving_stats_val_acc': best_moving_stats_val_acc,
            'best_moving_stats_epoch': best_moving_stats_epoch,
            'model_moving_stats_test_acc': model_moving_stats_test_acc,
            'best_mini_stats_val_acc': best_mini_stats_val_acc,
            'best_mini_stats_epoch': best_mini_stats_epoch,
            'model_mini_stats_test_acc': model_mini_stats_test_acc,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'model_test_acc': model_test_acc
        },
                   'results/{}_{}_{:.1f}_{}_latest.pt'.format(
                       args.batch_norm, args.dropout, args.dropout_rate,
                       args.dropout_mode))

    # save best result
    best_result = {
        'best_moving_stats_val_acc': best_moving_stats_val_acc,
        'model_moving_stats_test_acc': model_moving_stats_test_acc,
        'best_moving_stats_epoch': best_moving_stats_epoch,
        'best_mini_stats_val_acc': best_mini_stats_val_acc,
        'model_mini_stats_test_acc': model_mini_stats_test_acc,
        'best_mini_stats_epoch': best_mini_stats_epoch,
        'best_val_acc': best_val_acc,
        'model_test_acc': model_test_acc,
        'best_epoch': best_epoch
    }

    write_result_to_file(args, best_result, 'results/best_result.txt')


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser(description='LeNet')
    parser.add_argument('--batch_norm', type=str2bool, default=False)
    parser.add_argument('--dropout', type=str2bool, default=False)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument(
        '--dropout_mode',
        type=str,
        choices=['default', 'sparsity-normalization'],
        default='default')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=300)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset_seed', type=int, default=2019)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    main(args)
