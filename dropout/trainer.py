import os
import random
import argparse
import pickle
import collections

from tqdm import tqdm
import numpy as np
from scipy.special import logsumexp
import torch
from sklearn.metrics import roc_auc_score

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
    # define criterion and optimizer
    if args.dataset in [
            'boston', 'diabetes', 'california', 'white-wine', 'red-wine',
            'parkinsons', 'coil-2000', 'power-plant', 'grid-stability',
            'online-news', 'protein', 'gpu'
    ]:
        criterion = torch.nn.MSELoss()
        regression = True

        if args.dataset in [
                'boston', 'diabetes', 'california', 'white-wine', 'red-wine',
                'parkinsons', 'coil-2000', 'power-plant', 'grid-stability',
                'online-news', 'protein', 'gpu'
        ]:
            n_output_unit = 1
        else:
            raise NotImplementedError()

    elif args.dataset in ['iris', 'wine']:
        # weights = [0.5, 1.0, 1.0, 1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # class_weights = torch.FloatTensor(weights).cuda()
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion = torch.nn.CrossEntropyLoss()
        regression = False

        if args.dataset == 'iris':
            n_output_unit = 3
        elif args.dataset == 'wine':
            n_output_unit = 3
        elif args.dataset == 'breast-cancer':
            n_output_unit = 1
        else:
            raise NotImplementedError()

    elif args.dataset in ['breast-cancer']:
        criterion = torch.nn.BCEWithLogitsLoss()
        n_output_unit = 1
        regression = False
    else:
        raise NotImplementedError()

    # get dataset
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(
        args.dataset,
        args.input_normalization,
        dataset_seed=args.dataset_seed,
        regression=regression)

    # Setting seed for reproducing
    set_torch_seed(args.seed)
    logger = Logger(args)

    # construct model
    model = MLP(
        n_input_unit=train_x.shape[1],
        n_output_unit=n_output_unit,
        n_hidden_unit=args.n_hidden_unit,
        n_hidden_layer=args.n_hidden_layer,
        dropout_rate=args.dropout_rate,
        dropout_mode=args.dropout_mode,
        activation_fn=args.activation_fn)
    model = model.cuda()

    if not regression and n_output_unit == 1:
        train_y = train_y.squeeze().type(torch.FloatTensor).cuda()
        valid_y = valid_y.squeeze().type(torch.FloatTensor).cuda()
        test_y = test_y.squeeze().type(torch.FloatTensor).cuda()

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

    min_valid_loss = float('inf')
    min_valid_epoch_idx = -1
    final_test_loss = float('inf')
    final_test_performance = float('inf')
    for epoch_idx in tqdm(range(1, args.max_epoch + 1)):
        # training
        model.train()

        batch_losses = []
        for i in range(0, train_x.shape[0], n_batch):
            batch_x = train_x[i:i + n_batch]
            batch_y = train_y[i:i + n_batch]
            batch_y_pred = model(batch_x)
            if not regression and n_output_unit == 1:
                batch_y = batch_y.squeeze().type(torch.FloatTensor).cuda()
                batch_y_pred = batch_y_pred.squeeze()
            batch_loss = criterion(batch_y_pred, batch_y)
            batch_losses.append([batch_loss.item(), batch_x.shape[0]])

            # backprop
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        batch_loss = sum([l * c for l, c in batch_losses]) / sum(
            c for _, c in batch_losses)

        # testing
        with torch.no_grad():
            model.eval()
            train_y_pred = model(train_x)
            if not regression and n_output_unit == 1:
                train_y_pred = train_y_pred.squeeze()
            train_loss = criterion(train_y_pred, train_y).item()
            # train_rmse = np.sqrt(train_loss.item())

            valid_y_pred = model(valid_x)
            if not regression and n_output_unit == 1:
                valid_y_pred = valid_y_pred.squeeze()
            valid_loss = criterion(valid_y_pred, valid_y).item()
            # valid_rmse = np.sqrt(valid_loss.item())

            test_y_pred = model(test_x)
            if not regression and n_output_unit == 1:
                test_y_pred = test_y_pred.squeeze()
            test_loss = criterion(test_y_pred, test_y).item()
            # test_rmse = np.sqrt(test_loss.item())

        if min_valid_loss > valid_loss:
            if regression:
                final_test_performance = np.sqrt(test_loss)
            else:
                if n_output_unit == 1:
                    final_test_performance = roc_auc_score(
                        test_y.cpu(), torch.nn.Sigmoid()(test_y_pred).cpu())
                else:
                    _, predicted = torch.max(test_y_pred.data, 1)
                    final_test_performance = (
                        predicted == test_y).sum().item() / test_y.shape[0]

            min_valid_loss = valid_loss
            min_valid_epoch_idx = epoch_idx
            final_test_loss = test_loss
            print(valid_loss, epoch_idx, test_loss, final_test_performance)

        if min_valid_epoch_idx + 1000 < epoch_idx:
            break

            # torch.save(model.state_dict(),
            #            'dropout/model-{dropout_mode}-seed{seed}.pt'.format(
            #                dropout_mode=args.dropout_mode, seed=args.seed))

        # print(epoch_idx, train_rmse, test_rmse, min_test_rmse)

        # save results
        # logger.log_epoch(epoch_idx, batch_rmse, train_rmse, valid_rmse,
        #                  test_rmse)

        # model.load_state_dict(
        #     torch.load('dropout/model-{dropout_mode}-seed{seed}.pt'.format(
        #         dropout_mode=args.dropout_mode, seed=args.seed)))
        # with torch.no_grad():
        #     model.eval()
        #     test_y_preds, masks = [], []
        #     for _ in range(1000):
        #         _test_y_pred, _mask = model(
        #             test_x, return_mask=True, test_dropout_rate=args.dropout_rate)
        #         test_y_preds.append(_test_y_pred)
        #         masks.append(_mask)
        # result = {
        #     'test_y_preds': test_y_preds,
        #     'masks': masks,
        # }
        # with open('dropout/results-{dropout_mode}-seed{seed}.pickle'.format(
        #         dropout_mode=args.dropout_mode, seed=args.seed), 'wb') as f:
        #     pickle.dump(result, f)

    print('MIN Test Error:', final_test_loss, min_valid_epoch_idx,
          final_test_performance)

    from common.utils import append_text_results

    results = [
        args.learning_rate, args.dropout_mode, args.seed, final_test_loss,
        final_test_performance
    ]
    append_text_results('dropout', args.dataset, 'results',
                        args.input_normalization + '-' +
                        ('drop%d' % (args.dropout_rate * 100)), "\t".join(
                            str(r) for r in results))

    # with torch.no_grad():
    #     model.eval()
    #
    #     std = _get_variance(model, test_x, args)
    #     print('Mean Standard Deviation:', std.mean())
    #
    # with torch.no_grad():
    #     # plot relation of sparsity and expected activation
    #     model.eval()
    #
    #     for i in range(1, 100):
    #         rate = i * 0.01
    #         test_y_pred = model(test_x, test_dropout_rate=rate)
    #         test_y_pred_mean = test_y_pred.mean().item()
    #         if i % 10 == 0:
    #             print('{:.2f} {}'.format(rate, test_y_pred_mean))
    #         logger.log_test_drop(rate, test_y_pred_mean)


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
        # default='leaky_relu',
        choices=['relu', 'leaky_relu', 'elu', 'tanh', 'sigmoid'])
    parser.add_argument('--n_hidden_layer', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument(
        '--n_batch',
        type=int,
        # default=None,
        default=128,
        help='None이면 full batch를 의미합니다.')
    parser.add_argument(
        '--n_hidden_unit',
        default=256,
        # default=512,
        # default=1024,
        type=int)


if __name__ == '__main__':
    # arguement passing
    parser = argparse.ArgumentParser(description='dropout')
    parse_arguments(parser)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset_seed', type=int, default=1)
    parser.add_argument(
        '--dataset',
        type=str,
        # default='diabetes', # regression
        # default='boston', # regression
        # default='california', # regression
        # default='white-wine',  # regression
        # default='red-wine',  # regression
        # default='parkinsons',
        # default='coil-2000',
        # default='power-plant',
        # default='grid-stability',
        # default='online-news',
        default='protein',
        # default='gpu',
        # default='iris', # classification
        # default='wine',  # classification
        # default='breast-cancer',  # binary classification
        choices=[
            'boston', 'diabetes', 'california', 'iris', 'wine', 'breast-cancer'
        ])
    parser.add_argument(
        '--input_normalization',
        type=str,
        # default='none',
        default='min-max',
        # default='standardization',
        choices=['none', 'min-max', 'standardization'])
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument(
        '--dropout_mode',
        type=str,
        default='default',
        # default='sparsity-normalization',
        choices=['default', 'sparsity-normalization'])
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=20000,
        # default=1,
    )
    parser.add_argument('--T', type=int, default=10000)

    args = parser.parse_args()

    main(args)
